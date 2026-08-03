"""회귀 학습 엔진 (사양 §15, §16).

기존 `models/engine.py` 는 상승·보합·하락 3분류용이다. CrossEntropyLoss,
argmax, softmax, 정확도·F1, 그리고 입력의 0/2번 칸을 가격으로 읽는 코드가
얽혀 있어 회귀에 그대로 쓸 수 없다. 사양 §15 가 "제거하거나 **분리**한다" 를
허용하므로, 원본은 다른 모델(TLOB 등)이 계속 쓰도록 두고 회귀는 여기로 뺐다.

특히 `engine.py:109` 의
    mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2)
는 입력 0번 칸이 가격이라는 가정인데, 우리 입력의 0번 칸은 bid_of_l1 —
정규화된 주문흐름이다. 그대로 두면 흐름 값을 가격으로 착각한다. 가격은
전처리 때 저장해 둔 _px.npy 에서 백테스트가 따로 읽는다.

손실 스케일
-----------
수익률은 1e-4 규모라 MSE 가 1e-8 규모가 된다. Adam 의 기본 eps 가 1e-8 이라
분모에서 기울기와 같은 크기가 되어 갱신이 눌린다. 그래서 손실만 bp 단위
(x 1e4) 로 재서 O(1) 로 만든다. 상수배이므로 최적해는 그대로이고, 로그에
찍히는 숫자도 읽을 수 있게 된다. 성적 지표는 원래 소수 단위로 되돌려 계산한다.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from lightning import LightningModule
from torch import nn

import ofi_spec as spec
from utils.metrics import format_table, summarise

LOSS_SCALE = 1e4          # 소수 수익률 -> bp. 손실을 O(1) 로 만들기 위한 상수배


def build_loss(loss_type: str) -> nn.Module:
    """사양 §16. 기본 mse, huber 선택 가능."""
    key = (loss_type or spec.LOSS_TYPE).lower()
    if key == "mse":
        return nn.MSELoss()
    if key in ("huber", "smoothl1"):
        return nn.SmoothL1Loss()
    raise ValueError(f"모르는 손실함수: {loss_type!r}. 'mse' 또는 'huber'")


class RegressionEngine(LightningModule):
    """[B, SEQ, 22] -> [B, 10] 회귀 학습.

    검증·시험 dataloader 를 여러 개 넘기면 dataloader_idx 로 구분해 종목별
    성적을 따로 낸다. 합산 점수만 보면 한 종목이 벌고 다른 종목이 까먹는 걸
    놓친다.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        optimizer_name: str = "Adam",
        loss_type: str = spec.LOSS_TYPE,
        weight_decay: float = 0.0,
        eval_names: list[str] | None = None,
        ckpt_dir: str | None = None,
        horizons=tuple(spec.TARGET_HORIZONS_SEC),
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.eval_names = list(eval_names or ["all"])
        self.ckpt_dir = ckpt_dir
        self.horizons = list(horizons)

        self.criterion = build_loss(loss_type)
        self.save_hyperparameters(ignore=["model"])

        self._train_losses: list[float] = []
        self._buffers: dict[int, dict[str, list]] = {}
        self.best_val = float("inf")
        self.best_ckpt_path: str | None = None
        self.last_report: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def loss(self, pred, target):
        """bp 단위로 재서 계산. 상수배라 최적해는 같다."""
        return self.criterion(pred * LOSS_SCALE, target * LOSS_SCALE)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self._train_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self._train_losses:
            mean = torch.stack(self._train_losses).mean().item()
            print(f"[epoch {self.current_epoch}] train loss {mean:.5f}")
        self._train_losses.clear()

    # ------------------------------------------------------------------
    def _eval_step(self, batch, dataloader_idx: int):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        buf = self._buffers.setdefault(dataloader_idx, {"p": [], "t": [], "l": []})
        buf["p"].append(pred.detach().float().cpu())
        buf["t"].append(y.detach().float().cpu())
        buf["l"].append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._eval_step(batch, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._eval_step(batch, dataloader_idx)

    # ------------------------------------------------------------------
    def _finish_eval(self, stage: str):
        if not self._buffers:
            return

        report = {}
        for idx in sorted(self._buffers):
            buf = self._buffers[idx]
            name = self.eval_names[idx] if idx < len(self.eval_names) else f"set{idx}"
            pred = torch.cat(buf["p"]).numpy()
            target = torch.cat(buf["t"]).numpy()
            mean_loss = torch.stack(buf["l"]).mean().item()

            s = summarise(pred, target, self.horizons)
            s["loss"] = mean_loss
            report[name] = s

            self.log(f"{stage}_loss/{name}", mean_loss, add_dataloader_idx=False)
            self.log(f"{stage}_ic/{name}", float(np.nanmean(s["ic"])),
                     add_dataloader_idx=False)
            self.log(f"{stage}_dir/{name}", float(np.nanmean(s["dir_acc"])),
                     add_dataloader_idx=False)

        # 조기 종료와 체크포인트는 첫 dataloader(합산 검증셋) 기준
        primary = report[self.eval_names[0]] if self.eval_names[0] in report \
            else report[next(iter(report))]
        self.log(f"{stage}_loss", primary["loss"], prog_bar=True,
                 add_dataloader_idx=False)

        self._print_report(stage, report)
        self.last_report = report
        self._buffers.clear()

        if stage == "val" and primary["loss"] < self.best_val:
            self.best_val = primary["loss"]
            self._save_checkpoint(primary["loss"])

    def on_validation_epoch_end(self):
        self._finish_eval("val")

    def on_test_epoch_end(self):
        self._finish_eval("test")

    def _print_report(self, stage: str, report: dict):
        print()
        for name, s in report.items():
            head = f"[{stage}] {name}  손실 {s['loss']:.5f}  " \
                   f"평균IC {np.nanmean(s['ic']):+.4f}  " \
                   f"평균 방향정확도 {100*np.nanmean(s['dir_acc']):.2f}%"
            print(format_table(s, head))
            print()

    # ------------------------------------------------------------------
    def _save_checkpoint(self, loss: float):
        if not self.ckpt_dir:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(
            self.ckpt_dir, f"epoch{self.current_epoch}_val{loss:.5f}.ckpt")
        try:
            self.trainer.save_checkpoint(path)
        except Exception as exc:                       # noqa: BLE001
            print(f"체크포인트 저장 실패: {exc}")
            return
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            os.remove(self.best_ckpt_path)             # 최고 성적 하나만 남긴다
        self.best_ckpt_path = path
        print(f"체크포인트 저장: {path}")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        name = self.optimizer_name.lower()
        if name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
        elif name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        elif name == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9,
                                  weight_decay=self.weight_decay)
        else:
            raise ValueError(f"모르는 optimizer: {self.optimizer_name!r}")

        # 원본 engine.py 는 val_loss 개선폭을 절대값 0.002 와 비교해 학습률을
        # 반씩 깎았다. 분류 손실(약 1.0)에 맞춘 값이라 회귀에서는 매 epoch
        # 무조건 깎인다. 상대 개선폭을 보는 표준 스케줄러로 바꾼다.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=1, threshold=1e-3,
            threshold_mode="rel")
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss",
                             "interval": "epoch"},
        }
