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

출력 단위
---------
모델은 **bp(0.01%) 단위로 예측한다**. 정답만 1e4 를 곱해 올리고 예측은 그대로
쓴다.

처음에는 예측과 정답에 둘 다 1e4 를 곱했는데, 그건 손실 숫자만 키울 뿐
모델이 실제로 내놓아야 할 값은 여전히 0.0002 같은 극소값으로 남긴다. 신경망은
초기화 직후 0.1~1 규모를 출력하므로 3~4자릿수를 줄이는 데 학습을 다 쓴다.
실측으로 첫 epoch 손실이 3467 bp^2 (RMSE 59bp) 에서 시작했다 — 정답 크기
2bp 의 30배다. 2 epoch 을 돌고도 5.5bp 로, 아직 "평균값 답하기" 수준에도
도달하지 못했고 IC 는 0 이었다.

정답만 올리면 초기 출력(약 1)과 정답(약 2bp)의 크기가 맞아 곧바로 학습이
시작된다. 성적 지표와 백테스트는 예측을 다시 1e4 로 나눠 소수 수익률로
되돌려 쓴다 (사양 §13 의 저장 형식은 소수 그대로다).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from utils.lightning_compat import LightningModule

import ofi_spec as spec
from utils.metrics import format_table, summarise

TARGET_SCALE = 1e4        # 소수 수익률 -> bp. 모델은 bp 로 예측한다


def load_model_from_checkpoint(path: str, map_location="cpu"):
    """체크포인트에서 모델만 복원한다 (추론·백테스트용).

    LightningModule.load_from_checkpoint 는 __init__ 인자를 그대로 다시 넣어
    객체를 만드는데, 여기서는 model 이 객체라 저장돼 있지 않다. 그래서
    저장해 둔 model_config 로 구조를 다시 세우고 가중치만 얹는다.
    """
    from models.mlplob import MLPLOB

    ck = torch.load(path, map_location=map_location, weights_only=False)
    hp = ck.get("hyper_parameters", {})
    cfg = dict(hp.get("model_config") or {})
    if not cfg:
        raise ValueError(
            f"{path} 에 model_config 가 없습니다. 이 체크포인트는 구조 정보를 "
            "저장하기 전 버전입니다. 다시 학습하세요."
        )

    model = MLPLOB(**cfg)
    state = {k[len("model."):]: v for k, v in ck["state_dict"].items()
             if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"가중치가 맞지 않습니다. 없음={missing} 남음={unexpected}")
    return model.eval(), hp


def build_loss(loss_type: str, huber_beta: float = 1.0) -> nn.Module:
    """사양 §16. 기본 mse, huber 선택 가능.

    huber_beta 는 "어디까지를 신호로 보고 어디부터를 잡음으로 볼지"의 경계다.
    손실을 bp 단위로 재므로 beta 도 bp 다. beta 아래의 오차는 제곱으로(=MSE
    처럼), 위는 직선으로(=MAE 처럼) 벌점을 매긴다.

    기본값 1.0 은 우리 데이터에 너무 작다. 오차가 대개 2~5bp 라 거의 전 구간이
    직선이 되어 사실상 MAE 로 동작한다. 정답의 표준편차가 1.6~4.9bp 이므로
    그 2~3배 근처가 경계로 적당하고, tune.py 가 이 값을 탐색한다.
    """
    key = (loss_type or spec.LOSS_TYPE).lower()
    if key == "mse":
        return nn.MSELoss()
    if key in ("huber", "smoothl1"):
        return nn.SmoothL1Loss(beta=float(huber_beta))
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
        model_config: dict | None = None,
        pooled_name: str | None = "all",
        huber_beta: float = 1.0,
        target_scale_by_name: dict | None = None,
    ):
        super().__init__()
        self.model = model
        # 모델 객체 자체는 하이퍼파라미터로 저장할 수 없다. 대신 생성 인자를
        # 남겨서 나중에 체크포인트만으로 같은 구조를 다시 만들 수 있게 한다.
        # 이게 없으면 백테스트가 체크포인트를 열 수 없다.
        self.model_config = dict(model_config or {})
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.loss_type = loss_type
        self.huber_beta = huber_beta
        self.weight_decay = weight_decay
        self.eval_names = list(eval_names or ["all"])
        # 종목별 결과를 이어붙여 만들 합산 항목의 이름. None 이면 안 만든다.
        self.pooled_name = pooled_name
        self.ckpt_dir = ckpt_dir
        self.horizons = list(horizons)
        # 종목별 정답 표준편차 (소수 단위). 있으면 정답이 이미 표준화된 것이므로
        # 손실에 추가 배율을 걸지 않고, 지표 계산 전에 곱해서 되돌린다.
        self.target_scale_by_name = dict(target_scale_by_name or {})
        self.loss_scale = 1.0 if self.target_scale_by_name else TARGET_SCALE

        self.criterion = build_loss(loss_type, huber_beta)
        self.save_hyperparameters(ignore=["model"])

        self._train_losses: list[float] = []
        self._buffers: dict[int, dict[str, list]] = {}
        self._raw: dict[str, dict] = {}
        self.best_val = float("inf")
        self.best_ckpt_path: str | None = None
        self.last_report: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def loss(self, pred, target):
        """손실은 정답과 같은 단위에서 잰다.

        정답 정규화를 켜면 Dataset 이 이미 표준편차로 나눠서 O(1) 로 주므로
        추가 배율이 필요 없다. 끄면 정답이 소수 수익률(2e-4 규모)이라 그대로
        두면 MSE 가 1e-8 이 되어 Adam 의 eps 에 눌린다. 그때만 bp 로 올린다.
        """
        return self.criterion(pred, target * self.loss_scale)

    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(batch):
        """(x, y) 또는 (x, y, scale). scale 은 예측을 원래 수익률로 되돌릴 배율."""
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def training_step(self, batch, batch_idx):
        x, y, _ = self._unpack(batch)
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
        x, y, scale = self._unpack(batch)
        pred = self(x)
        loss = self.loss(pred, y)
        buf = self._buffers.setdefault(dataloader_idx,
                                       {"p": [], "t": [], "l": [], "s": []})
        # 여기서는 모델이 낸 그대로 담고, 단위 환원은 _finish_eval 에서 한 번에
        # 한다. 롤링 정규화에서는 배율이 **표본마다** 다르다.
        buf["p"].append(pred.detach().float().cpu())
        buf["t"].append(y.detach().float().cpu())
        buf["l"].append(loss.detach())
        if scale is not None:
            buf["s"].append(scale.detach().float().cpu())
        return loss

    def _to_decimal_rolling(self, pred, target, scale):
        """표본별 배율로 되돌린다.  z * scale = 소수 수익률.

        scale = sigma * level * tick / price 이고 sigma 는 과거만 보고 만든
        값이라, 되돌린 정답은 **자르지 않은 원본 수익률**과 정확히 같다.
        성적이 부풀려지지 않는다.
        """
        return pred * scale, target * scale

    def _to_decimal(self, pred, target, name: str):
        """지표 계산 전에 둘 다 소수 수익률로 되돌린다 (사양 §13).

        정답 정규화를 켰다면 Dataset 이 정답을 그 종목·horizon 의 표준편차로
        나눠서 줬으므로, 곱해서 되돌린다. **평가 정답은 자르지 않았으므로**
        되돌린 값이 곧 원본 수익률이다 — 성적이 부풀려지지 않는다.
        """
        sc = self.target_scale_by_name.get(name)
        if sc is not None:
            s = np.asarray(sc, dtype=np.float64)
            return pred * s, target * s
        return pred / TARGET_SCALE, target

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
            p = torch.cat(buf["p"]).numpy().astype(np.float64)
            t = torch.cat(buf["t"]).numpy().astype(np.float64)
            if buf.get("s"):
                sc = torch.cat(buf["s"]).numpy().astype(np.float64)
                pred, target = self._to_decimal_rolling(p, t, sc)
            else:
                pred, target = self._to_decimal(p, t, name)
            mean_loss = torch.stack(buf["l"]).mean().item()

            s = summarise(pred, target, self.horizons)
            s["loss"] = mean_loss
            report[name] = s
            self._raw[name] = {"pred": pred, "target": target}

            self.log(f"{stage}_loss/{name}", mean_loss, add_dataloader_idx=False)
            self.log(f"{stage}_ic/{name}", float(np.nanmean(s["ic"])),
                     add_dataloader_idx=False)
            self.log(f"{stage}_dir/{name}", float(np.nanmean(s["dir_acc"])),
                     add_dataloader_idx=False)
            self.log(f"{stage}_r2/{name}", float(np.nanmean(s["r2_cal"])),
                     add_dataloader_idx=False)
            # 학습에 쓴 손실함수와 무관한 지표. huber 와 mse 는 같은 예측에도
            # 값이 10배 다르게 나오므로, 설정끼리 비교하려면 공통 자가 필요하다.
            self.log(f"{stage}_mse/{name}", float(np.nanmean(s["mse"])) * 1e8,
                     add_dataloader_idx=False)

        # 합산 성적은 종목별 결과를 이어붙여 계산한다. 합산용 dataloader 를
        # 따로 두면 같은 데이터를 두 번 훑어 검증 시간이 두 배가 된다.
        if len(report) > 1 and self.pooled_name:
            all_pred = np.concatenate([b["pred"] for b in self._raw.values()])
            all_target = np.concatenate([b["target"] for b in self._raw.values()])
            s = summarise(all_pred, all_target, self.horizons)
            s["loss"] = float(np.mean([r["loss"] for r in report.values()]))
            report = {self.pooled_name: s, **report}
            self.log(f"{stage}_loss/{self.pooled_name}", s["loss"],
                     add_dataloader_idx=False)
            self.log(f"{stage}_ic/{self.pooled_name}",
                     float(np.nanmean(s["ic"])), add_dataloader_idx=False)

        # 조기 종료와 체크포인트는 합산 성적 기준
        primary = report[next(iter(report))]
        self.log(f"{stage}_loss", primary["loss"], prog_bar=True,
                 add_dataloader_idx=False)
        # bp^2 단위. 손실함수 종류가 달라도 이 값끼리는 비교된다.
        self.log(f"{stage}_mse", float(np.nanmean(primary["mse"])) * 1e8,
                 add_dataloader_idx=False)
        self.log(f"{stage}_ic", float(np.nanmean(primary["ic"])),
                 add_dataloader_idx=False)
        # 회귀 성적의 본 지표. 보정 전은 출력 크기가 어긋난 만큼 깎이므로
        # 설정끼리 비교할 때는 보정 후를 쓴다 (utils.metrics.r2_calibrated).
        self.log(f"{stage}_r2_raw", float(np.nanmean(primary["r2"])),
                 add_dataloader_idx=False)
        self.log(f"{stage}_r2", float(np.nanmean(primary["r2_cal"])),
                 prog_bar=True, add_dataloader_idx=False)

        self._print_report(stage, report)
        self.last_report = report
        self._buffers.clear()
        self._raw.clear()

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
