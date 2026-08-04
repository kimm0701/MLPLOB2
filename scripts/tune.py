"""Optuna 로 하이퍼파라미터를 탐색한다 (가지치기 포함).

    pip install optuna
    python scripts/tune.py --trials 25

한 번 학습에 시간이 오래 걸리므로 가지치기(pruning)를 켠다. 몇 epoch 돌려보고
지금까지의 중앙값보다 나쁘면 그 시도를 중간에 끊는다. 전체 탐색 시간이 보통
3~5배 줄어든다.

지키는 것
---------
* 검증 성적만 보고 고른다. **최종시험 날짜는 건드리지 않는다.** 튜닝에 시험
  데이터를 쓰면 시험 문제를 미리 보고 공부한 셈이 되어 성적이 부풀려진다.
* 사양 고정값(SEQ_LEN, BUCKET_MS, INPUT_DIM, OUTPUT_DIM, OFI_WINDOWS,
  horizon)은 탐색 대상이 아니다.
* stride 도 탐색하지 않는다. 데이터를 많이 쓸수록 검증 성적이 좋아지므로
  Optuna 는 무조건 제일 촘촘한 값을 고르고, 그러면 속도를 위해 stride 를 둔
  의미가 사라진다. 시도끼리 채점 조건도 달라져 비교가 성립하지 않는다.
* 시도 횟수를 너무 늘리지 않는다. 검증셋에 대해서까지 과최적화된다.
"""

import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                        # noqa: E402
from utils.lightning_compat import (                           # noqa: E402
    Callback,
    EarlyStopping,
    Trainer,
)
from models.mlplob import MLPLOB                               # noqa: E402
from models.regression_engine import RegressionEngine          # noqa: E402
from preprocessing.ofi_dataset import build_split, split_dates  # noqa: E402
from scripts.download_data import FILE_IDS, WEEKDAYS           # noqa: E402


class _FallbackPruningCallback(Callback):
    """공식 통합 패키지가 없을 때 쓰는 최소 구현.

    하는 일은 공식과 같다 — epoch 마다 검증 손실을 Optuna 에 보고하고,
    지금까지의 중앙값보다 나쁘면 그 시도를 중간에 끊는다.
    """

    def __init__(self, trial, monitor: str = "val_ic"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        import optuna

        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"epoch {trainer.current_epoch} 에서 가지치기 "
                f"({self.monitor}={float(value):.5f})")


def make_pruning_callback(trial, monitor: str = "val_ic"):
    """가지치기 콜백. 공식 통합 패키지가 있으면 그걸 쓴다.

        pip install optuna-integration

    공식 쪽은 lightning / pytorch_lightning 중 한쪽 Trainer 만 받아들이도록
    타입 검사를 하는 판본이 있어서, 이 저장소처럼 두 배포판을 모두 지원하는
    환경에서는 실패할 수 있다. 그래서 실패하면 같은 동작의 자체 구현으로
    넘어간다 — 탐색 자체가 멈추는 것보다 낫다.
    """
    try:
        from optuna_integration.pytorch_lightning import (
            PyTorchLightningPruningCallback,
        )
        return PyTorchLightningPruningCallback(trial, monitor=monitor), "공식"
    except Exception:                                   # noqa: BLE001
        pass
    try:
        from optuna.integration import PyTorchLightningPruningCallback
        return PyTorchLightningPruningCallback(trial, monitor=monitor), "공식(구버전)"
    except Exception:                                   # noqa: BLE001
        return _FallbackPruningCallback(trial, monitor), "자체"


def make_loader(ds, batch_size, shuffle, workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def thin(ds, stride):
    if stride > 1:
        for p in ds.datasets:
            p.indices = p.indices[::stride]
        ds.cumulative_sizes = ds.cumsum(ds.datasets)
    return ds


# 실측 기준값. hidden_dim 40 / batch 1024 로 2,428 배치를 404초에 처리했고
# (초당 6,158 샘플), 검증은 배치당 약 4배 빨랐다. 탐색 범위 중앙인 hidden 128
# 은 파라미터가 2.8배라 보수적으로 3배 느리다고 본다.
# 상수 예측 등으로 IC 가 정의되지 않을 때 줄 점수. 실제 IC 는 -1 아래로
# 내려갈 수 없으므로 어떤 정상 시도보다도 나쁘다.
BAD_SCORE = -1.0

TRAIN_SAMPLES_PER_SEC = 2_000
VAL_SAMPLES_PER_SEC = 8_000


def _print_budget(args, n_train, n_val):
    """돌리기 전에 예상 시간을 보여준다.

    한 번 학습이 얼마나 걸리는지 모르고 시작하면, 탐색이 하루를 넘겨도 중간에
    알 수가 없다. 배치 개수가 아니라 **샘플 수**로 환산한다 — 배치 크기를
    바꾸면 배치당 시간도 같이 변해서 배치 수로는 비교가 안 된다.
    """
    train_n = n_train * args.limit_train_batches
    val_n = n_val * args.limit_val_batches
    per_epoch = train_n / TRAIN_SAMPLES_PER_SEC + val_n / VAL_SAMPLES_PER_SEC
    per_trial = per_epoch * args.max_epochs
    # 가지치기로 상당수가 조기 종료되므로 전체는 단순 곱보다 짧다
    total_hi = per_trial * args.trials
    total_lo = total_hi * 0.45

    print()
    print(f"예상 소요  1 epoch 약 {per_epoch/60:.0f}분  "
          f"(학습 {train_n/TRAIN_SAMPLES_PER_SEC/60:.0f}분 + "
          f"검증 {val_n/VAL_SAMPLES_PER_SEC/60:.0f}분)")
    print(f"           1 시도 {per_trial/60:.0f}분,  "
          f"{args.trials}회 전체 {total_lo/3600:.1f} ~ {total_hi/3600:.1f}시간")
    if total_hi / 3600 > 8:
        print("           ! 길다. --train-days 를 줄이거나 --stride 를 키우거나 "
              "--limit-val-batches 를 낮추세요.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=30,
                    help="탐색 중에는 더 성글게 뽑아 한 시도를 빨리 끝낸다")
    ap.add_argument("--train-days", type=int, default=4,
                    help="0 이면 학습 날짜 전부. 줄이면 한 시도가 빨라진다")
    ap.add_argument("--limit-train-batches", type=float, default=1.0,
                    help="한 시도의 학습 분량. 0.3 이면 30%만 돌린다")
    ap.add_argument("--limit-val-batches", type=float, default=0.1,
                    help="한 시도의 검증 분량. 탐색 중에는 일부만 봐도 순위가 갈린다")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--study", default="mlplob_ofi")
    ap.add_argument("--storage", default=None,
                    help="예: sqlite:///data/optuna.db  중단 후 이어서 하려면")
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        print("optuna 가 없습니다.  pip install optuna", file=sys.stderr)
        return 1

    train_dates, val_dates, test_dates = split_dates(
        args.dates, args.n_val, args.n_test)
    if args.train_days:
        train_dates = train_dates[-args.train_days:]

    print(f"학습 {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}일)")
    print(f"검증 {val_dates}")
    print(f"시험 {test_dates}  <- 탐색에 쓰지 않는다\n")

    train_ds = thin(build_split(args.cache, args.symbols, train_dates), args.stride)
    val_ds = build_split(args.cache, args.symbols, val_dates)
    print(f"학습 {len(train_ds):,}샘플 (간격 {args.stride})   검증 {len(val_ds):,}샘플\n")

    def objective(trial):
        hidden = trial.suggest_categorical("hidden_dim", [64, 128, 144, 192, 256])
        layers = trial.suggest_int("num_layers", 2, 6)
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [128, 256, 512])
        loss = trial.suggest_categorical("loss_type", ["mse", "huber"])
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        # 오차 몇 bp 까지를 신호로 볼지. huber 일 때만 쓰인다.
        beta = trial.suggest_float("huber_beta", 1.0, 10.0) if loss == "huber" else 1.0

        pruner_cb, kind = make_pruning_callback(trial)
        if trial.number == 0:
            print(f"가지치기 콜백: {kind}")

        cfg = dict(hidden_dim=hidden, num_layers=layers, seq_size=spec.SEQ_LEN,
                   num_features=spec.INPUT_DIM, dataset_type="OFI")
        engine = RegressionEngine(model=MLPLOB(**cfg), lr=lr, loss_type=loss,
                                  weight_decay=wd, eval_names=["all"],
                                  model_config=cfg, pooled_name=None,
                                  huber_beta=beta)

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=args.max_epochs,
            callbacks=[pruner_cb,
                       EarlyStopping(monitor="val_ic", mode="max", patience=1)],
            num_sanity_val_steps=0,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(engine,
                    make_loader(train_ds, batch, True, args.workers),
                    make_loader(val_ds, batch, False, args.workers))

        # 채점은 IC 로 한다.
        #  - 학습 손실로 채점하면 huber 가 mse 보다 항상 작은 값을 내서
        #    (실측 2.08 대 20.80) 성능과 무관하게 huber 만 뽑힌다.
        #  - val_mse 로 채점하면 반대로 mse 로 학습한 쪽이 유리하다. 자기가
        #    최적화한 지표로 채점받기 때문이다.
        #  - IC 는 어느 손실로 학습했든 공평하고, R2 <= IC^2 이므로 R2 의
        #    천장을 직접 올린다. 크기는 학습 후 배율 보정으로 맞춘다.
        value = trainer.callback_metrics.get("val_ic")
        if value is None or not math.isfinite(float(value)):
            # 예측이 상수로 무너지면 상관계수가 정의되지 않아 NaN 이 된다.
            # 실측: weight_decay 8.6e-3 이 모델을 눌러 이 상태를 만들었다.
            # NaN 을 그대로 돌려주면 Optuna 가 시도를 FAIL 로 버리고 아무것도
            # 배우지 못해 같은 영역을 다시 뽑는다. 나쁜 점수로 기록해야
            # TPE 가 그 근처를 피한다.
            print(f"  Trial {trial.number}: 예측이 상수로 무너짐 "
                  f"(wd={wd:.2e}) -> 최하점 처리")
            return BAD_SCORE
        return float(value)

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",   # IC 는 클수록 좋다
        sampler=optuna.samplers.TPESampler(seed=1),
        # 3 epoch 까지는 지켜보고, 그 뒤로 중앙값보다 나쁘면 끊는다
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.trials, catch=(RuntimeError,))

    print("\n" + "=" * 60)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    print(f"완료 {len(done)} / 가지치기 {len(pruned)} / 전체 {len(study.trials)}")
    print(f"\n최고 검증 손실 {study.best_value:.5f}")
    print("최적 설정:")
    for k, v in study.best_params.items():
        print(f"    {k:<14} {v}")

    cmd = " ".join(f"--{k.replace('_','-')} {v}" for k, v in study.best_params.items())
    print(f"\n이 설정으로 전체 학습:\n    python scripts/train.py {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
