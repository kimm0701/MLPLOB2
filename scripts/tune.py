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


class PruningCallback(Callback):
    """epoch 마다 검증 손실을 Optuna 에 보고하고, 가망 없으면 끊는다."""

    def __init__(self, trial, monitor: str = "val_loss"):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max-epochs", type=int, default=4)
    ap.add_argument("--stride", type=int, default=10,
                    help="탐색 중에는 더 성글게 뽑아 한 시도를 빨리 끝낸다")
    ap.add_argument("--train-days", type=int, default=0,
                    help="0 이면 학습 날짜 전부. 줄이면 한 시도가 빨라진다")
    ap.add_argument("--limit-train-batches", type=float, default=1.0,
                    help="한 시도의 학습 분량. 0.3 이면 30%만 돌린다")
    ap.add_argument("--limit-val-batches", type=float, default=0.25,
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

        cfg = dict(hidden_dim=hidden, num_layers=layers, seq_size=spec.SEQ_LEN,
                   num_features=spec.INPUT_DIM, dataset_type="OFI")
        engine = RegressionEngine(model=MLPLOB(**cfg), lr=lr, loss_type=loss,
                                  weight_decay=wd, eval_names=["all"],
                                  model_config=cfg, pooled_name=None)

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=args.max_epochs,
            callbacks=[PruningCallback(trial),
                       EarlyStopping(monitor="val_loss", mode="min", patience=1)],
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

        value = trainer.callback_metrics.get("val_loss")
        return float(value) if value is not None else float("inf")

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="minimize",
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
