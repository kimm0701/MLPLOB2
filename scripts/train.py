"""범용(pooled) 학습 실행 (사양 §14~§17).

    python scripts/train.py
    python scripts/train.py --hidden-dim 64 --num-layers 4 --max-epochs 5
    python scripts/train.py --held-out META     # META 를 학습에서 빼고 시험만

종목 3개를 한 덩어리로 합쳐 모델 **1개**를 학습한다. 모델에는 종목이 뭔지
알려주지 않는다 — 입력은 22개 숫자뿐이다. 그래야 종목을 외우지 못하고 모든
종목에 공통인 흐름 패턴을 배우게 된다.

검증은 세 겹으로 본다.
  1) 합친 검증셋 전체
  2) 종목별로 쪼갠 성적 — 한 종목만 잘하는 편향인지 확인
  3) --held-out 으로 학습에 안 쓴 종목 시험 — 범용성의 가장 강한 증거
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as cst                                        # noqa: E402
from utils.lightning_compat import (                           # noqa: E402
    EarlyStopping,
    TQDMProgressBar,
    Trainer,
    seed_everything,
)
import ofi_spec as spec                                        # noqa: E402
from models.mlplob import MLPLOB                               # noqa: E402
from models.regression_engine import RegressionEngine          # noqa: E402
from preprocessing.ofi_dataset import (                        # noqa: E402
    build_split,
    target_scales,
    describe,
    split_dates,
)
from scripts.download_data import FILE_IDS, WEEKDAYS           # noqa: E402


def make_loader(ds, batch_size, shuffle, workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--held-out", default=None,
                    help="이 종목은 학습·검증에서 빼고 시험에만 쓴다")
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)

    ap.add_argument("--hidden-dim", type=int, default=128,
                    help="원본 config.py 의 hyperparameters_sweep 값. "
                         "hyperparameters_fixed 의 40 은 BTC 전용 덮어쓰기이고, "
                         "논문 주력 데이터셋(FI-2010)은 144 를 쓴다")
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="원본은 32~128. 배치가 크면 epoch 당 가중치 갱신 "
                         "횟수가 그만큼 줄어 과소적합으로 이어진다")
    ap.add_argument("--optimizer", default="Adam")
    # tune.py 가 찾아낸 설정을 그대로 붙여넣을 수 있게 Optuna 파라미터 이름
    # (loss_type) 을 그대로 쓴 별칭도 받는다.
    ap.add_argument("--loss", "--loss-type", dest="loss",
                    default=spec.LOSS_TYPE, choices=["mse", "huber"])
    ap.add_argument("--weight-decay", type=float, default=0.0)

    # 논문(Kolm et al. 2023, Table 3)과 동일: 50 epoch, 조기종료 patience 5.
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5,
                    help="학습 샘플 간격. 5 면 0.25초마다 하나. 이웃 샘플은 "
                         "100칸 중 99칸이 겹치지만, 20 으로 두면 데이터의 "
                         "1/20 만 쓰게 되어 갱신 횟수가 크게 부족해진다. "
                         "검증·시험은 항상 전부(1) 쓴다")
    ap.add_argument("--limit-train-batches", type=float, default=1.0)
    ap.add_argument("--limit-val-batches", type=float, default=1.0,
                    help="빠른 점검용. 1.0 미만이면 검증을 일부만 돌린다")
    ap.add_argument("--limit-test-batches", type=float, default=1.0)
    ap.add_argument("--ckpt-dir", default=None)
    # 정답 정규화 (논문 §3.2.2). 학습 정답만 자르고 평가 정답은 원본을 쓴다 —
    # 평가까지 자르면 문제가 쉬워져서 성적이 부풀려진다.
    ap.add_argument("--normalize-target", action="store_true",
                    help="정답을 종목·horizon 별 표준편차로 나눈다")
    ap.add_argument("--winsorize", action="store_true",
                    help="학습 정답의 상하위 0.5%%를 경계값으로 자른다 "
                         "(--normalize-target 필요). 평가 정답은 자르지 않는다")
    ap.add_argument("--no-rolling", action="store_true",
                    help="구 방식(bp 정답, 고정 배율)으로 되돌린다")
    ap.add_argument("--no-normalize", action="store_true",
                    help="종목별 사전 정규화를 끄고 비교해 볼 때")
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def thin(ds, stride: int):
    """학습셋만 솎아낸다. 검증·시험은 건드리지 않는다."""
    if stride <= 1:
        return ds
    for part in ds.datasets:
        part.indices = part.indices[::stride]
    ds.cumulative_sizes = ds.cumsum(ds.datasets)
    return ds


def main() -> int:
    args = build_args()
    seed_everything(args.seed, workers=True)

    train_dates, val_dates, test_dates = split_dates(
        args.dates, args.n_val, args.n_test)
    train_syms = [s for s in args.symbols if s != args.held_out]
    if not train_syms:
        print("학습할 종목이 없습니다.", file=sys.stderr)
        return 1

    normalize = not args.no_normalize
    print(f"학습 종목 {train_syms}   날짜 {train_dates[0]}~{train_dates[-1]}")
    print(f"검증 {val_dates}   시험 {test_dates}")
    if args.held_out:
        print(f"제외 종목 {args.held_out} — 학습에 쓰지 않고 시험만 한다")
    print(f"종목별 사전 정규화: {'켜짐' if normalize else '꺼짐'}\n")

    kw = dict(normalize=normalize, normalize_target=args.normalize_target,
              rolling=not args.no_rolling)
    # 자르기는 학습에만 적용한다
    train_ds = thin(build_split(args.cache, train_syms, train_dates,
                                winsorize=args.winsorize, **kw), args.stride)
    val_all = build_split(args.cache, train_syms, val_dates, **kw)
    print(f"학습셋   {describe(train_ds)}   (간격 {args.stride})")
    print(f"검증셋   {describe(val_all)}\n")

    # 종목별 dataloader 만 둔다. 합산 성적은 엔진이 이어붙여 계산하므로
    # 합산용 loader 를 따로 두면 같은 데이터를 두 번 훑게 된다.
    val_names = list(train_syms)
    val_loaders = [make_loader(build_split(args.cache, [s], val_dates, **kw),
                               args.batch_size, False, args.workers)
                   for s in train_syms]

    # 시험셋은 설정이 확정될 때까지 만들지 않는다. 파일이 없으면 조용히
    # 건너뛴다 - 정책을 코드가 강제하게 두는 편이 낫다.
    test_syms = args.symbols                       # 제외 종목도 시험에는 포함
    test_names, test_loaders = [], []
    for s_ in test_syms:
        try:
            test_loaders.append(make_loader(
                build_split(args.cache, [s_], test_dates, **kw),
                args.batch_size, False, args.workers))
            test_names.append(s_)
        except FileNotFoundError:
            pass
    if not test_loaders:
        print("시험셋 없음 - 검증셋으로만 학습한다 (의도된 상태)")

    model_config = dict(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_size=spec.SEQ_LEN,
        num_features=spec.INPUT_DIM,
        dataset_type="OFI",
    )
    model = MLPLOB(**model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 {n_params:,}개\n")

    ckpt_dir = args.ckpt_dir or os.path.join(
        cst.DIR_SAVED_MODEL, "MLPLOB_OFI",
        f"h{args.hidden_dim}_l{args.num_layers}_lr{args.lr}_{args.loss}"
        + (f"_no{args.held_out}" if args.held_out else ""))

    engine = RegressionEngine(
        model=model,
        lr=args.lr,
        optimizer_name=args.optimizer,
        loss_type=args.loss,
        weight_decay=args.weight_decay,
        eval_names=val_names,
        ckpt_dir=ckpt_dir,
        model_config=model_config,
        target_scale_by_name=(target_scales(args.cache, test_names)
                              if args.normalize_target else None),
        # 롤링 경로에서는 배율이 배치에 실려 오므로 종목 상수를 두지 않는다
        target_normalized=not args.no_rolling,
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=args.max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min",
                          patience=args.patience, verbose=True),
            TQDMProgressBar(refresh_rate=200),
        ],
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        logger=False,
        enable_checkpointing=False,       # RegressionEngine 이 직접 저장한다
    )
    # 학습셋만 섞는다. 날짜로 이미 잘라놨으므로 섞어도 미래가 새지 않는다.
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    trainer.fit(engine, train_loader, val_loaders)

    print("\n" + "=" * 60)
    print("최종시험 (여기서 처음이자 마지막으로 시험 날짜를 쓴다)")
    print("=" * 60)
    if test_loaders:
        engine.eval_names = test_names
        trainer.test(engine, test_loaders)
    else:
        print("시험 단계 생략 - 시험셋 파일을 만들지 않았다 (의도된 상태)")

    if engine.best_ckpt_path:
        print(f"\n최고 성적 모델: {engine.best_ckpt_path}")
        print("다음: python scripts/backtest.py --ckpt " + engine.best_ckpt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
