"""예측의 크기를 맞춘다 (배율 보정).

    python scripts/calibrate.py --ckpt data/checkpoints/.../epochN_valX.ckpt

방향은 맞는데 크기가 어긋나면 회귀로서는 반쪽이다. R2 는 방향과 크기를 함께
보므로, 크기가 어긋난 만큼 그대로 깎인다.

    R2 <= IC^2      (등호는 크기가 정확히 맞았을 때)

MSE 학습은 원래 크기를 맞추는 방향으로 수렴하지만, 덜 학습되면 예측이 과하게
크거나 작은 채로 남는다. 실측(epoch 1)에서 1초는 이론값의 48%, 10초는 8%
수준이었다.

여기서는 horizon 마다 배율 하나만 최소제곱으로 구한다.

    최종 예측 = 모델 출력 x alpha

alpha 는 **검증 구간에서만** 구하고 시험 구간에는 적용만 한다. 시험 데이터로
배율을 맞추면 시험 문제를 보고 답을 고치는 셈이 된다.

절편(상수항)은 기본적으로 넣지 않는다. 수익률에 상수를 더하는 건 "항상 조금
오른다"는 방향성 편향을 심는 것이고, 학습 구간의 추세가 시험 구간에 이어진다는
보장이 없다. --with-intercept 로 켤 수는 있다.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                        # noqa: E402
import torch                                                   # noqa: E402
from models.regression_engine import load_model_from_checkpoint  # noqa: E402
from preprocessing.ofi_dataset import load_normalizer, split_dates  # noqa: E402
from scripts.backtest import predict_day                       # noqa: E402
from scripts.download_data import FILE_IDS, WEEKDAYS           # noqa: E402
from utils.metrics import (                                    # noqa: E402
    fit_scale,
    information_coefficient,
    r2,
)


def gather(model, cache, symbols, dates, norm, device):
    """여러 종목·날짜의 (예측, 정답) 을 모아 [N, 10] 두 개로 돌려준다."""
    preds, targets = [], []
    for sym in symbols:
        for date in dates:
            got = predict_day(model, cache, sym, date, norm, device)
            if got is None:
                continue
            idx, p, _ = got
            y = np.load(os.path.join(cache, sym, f"{date}_y.npy"), mmap_mode="r")
            preds.append(p)
            targets.append(np.asarray(y[idx], dtype=np.float64))
    if not preds:
        return None, None
    return np.concatenate(preds), np.concatenate(targets)


def report(pred, target, alpha, beta, title):
    ic = information_coefficient(pred, target)
    before = r2(pred, target)
    after = r2(pred * alpha + beta, target)

    print(title)
    print(f"{'horizon':>8}{'IC':>9}{'IC^2':>10}{'R2 (전)':>11}{'R2 (후)':>11}"
          f"{'배율 a':>10}{'회수율':>9}")
    print("-" * 70)
    for j, h in enumerate(spec.TARGET_HORIZONS_SEC):
        ideal = ic[j] ** 2
        rec = after[j] / ideal * 100 if ideal > 0 else float("nan")
        print(f"{h:>6}초{ic[j]:>9.4f}{ideal:>10.5f}{before[j]:>11.5f}"
              f"{after[j]:>11.5f}{alpha[j]:>10.3f}{rec:>8.0f}%")
    print(f"\n평균 R2  {before.mean():.5f}  ->  {after.mean():.5f}"
          f"   ({after.mean()/before.mean() if before.mean() != 0 else float('nan'):.2f}배)\n")
    return before, after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--with-intercept", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--out", default=None, help="기본값: <cache>/calibration.json")
    args = ap.parse_args()

    _, val_dates, test_dates = split_dates(args.dates, args.n_val, args.n_test)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"체크포인트 {args.ckpt}")
    print(f"배율 학습 {val_dates} (검증)   적용·평가 {test_dates} (시험)")
    print(f"장치 {device}\n")

    model, _ = load_model_from_checkpoint(args.ckpt, map_location=device)
    model = model.to(device).eval()
    norm = {} if args.no_normalize else load_normalizer(args.cache)

    vp, vt = gather(model, args.cache, args.symbols, val_dates, norm, device)
    if vp is None:
        print("검증 데이터가 없습니다.", file=sys.stderr)
        return 1
    alpha, beta = fit_scale(vp, vt, args.with_intercept)
    report(vp, vt, alpha, beta, f"=== 검증셋 (배율을 여기서 구함) — {vp.shape[0]:,}개 ===")

    tp, tt = gather(model, args.cache, args.symbols, test_dates, norm, device)
    if tp is not None:
        report(tp, tt, alpha, beta,
               f"=== 시험셋 (배율 적용만) — {tp.shape[0]:,}개 ===")

        print("=== 시험셋 종목별 ===")
        for sym in args.symbols:
            sp, st = gather(model, args.cache, [sym], test_dates, norm, device)
            if sp is None:
                continue
            b, a = r2(sp, st), r2(sp * alpha + beta, st)
            ic = information_coefficient(sp, st)
            print(f"  {sym:<6} 평균IC {ic.mean():+.4f}   "
                  f"평균R2 {b.mean():.5f} -> {a.mean():.5f}")
        print()

    out = args.out or os.path.join(args.cache, "calibration.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"horizons_sec": spec.TARGET_HORIZONS_SEC,
                   "alpha": alpha.tolist(), "beta": beta.tolist(),
                   "fitted_on": val_dates, "checkpoint": args.ckpt}, fh, indent=2)
    print(f"저장: {out}")
    print("\n회수율 = 실제 R2 / 이론상 최대(IC^2). 100% 면 크기가 완전히 맞은 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
