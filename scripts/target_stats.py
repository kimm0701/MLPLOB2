"""학습·검증·시험 구간의 정답 분산을 잰다.

손실만 보고 과적합인지 과소적합인지 판단하면 틀린다. 구간마다 정답의 변동폭이
다르기 때문이다. 변동이 큰 날이 검증에 걸리면 손실이 커 보이는 게 당연하고,
그건 모델 문제가 아니다.

    R2 = 1 - 손실 / 정답분산

이 분산을 알아야 손실 숫자를 R2 로 옮겨 비교할 수 있다.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                        # noqa: E402
from preprocessing.ofi_dataset import split_dates              # noqa: E402
from scripts.download_data import FILE_IDS, WEEKDAYS           # noqa: E402


def variance(cache, symbols, dates, step=50):
    chunks = []
    for sym in symbols:
        for d in dates:
            base = os.path.join(cache, sym, d)
            if not os.path.exists(base + "_y.npy"):
                continue
            y = np.load(base + "_y.npy", mmap_mode="r")
            idx = np.load(base + "_idx.npy")
            if idx.size:
                chunks.append(np.asarray(y[idx[::step]], dtype=np.float64))
    if not chunks:
        return None
    a = np.concatenate(chunks) * 1e4          # bp
    return np.nanvar(a, axis=0), a.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--loss", nargs="*", type=float, default=None,
                    help="관측된 손실 (학습 검증 시험 순). 주면 R2 로 환산해 준다")
    args = ap.parse_args()

    tr, va, te = split_dates(args.dates, args.n_val, args.n_test)
    print(f"{'구간':<6}{'날짜수':>7}{'표본':>12}   horizon별 정답분산 (bp^2)")
    print("-" * 78)

    means = {}
    for name, dates in [("학습", tr), ("검증", va), ("시험", te)]:
        got = variance(args.cache, args.symbols, dates)
        if got is None:
            print(f"{name:<6}  데이터 없음")
            continue
        var, n = got
        means[name] = float(var.mean())
        head = "  ".join(f"{v:6.1f}" for v in var[:5])
        print(f"{name:<6}{len(dates):>7}{n:>12,}   {head} ...   평균 {var.mean():6.2f}")

    if args.loss:
        print(f"\n{'구간':<6}{'손실':>10}{'정답분산':>10}{'R2':>10}")
        print("-" * 40)
        for (name, _), loss in zip([("학습", tr), ("검증", va), ("시험", te)], args.loss):
            if name in means:
                r2 = 1 - loss / means[name]
                print(f"{name:<6}{loss:>10.2f}{means[name]:>10.2f}{r2:>10.5f}")
        print("\nR2 가 0 근처면 '평균값 답하기' 수준, 음수면 그보다 못함.")
        print("학습 R2 도 0 근처면 과적합이 아니라 과소적합이다.")
    else:
        print("\n손실을 함께 주면 R2 로 환산해 준다:")
        print("    python scripts/target_stats.py --loss 13.56 26.10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
