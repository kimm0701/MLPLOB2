"""종목별 사전 정규화 기준값을 학습 구간에서만 계산한다 (사양 §17).

    python scripts/fit_normalizer.py
    python scripts/fit_normalizer.py --symbols MRVL AMD META NVDA   # 종목 추가

깊이 정규화(§6, §9)를 거쳤는데도 특징의 퍼짐 정도가 종목마다 크게 다르다.
2026-07-30 실측: 전체 표준편차는 2.5배, L10 매수흐름은 **68배** 차이가 났다
(MRVL 0.31 / AMD 1.29 / META 21.22). 이대로 한 솥에 부으면 모델이 흐름 패턴
대신 "이 스케일이면 META" 라는 종목 정체성을 외운다.

퍼짐 정도로 (p99 - p1) / 4.652 를 쓴다. 정규분포에서 그 구간 폭이 4.652 표준
편차에 해당한다는 관계를 빌리되, 양 끝 2% 를 빼므로 극단값에 흔들리지 않는다
(META 최대 10,063 같은 값). 평균·표준편차를 그냥 쓰면 그런 값 하나에 기준이
끌려가 정작 대부분의 값이 0 근처로 눌린다.

IQR 은 쓸 수 없다. 이 특징은 80~89% 가 정확히 0 이라 (이벤트가 없던 버킷)
가운데 50% 구간이 통째로 0 이고, 그대로 나누면 값이 100만 배로 튄다.
MRVL 실측: IQR 기준 0.0000 대 (p99-p1) 기준 0.9503.

기준값은 **학습 날짜에서만** 계산한다. 검증·시험 날짜가 섞이면 미래를 미리 본
셈이 된다. 종목마다 독립으로 계산하므로 나중에 종목을 추가해도 기존 값은
다시 구할 필요가 없다.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                   # noqa: E402
from preprocessing.ofi_dataset import split_dates         # noqa: E402
from scripts.download_data import WEEKDAYS, FILE_IDS      # noqa: E402

P_LO, P_HI = 1.0, 99.0    # 양 끝 1% 씩 잘라 극단값의 영향을 배제
RANGE_TO_SIGMA = 4.652    # 정규분포에서 p99 - p1 = 4.652 x 표준편차
MIN_SCALE = 1e-6          # 상수 특징에서 0 으로 나누는 것을 막는다

# 정답 쪽 기준값 (Kolm et al. 2023 §3.2.2)
T_LO, T_HI = 0.5, 99.5    # winsorize 경계. 논문과 동일
MIN_T_SCALE = 1e-9        # 수익률 표준편차의 하한 (소수 단위)


# 정답 쪽 기준값(winsorize 경계·표준편차)은 scripts/make_targets.py 가 만든다.
# 정답이 bp·고정 horizon 에서 틱·절대 초로 바뀌면서 이 파일이 다룰 게
# 없어졌다. 여기는 입력 22 개의 종목별 center/scale 만 담당한다.


def fit_symbol(cache: str, symbol: str, train_dates, max_rows: int = 2_000_000):
    """한 종목의 (중앙값, 퍼짐 정도) 를 22개 특징 각각에 대해 구한다."""
    per_day = max(1, max_rows // max(len(train_dates), 1))
    chunks = []

    for date in train_dates:
        base = os.path.join(cache, symbol, date)
        if not os.path.exists(base + "_x.npy"):
            continue
        x = np.load(base + "_x.npy", mmap_mode="r")
        idx = np.load(base + "_idx.npy")
        if idx.size == 0:
            continue
        # 유효 시점에서 고르게 뽑는다. 하루를 통째로 올리면 메모리가 감당 안 된다.
        step = max(1, idx.size // per_day)
        chunks.append(np.asarray(x[idx[::step]], dtype=np.float64))

    if not chunks:
        return None

    sample = np.concatenate(chunks, axis=0)
    center = np.median(sample, axis=0)
    hi, lo = np.percentile(sample, [P_HI, P_LO], axis=0)
    raw_scale = (hi - lo) / RANGE_TO_SIGMA
    scale = np.maximum(raw_scale, MIN_SCALE)

    degenerate = [spec.OF_FEATURE_NAMES[j] for j in np.flatnonzero(raw_scale <= MIN_SCALE)]

    return dict(
        center=center.tolist(),
        scale=scale.tolist(),
        n_rows=int(sample.shape[0]),
        n_days=len(chunks),
        train_dates=list(train_dates),
        degenerate_features=degenerate,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--out", default=None, help="기본값: <cache>/normalizer.json")
    args = ap.parse_args()

    train_dates, val_dates, test_dates = split_dates(args.dates, args.n_val, args.n_test)
    print(f"학습 {len(train_dates)}일 ({train_dates[0]}~{train_dates[-1]})  "
          f"검증 {val_dates}  시험 {test_dates}")
    print("검증·시험 날짜는 기준값 계산에 쓰지 않는다\n")

    out_path = args.out or os.path.join(args.cache, "normalizer.json")
    stats = {}
    if os.path.exists(out_path):          # 기존 종목 값은 보존하고 덮어쓰기만
        with open(out_path, encoding="utf-8") as fh:
            stats = json.load(fh).get("symbols", {})

    print(f"{'종목':<7}{'표본행':>12}{'일수':>6}   특징별 퍼짐 정도 (일부)")
    print("-" * 72)
    for sym in args.symbols:
        st = fit_symbol(args.cache, sym, train_dates)
        if st is None:
            print(f"{sym:<7}  전처리 결과 없음, 건너뜀")
            continue
        stats[sym] = st
        if st["degenerate_features"]:
            print(f"{'':<7}  경고: 퍼짐이 0 인 특징 {st['degenerate_features']}")
        sc = np.asarray(st["scale"])
        print(f"{sym:<7}{st['n_rows']:>12,}{st['n_days']:>6}   "
              f"L1:{sc[0]:.3f}  L10:{sc[9]:.3f}  ofi500:{sc[20]:.3f}")

    if not stats:
        print("\n계산된 기준값이 없습니다. scripts/preprocess.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 1

    payload = dict(
        feature_names=spec.OF_FEATURE_NAMES,
        method="p1_p99_range",
        range_to_sigma=RANGE_TO_SIGMA,
        symbols=stats,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"\n저장: {out_path}  (종목 {len(stats)}개: {', '.join(sorted(stats))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
