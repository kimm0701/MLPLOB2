"""틱 단위 정답과 인과적 변동성을 만든다.

    python scripts/make_targets.py

전처리를 다시 하지 않는다. 저장해 둔 _px.npy (미드/매수/매도) 만 읽어서
정답을 새로 만든다.

왜 bp 가 아니라 틱인가
----------------------
bp 는 가격으로 나누므로 가격이 움직이면 같은 사건의 라벨이 밀린다.
실측: 14일 동안 AMD -19.6%, MRVL -24.0%, META -10.9% 움직였다. 그만큼
라벨 눈금이 표류한다. 틱은 거래소가 고정한 값이라 이 문제가 없다.
논문(Kolm et al. 2023 §3.2.1)도 bp 가 아니라 달러를 쓴다 - 한 종목 안에서
달러는 틱의 상수배다.

왜 고정 sigma 가 아니라 롤링인가
--------------------------------
하루 안에서 변동성이 크게 뭉친다. 실측(30분 구간 48개): AMD 16.3배,
META 13.9배, MRVL 19.0배. 고정 sigma 하나로 나누면 잔잔한 구간은 손실에
거의 기여하지 못하고 요동 구간이 학습을 지배한다.

sigma 는 **과거만** 본다. 시점 t 의 sigma 는 t 까지의 수익률만 쓴다.

horizon 을 왜 종목마다 다르게 두는가
-----------------------------------
Delta_t = "가격이 한 번 변하는 데 걸리는 평균 시간" (Kolm 식 16). 종목마다
다르다(AMD 0.26초 / MRVL 0.36초 / META 0.58초). 시계 초로 고정하면 빠른
종목은 먼 미래를, 느린 종목은 가까운 미래를 보게 되어 같은 출력 칸이
종목마다 다른 뜻이 된다. Delta_t 의 배수로 두면 "가격변화 k 회 뒤" 로
통일된다.
"""

import argparse
import json
import os
import sys

import numpy as np


def cache_dir(a):
    return a.cache


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                    # noqa: E402
from preprocessing.ofi_dataset import split_dates          # noqa: E402
from scripts.download_data import FILE_IDS, WEEKDAYS       # noqa: E402

# horizon 은 ofi_spec.TARGET_HORIZONS_SEC (절대 초) 를 쓴다
EWMA_HALFLIFE_SEC = 300                 # 초. 종목마다 이벤트 수로 환산한다.
# 버킷이 시간에서 이벤트로 바뀌면서 "1200 버킷" 이 종목마다 다른 시간이 됐다
# (AMD 86초 / MRVL 68초 / META 166초). 초로 지정해야 세 종목이 같은 구간을 본다.
# 후보 비교(구간분산 변동계수, 낮을수록 좋음): 10s 1.220 / 30s 1.118 /
# 60s 1.069 / 120s 1.044 / 300s 1.021 / 900s 1.007 / 고정 1.892.
# 120~900초가 평평하므로 가운데인 300초를 쓴다 - 경계값을 피하고, 짧을수록
# 구간 변화에 빨리 적응한다.
MIN_SIGMA_PCT = 5.0                     # sigma 하한 (그날 분포의 하위 %)
WINSOR_PCT = 0.5                        # 학습 정답을 자를 상하위 % (논문과 동일)


def infer_tick(cache: str, symbol: str, dates) -> float:
    """매수호가의 0 이 아닌 최소 변화폭 = 틱.

    미드로 재면 안 된다. (매수+매도)/2 라 반틱 단위로 움직인다.
    """
    best = None
    for date in dates:
        f = os.path.join(cache, symbol, date + "_px.npy")
        if not os.path.exists(f):
            continue
        bid = np.asarray(np.load(f, mmap_mode="r")[:, 1], dtype=np.float64)
        bid = bid[np.isfinite(bid) & (bid > 0)]
        d = np.abs(np.diff(bid))
        d = d[d > 1e-12]
        if d.size:
            t = float(np.min(d))
            best = t if best is None else min(best, t)
    if best is None:
        return 0.0
    # 검증: 모든 호가가 틱의 정수배여야 한다
    f = os.path.join(cache, symbol, dates[0] + "_px.npy")
    bid = np.asarray(np.load(f, mmap_mode="r")[:, 1], dtype=np.float64)
    bid = bid[np.isfinite(bid) & (bid > 0)]
    if not np.allclose(bid / best, np.round(bid / best), atol=1e-6):
        print(f"  경고: {symbol} 의 호가가 틱 {best} 의 정수배가 아니다",
              file=sys.stderr)
    return best


def measure_dt(cache: str, symbol: str, dates) -> float:
    """Delta_t: 미드가격이 한 번 변하는 데 걸리는 평균 시간(초)."""
    chg = tot = 0
    for date in dates:
        f = os.path.join(cache, symbol, date + "_px.npy")
        if not os.path.exists(f):
            continue
        m = np.asarray(np.load(f, mmap_mode="r")[:, 0], dtype=np.float64)
        m = m[np.isfinite(m)]
        chg += int((np.diff(m) != 0).sum())
        tot += m.size
    if not chg:
        return float("nan")
    return tot * spec.BUCKET_MS / 1000.0 / chg


def _apply_floor(sig: np.ndarray, halflife: int) -> np.ndarray:
    """0 에 가까운 sigma 로 나누는 것을 막되, **미래를 보지 않고** 막는다.

    예전에는 하루 전체 sigma 분포의 하위 5% 를 하한으로 썼다. 그 한 숫자에
    미래가 섞여 있었다 - 단위 테스트가 "뒤쪽 절반을 바꾸면 앞쪽 sigma 가
    변한다" 로 잡아냈다.

    이제 하한을 **워밍업 구간(앞쪽 halflife 개)** 에서만 구한다. 그 구간은
    valid_sample_indices 가 어차피 버리므로, 실제로 쓰는 모든 시점에 대해
    하한은 과거 정보다.
    """
    w = min(max(halflife, 100), sig.size)
    floor = float(np.percentile(sig[:w], MIN_SIGMA_PCT)) if w else 0.0
    return np.maximum(sig, max(floor, 1e-6))


def targets_by_seconds(mid, ts_ms, tick, horizons_sec):
    """t 에서 t+h 초 사이의 미드 변화 (틱).  [K, len(horizons_sec)]

    이벤트 버킷이라 배열 인덱스와 시간이 비례하지 않는다. ts_ms 로 시각을
    찾아야 한다. 하루 끝을 넘어가는 시점은 NaN.
    """
    n = mid.size
    y = np.full((n, len(horizons_sec)), np.nan, dtype=np.float32)
    for j, h in enumerate(horizons_sec):
        k = np.searchsorted(ts_ms, ts_ms + h * 1000.0, side="left")
        ok = k < n
        y[ok, j] = ((mid[k[ok]] - mid[ok]) / tick).astype(np.float32)
    return y


def causal_sigma_sec(mid, ts_ms, tick, base_sec, halflife):
    """**base_sec 초 구간** 움직임의 인과적 EWMA 표준편차.

    정답의 기준 horizon 과 같은 구간으로 재야 단위가 맞는다. 시점 t 의 값은
    t 까지 **완료된** 구간만 쓴다 - r[i] 는 (i - base 구간) 의 움직임이므로
    미래를 보지 않는다.
    """
    n = mid.size
    back = np.searchsorted(ts_ms, ts_ms - base_sec * 1000.0, side="left")
    r = np.zeros(n, dtype=np.float64)
    good = back < np.arange(n)
    d = (mid - mid[np.minimum(back, n - 1)]) / tick
    r[good] = np.where(np.isfinite(d[good]), d[good], 0.0)

    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1))
    warm = min(halflife, n)
    seed = float(np.mean(r[:warm] ** 2)) if warm else 1.0
    acc = seed if np.isfinite(seed) and seed > 0 else 1.0
    var = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc += alpha * (r[i] * r[i] - acc)
        var[i] = acc
    sig = np.sqrt(np.maximum(var, 1e-12))
    return _apply_floor(sig, halflife)


def build_day(cache: str, symbol: str, date: str, tick: float,
              horizons_buckets, halflife_sec: float):
    """하루치 정답(틱)과 sigma 를 만들어 저장한다.

    sigma 는 1Δt 기준 하나만 저장한다. horizon j 의 크기는 읽을 때
    sigma * sqrt(h_j / h_1) 로 만든다 - 저장 용량을 3배로 늘릴 이유가 없다.
    """
    base = os.path.join(cache, symbol, date)
    if not os.path.exists(base + "_px.npy"):
        return None
    px = np.asarray(np.load(base + "_px.npy", mmap_mode="r"), dtype=np.float64)
    mid = px[:, 0]
    n = mid.size
    # 미드에 구멍이 있으면 앞의 값으로 메운다. 안 그러면 EWMA 가 그 뒤로 전부
    # NaN 이 된다 (실측: AMD 하루치가 통째로 NaN 이 됐다).
    bad = ~np.isfinite(mid)
    if bad.any():
        idx = np.maximum.accumulate(np.where(~bad, np.arange(n), 0))
        mid = mid[idx]

    ts = np.load(base + "_ts.npy").astype(np.float64)
    # 반감기(초) -> 이벤트 수. 종목·날짜마다 이벤트 속도가 다르다.
    span = max((ts[-1] - ts[0]) / 1000.0, 1.0)
    halflife = max(50, int(round(halflife_sec * ts.size / span)))
    y = targets_by_seconds(mid, ts, tick, spec.TARGET_HORIZONS_SEC)
    y[bad] = np.nan                        # 원래 구멍이던 시점은 무효 처리

    # sigma 도 가장 짧은 horizon 과 같은 구간으로 잰다
    sig = causal_sigma_sec(mid, ts, tick,
                           spec.TARGET_HORIZONS_SEC[0], halflife).astype(np.float32)

    np.save(base + "_yt.npy", y)          # [K, 3]  틱 단위, 정규화 전
    np.save(base + "_sg.npy", sig)        # [K]     1Δt 인과적 sigma
    return y, sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=spec.N_VAL)
    ap.add_argument("--n-test", type=int, default=spec.N_TEST)
    ap.add_argument("--halflife-sec", type=float, default=EWMA_HALFLIFE_SEC,
                    help=f"EWMA 반감기(초). 기본 {EWMA_HALFLIFE_SEC}초. "
                         "종목별 이벤트 속도로 환산한다")
    ap.add_argument("--include-test", action="store_true",
                    help="시험일 파일도 만든다. 기본은 만들지 않는다")
    args = ap.parse_args()

    train, val, test = split_dates(args.dates, args.n_val, args.n_test)
    print(f"틱·Delta_t 산출 구간: 학습 {len(train)}일 ({train[0]}~{train[-1]})")
    print(f"검증 {val}   시험 {test}   <- 산출에 쓰지 않는다")
    print(f"EWMA 반감기 {args.halflife_sec:.0f}초 (종목별 이벤트 수로 환산)")
    print(f"horizon {spec.TARGET_HORIZONS_SEC}초\n")

    meta = {}
    for sym in args.symbols:
        tick = infer_tick(args.cache, sym, train)
        dt = measure_dt(args.cache, sym, train)
        if not tick or not np.isfinite(dt):
            print(f"{sym}: 데이터 없음, 건너뜀")
            continue
        hb = list(spec.TARGET_HORIZONS_SEC)
        print(f"[{sym}]  틱 {tick:g}   가격변화 1회당 평균 {dt:.3f}초")
        print(f"   horizon = {hb} 초  (절대 시간, 종목 공통)")

        targets = list(train) + list(val) + (list(test) if args.include_test else [])
        made = 0
        for date in targets:
            if build_day(args.cache, sym, date, tick, hb, args.halflife_sec):
                made += 1
        # 롤링 sigma 가 시간 변동을 잡고 나면 종목별 **수준** 차이가 남는다.
        # (반감기 1분 실측: AMD 1.11 / META 1.38 / MRVL 1.07 - 25% 차이)
        # 짧은 창일수록 EWMA 가 앞선 값이라 실현분산을 과소평가하는데 그 정도가
        # 종목마다 다르기 때문이다. 학습 구간에서 상수 하나로 정확히 없앤다.
        vs = []
        for date in train:
            b = os.path.join(cache_dir(args), sym, date)
            if not os.path.exists(b + "_yt.npy"):
                continue
            yy = np.load(b + "_yt.npy")[:, 0]
            ss = np.load(b + "_sg.npy")
            v = np.nanvar(yy / ss)
            if np.isfinite(v) and v > 0:
                vs.append(v)
        # 중앙값을 쓴다. 학습 10일의 정규화 후 분산이 0.03~7.53 로 흔들려서
        # 평균을 쓰면 극단적인 하루가 보정값을 통째로 끌고 간다 (실측).
        lvl = float(np.sqrt(np.median(vs))) if vs else 1.0

        # winsorize 경계를 **정규화된 z 공간**에서 잡는다. 모델이 실제로 보는
        # 값이라 "몇 표준편차에서 자른다" 로 해석되고, 틱/bp 같은 단위 혼동이
        # 원천적으로 없다. 학습 날짜에서만 구한다.
        zs = []
        for date in train:
            b = os.path.join(args.cache, sym, date)
            if not os.path.exists(b + "_yt.npy"):
                continue
            zz = np.load(b + "_yt.npy") / (np.load(b + "_sg.npy")[:, None] * lvl)
            zs.append(zz)
        clip_lo = clip_hi = None
        if zs:
            z = np.concatenate(zs)
            ok = np.isfinite(z).all(axis=1)
            clip_lo, clip_hi = np.percentile(
                z[ok], [WINSOR_PCT, 100 - WINSOR_PCT], axis=0)
            keep = 1 - np.clip(z[ok], clip_lo, clip_hi).var(0) / z[ok].var(0)
            print(f"   winsorize {WINSOR_PCT}%  경계 "
                  f"{np.round(clip_lo, 2).tolist()} ~ {np.round(clip_hi, 2).tolist()}"
                  f"   분산 {np.round(keep * 100, 1).tolist()}% 제거")

        meta[sym] = dict(tick=tick, delta_t_sec=dt,
                         horizons_sec=list(spec.TARGET_HORIZONS_SEC),
                         horizon_buckets=hb, halflife_sec=args.halflife_sec,
                         level=lvl, fitted_on=list(train),
                         winsor_pct=WINSOR_PCT,
                         clip_lo=None if clip_lo is None else clip_lo.tolist(),
                         clip_hi=None if clip_hi is None else clip_hi.tolist())
        print(f"   종목 수준 보정 {lvl:.4f}  (1Dt 분산을 정확히 1 로 맞춤)")
        print(f"   {made}일 저장\n")

    if not meta:
        print("만든 것이 없습니다.", file=sys.stderr)
        return 1

    out = os.path.join(args.cache, "targets.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"저장: {out}")

    print("\n검증일 정규화 후 분산  (기대 1 / 2 / 3, 종목 간에도 같아야 함)")
    print(f"{'종목':<7}{'1dt':>9}{'2dt':>9}{'3dt':>9}")
    print("-" * 36)
    for sym in meta:
        d = os.path.join(args.cache, sym, val[-1])
        if not os.path.exists(d + "_yt.npy"):
            continue
        y = np.load(d + "_yt.npy")
        sg = np.load(d + "_sg.npy")
        v = np.nanvar(y / (sg[:, None] * meta[sym]["level"]), axis=0)
        print(f"{sym:<7}{v[0]:>9.2f}{v[1]:>9.2f}{v[2]:>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
