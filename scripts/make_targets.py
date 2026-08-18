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
MIN_SIGMA_PCT = 5.0                     # 전날 분포의 하위 % 를 하한으로 쓴다
# 종목·시점과 무관한 물리적 하한 (틱 단위).
#
# 미드가 틱의 1/100 도 안 움직이는 구간의 '변동성' 은 뜻이 없다. 예전 값
# 1e-6 은 0 나눗셈만 막자는 의도였는데 너무 작아서, 조용한 구간 뒤 첫 움직임
# 하나가 |z| 를 수십억까지 밀어 올렸다 (실측: 첫날 최대 7.79e9).
# winsorize 가 학습은 막아 주고 채점은 되돌릴 때 상쇄되지만, 그 표본들이
# '최대 강도 정답' 으로 학습에 들어가고 분산 통계가 못 쓰게 된다.
ABS_MIN_SIGMA = 0.01
# 호가창 유효성 상한 (틱). 이보다 벌어진 스냅샷은 시장 상태가 아니라 캡처 결함으로 본다.
#
# 실측(최근 6일): 진짜 넓은 스프레드는 p99.99 기준 32~36틱이다. 그런데 캡처
# 시작 직후 한쪽만 갱신된 스냅샷에서 13,143틱(META 20260714 idx4)이 나왔다 -
# 매수는 521.76 에 멈춰 있고 매도만 653.19 로 들어온 상태였다. 0.02초 뒤
# 652.19 로 정상화된다.
#
# 100틱은 p99.99 의 약 3배 밖이라 걸리는 비율이 0.0002~0.0004% 다. 다만
# 100~326틱 구간이 전부 결함인지는 개별 확인하지 않았다.
#
# **실전에도 같은 규칙을 건다.** 스프레드가 이만큼 벌어지면 데이터가 깨진
# 것이므로 예측을 멈추고 호가를 내지 않는다. 학습에서만 빼는 조작이 아니다.
MAX_SPREAD_TICKS = 100.0

# ---- 일중 주기 계수 -------------------------------------------------------
#
# 변동성은 하루 안에서 예측 가능한 모양을 그린다. 실측(1분 해상도, 학습 17일):
# 미국 개장 13:30 UTC 에 하루 평균의 6~10배로 뛰고 30분에 걸쳐 내려온다.
# 최저(21:00 무렵 0.15배)와 비교하면 40~55배 차이다.
#
# EWMA 는 이걸 못 따라간다. 반감기가 개장 기준 약 27초라 6배 점프를 90% 따라
# 잡는 데 90초가 걸리는데, 급첨 자체가 1분 만에 끝난다. 정점을 지나서야 반영된다.
#
# 그래서 두 층으로 나눈다 (Andersen & Bollerslev 1997, Engle & Sokalska 2012).
#
#     sigma(t) = c(t) x EWMA(r/c)
#
#     c(t)  모양. 달력에서 온다. 예측이 아니라 시계를 보는 것
#     EWMA  높이. 오늘 이 종목이 평소보다 센지 약한지
#
# **표는 과거 날짜에서만 만든다.** 워크포워드 검증 결과(로그 RMSE):
#     개장 13:30~14:30   계수없음 1.040 -> 과거10일 0.244   76% 감소
#     하루 전체          계수없음 0.724 -> 과거10일 0.544   25% 감소
# 과거 10일이면 포화라(17일과 0.0006 차이) 창을 15일로 둔다.
DIURNAL_BINS = 1440             # 1분 해상도. 시간 단위로 뭉개면 급첨이 사라진다
DIURNAL_WINDOW = 15             # 롤링 창 (거래일)
DIURNAL_MIN_DAYS = 3            # 이보다 적으면 계수를 쓰지 않는다 (c=1)
DIURNAL_MIN_COUNT = 100         # 한 칸에 이만큼은 있어야 값을 믿는다
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


def day_floor(sig: np.ndarray) -> float:
    """그날 sigma 분포의 하위 MIN_SIGMA_PCT %.  **다음 날에 쓸 하한**이다."""
    sig = sig[np.isfinite(sig)]
    return float(np.percentile(sig, MIN_SIGMA_PCT)) if sig.size else 0.0


def _apply_floor(sig: np.ndarray, floor: float | None) -> np.ndarray:
    """0 에 가까운 sigma 로 나누는 것을 막되, **미래를 보지 않고** 막는다.

    하한은 **전날** 분포에서 구한 값을 받는다. 실시간에서도 그대로 재현된다 -
    장이 열릴 때 어제 값은 이미 알고 있다.

    첫날은 전날이 없으므로 절대 하한 1e-6 만 쓴다. 하루치를 버리는 것보다
    낫고, 어차피 학습 구간의 맨 앞이다.

    이전 두 판본이 모두 틀렸다.
      1) 하루 전체 분포의 하위 5%  - 명백히 미래를 봤다
      2) 그날 앞쪽 halflife 개 구간 - "그 구간은 어차피 학습에서 버린다" 는
         전제로 넣었는데, 실측해 보니 표본은 인덱스 199 부터 시작하고 그 창은
         3,400~5,200 이라 **버리지 않았다**. 하루 표본의 0.55% 가 미래로 만든
         하한을 쓰고 있었다.
    """
    f = ABS_MIN_SIGMA
    if floor is not None and np.isfinite(floor):
        f = max(float(floor), ABS_MIN_SIGMA)
    return np.maximum(sig, f)


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


def back_returns(mid, ts_ms, tick, base_sec):
    """각 시점에서 **직전 base_sec 초** 동안의 움직임(틱). 미래를 보지 않는다."""
    n = mid.size
    back = np.searchsorted(ts_ms, ts_ms - base_sec * 1000.0, side="left")
    r = np.zeros(n, dtype=np.float64)
    good = back < np.arange(n)
    d = (mid - mid[np.minimum(back, n - 1)]) / tick
    r[good] = np.where(np.isfinite(d[good]), d[good], 0.0)
    return r


def _slots(ts_ms, nb=DIURNAL_BINS):
    return (((ts_ms.astype(np.int64) // 1000) % 86400) // (86400 // nb)).astype(int)


def day_profile(r, ts_ms, nb=DIURNAL_BINS):
    """하루의 1분 칸별 변동성을 그날 평균으로 나눈 것. 계수표의 재료."""
    sl = _slots(ts_ms, nb)
    ok = np.isfinite(r)
    s2 = np.bincount(sl[ok], weights=r[ok] ** 2, minlength=nb)
    c = np.bincount(sl[ok], minlength=nb)
    v = np.sqrt(np.divide(s2, np.maximum(c, 1)))
    v[c < DIURNAL_MIN_COUNT] = np.nan
    m = np.nanmean(v)
    return v / m if np.isfinite(m) and m > 0 else np.full(nb, np.nan)


def diurnal_coef(past, ts_ms):
    """과거 프로파일들의 중앙값 -> 이벤트별 계수.

    과거가 부족하면 1 을 돌려준다 - 계수를 안 쓰는 것과 같다. 학습 구간
    맨 앞 며칠이 여기 해당하고, 워크포워드 측정상 손해는 작다.
    """
    n = ts_ms.size
    if len(past) < DIURNAL_MIN_DAYS:
        return np.ones(n)
    tab = np.nanmedian(np.asarray(past[-DIURNAL_WINDOW:]), axis=0)
    tab = np.where(np.isfinite(tab) & (tab > 0), tab, 1.0)
    return tab[_slots(ts_ms)]


def causal_sigma_sec(mid, ts_ms, tick, base_sec, halflife,
                     floor=None, seed=None, coef=None, return_state=False):
    """**base_sec 초 구간** 움직임의 인과적 EWMA 표준편차.

    정답의 기준 horizon 과 같은 구간으로 재야 단위가 맞는다. 시점 t 의 값은
    t 까지 **완료된** 구간만 쓴다 - r[i] 는 (i - base 구간) 의 움직임이므로
    미래를 보지 않는다.

    floor / seed 는 **전날**에서 받는다. 그래야 하루 앞쪽 구간도 미래를
    보지 않는다. 실시간 시스템이 장 시작 때 어제 상태를 이어받는 것과 같다.
    """
    n = mid.size
    r = back_returns(mid, ts_ms, tick, base_sec)

    # 계수로 나눠 주기를 걷어낸 뒤 EWMA 를 돌린다. 남는 것은 '평소 그 시간대
    # 대비 얼마나 센가' 라서, 개장이든 새벽이든 같은 척도가 된다.
    c = np.ones(n) if coef is None else np.where(np.isfinite(coef) & (coef > 0), coef, 1.0)
    r = r / c

    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1))
    warm = min(halflife, n)
    var = np.empty(n, dtype=np.float64)

    # 초기값도 미래를 보면 안 된다.
    #
    # 예전에는 r[:halflife] 의 평균제곱을 초기값으로 썼다. 그러면 시점 0 의
    # sigma 가 시점 halflife 까지의 데이터에 좌우된다 - 하루 앞쪽 0.3% 구간이
    # 통째로 미래 참조였다. 회귀 테스트가 '뒤쪽 절반'만 봐서 못 잡았다.
    #
    # 이제 두 가지로 나눈다.
    #   전날 값이 있으면  그걸 이어받는다 (실시간 시스템이 하는 그대로)
    #   없으면(첫날)      확장 평균으로 시작한다 - 시점 i 는 r[:i+1] 만 쓴다
    acc = float(seed) if seed is not None and np.isfinite(seed) and seed > 0 else None
    run = 0.0
    for i in range(n):
        r2 = r[i] * r[i]
        if acc is None or (i < warm and seed is None):
            run += r2
            acc = run / (i + 1)
        else:
            acc += alpha * (r2 - acc)
        var[i] = acc
    sig_t = np.sqrt(np.maximum(var, 1e-12))     # 주기를 걷어낸 '높이'
    sig = _apply_floor(c * sig_t, floor)        # 다시 곱해 실제 변동성으로
    if return_state:
        return sig, (float(var[-1]) if var.size else None)
    return sig


def build_day(cache: str, symbol: str, date: str, tick: float,
              horizons_buckets, halflife_sec: float, floor=None, seed=None,
              past_profiles=None):
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
    # 호가창이 물리적으로 성립하지 않는 시점을 무효로 본다.
    #   - 미드가 NaN (한쪽 호가가 없음)
    #   - 스프레드가 MAX_SPREAD_TICKS 초과 (한쪽만 갱신된 스냅샷)
    bad = ~np.isfinite(mid)
    if px.shape[1] >= 3:
        sp = (px[:, 2] - px[:, 1]) / tick
        bad |= ~np.isfinite(sp) | (sp <= 0) | (sp > MAX_SPREAD_TICKS)
    if bad.any():
        idx = np.maximum.accumulate(np.where(~bad, np.arange(n), 0))
        mid = mid[idx]

    ts = np.load(base + "_ts.npy").astype(np.float64)
    # 반감기(초) -> 이벤트 수. 종목·날짜마다 이벤트 속도가 다르다.
    span = max((ts[-1] - ts[0]) / 1000.0, 1.0)
    halflife = max(50, int(round(halflife_sec * ts.size / span)))
    y = targets_by_seconds(mid, ts, tick, spec.TARGET_HORIZONS_SEC)
    y[bad] = np.nan                        # 원래 구멍이던 시점은 무효 처리

    # 깨진 스냅샷은 **입력 창에 섞이기만 해도** 그 표본을 버린다.
    #
    # 정답만 무효로 하면 그 시점은 예측 대상에서 빠지지만, 100개짜리 입력 창
    # 안에는 그대로 들어온다. 창 하나에 스프레드 59,091틱 같은 값이 섞이면
    # 그 표본의 예측은 잡음이고, 잡음 예측의 제곱오차가 손실을 왜곡한다.
    #
    # 정답을 NaN 으로 만들어 두면 Dataset 이 '정답이 유한한 것만' 고르는
    # 기존 필터에 그대로 걸린다. 별도 배선이 필요 없다.
    if bad.any():
        c = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
        k = np.arange(n)
        lo = np.maximum(k - spec.SEQ_LEN + 1, 0)
        y[(c[k + 1] - c[lo]) > 0] = np.nan

    # sigma 도 가장 짧은 horizon 과 같은 구간으로 잰다
    # 계수는 **과거 날짜에서만** 만든 표를 쓴다. 이 날짜의 데이터는 안 들어간다.
    base_r = back_returns(mid, ts, tick, spec.TARGET_HORIZONS_SEC[0])
    coef = diurnal_coef(past_profiles or [], ts)
    raw, var_t = causal_sigma_sec(mid, ts, tick, spec.TARGET_HORIZONS_SEC[0],
                                  halflife, floor=None, seed=seed, coef=coef,
                                  return_state=True)
    sig = _apply_floor(raw, floor).astype(np.float32)
    # 다음 날에 넘길 상태
    #   하한   하한 씌우기 전 sigma 분포에서
    #   초기값 **주기를 걷어낸** 분산. 계수를 곱한 값을 넘기면 자정 계수가 두 번 곱해진다
    #   표     이 날짜의 프로파일. 다음 날 표에 들어간다
    nxt = (day_floor(raw), var_t, day_profile(base_r, ts))

    np.save(base + "_yt.npy", y)          # [K, 3]  틱 단위, 정규화 전
    np.save(base + "_sg.npy", sig)        # [K]     1Δt 인과적 sigma
    return y, sig, nxt


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
        # 하한은 **전날** 것을 쓴다. 그래서 날짜 순서대로 돌면서 이어 넘긴다.
        # targets 는 학습->검증(->시험) 순이라 이미 시간순이다.
        floor = seed = None
        past = []          # 지난 날들의 1분 프로파일. 다음 날 계수표의 재료
        for date in targets:
            got = build_day(args.cache, sym, date, tick, hb,
                            args.halflife_sec, floor=floor, seed=seed,
                            past_profiles=past)
            if got:
                made += 1
                floor, seed, prof = got[2]
                if np.isfinite(prof).any():
                    past.append(prof)
        print(f"   일중 계수: 창 {DIURNAL_WINDOW}일, 최소 {DIURNAL_MIN_DAYS}일, "
              f"{DIURNAL_BINS}칸  (앞 {DIURNAL_MIN_DAYS}일은 계수 1)")
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
