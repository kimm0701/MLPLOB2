"""기존 22개 특징에 호가창 상태 2개 + 체결 3개를 덧붙여 27개로 만든다.

    python scripts/make_features.py

전처리를 다시 하지 않는다. _x.npy (22개) 와 _px.npy (미드/매수/매도) 만
읽어서 _x2.npy (24개) 를 새로 쓴다.

왜 이 둘인가
------------
22개 주문흐름 특징은 최근 평균 잔량으로 나눠서 무차원이다. 덕분에 가격대가
다른 종목이 같은 입력층을 쓸 수 있지만, 대가로 **호가창이 지금 어떤 상태인지**
가 지워진다. 같은 OFI 라도 호가창 상태에 따라 뜻이 다르다.

    스프레드 1틱   미드는 1호가 물량이 소진될 때만 움직인다.
                   예측 대상 = 큐 소진 경쟁. 경로가 하나뿐이라 잘 맞는다.
    스프레드 넓음  누군가 안쪽에 걸면 즉시 미드가 움직인다.
                   예측 대상이 완전히 다르다.

Kolm et al. (2023) §4.2.1 에서 틱 크기 하나가 종목 간 예측력 차이의 58.6%
를 설명했고, §4.4 는 Gould & Bonart (2016) 를 인용해 위 메커니즘을 설명한다.

**중요**: 이 둘은 종목 신분증이 아니라 **상태 변수**다. 같은 종목 안에서도
순간순간 바뀐다 - 한산할 때 스프레드가 벌어지고 체결이 몰리면 좁아진다.
그래서 종목이 3개뿐이어도 정보를 준다. 반면 tick_bp(틱/가격)는 종목마다
거의 고정이라 진짜 신분증이고, 종목을 늘리기 전에는 넣지 않는다.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                    # noqa: E402
from preprocessing.ofi_dataset import split_dates          # noqa: E402
from scripts.download_data import FILE_IDS, WEEKDAYS       # noqa: E402
from preprocessing.binance_capture import iter_trades      # noqa: E402
from preprocessing.trade_features import trade_features    # noqa: E402
from scripts.make_targets import infer_tick                # noqa: E402

EXTRA_NAMES = ["log_spread_ticks", "log_secs_since_move",
               "trade_flow_10", "trade_flow_20", "log_trade_intensity"]
MAX_GAP_SEC = 60.0          # 이벤트 간격 상한 (거래 공백 구간 방어)


def spread_ticks(px: np.ndarray, tick: float) -> np.ndarray:
    """(매도 - 매수) / 틱.  로그를 씌워 범위를 좁힌다.

    실측으로 종목·시점에 따라 1틱에서 수십 틱까지 벌어진다. 그냥 넣으면 값이
    크다는 이유만으로 첫 층 가중치를 지배하므로 log10 을 씌운다.
    """
    sp = (px[:, 2] - px[:, 1]) / tick
    sp = np.where(np.isfinite(sp) & (sp > 0), sp, 1.0)
    return np.log10(np.maximum(sp, 1.0)).astype(np.float32)


def secs_since_move(mid: np.ndarray) -> np.ndarray:
    """마지막으로 미드가 움직인 뒤 흐른 시간(초)의 log10.

    이벤트 시계와 시계 시간을 잇는 값이다. 같은 100버킷 입력이라도 담고 있는
    가격변화 횟수가 종목·시간대마다 다른데, 그 밀도를 모델에 알려준다.
    과거만 본다 - 시점 t 의 값은 t 까지의 이력만 쓴다.
    """
    n = mid.size
    moved = np.empty(n, dtype=bool)
    moved[0] = True
    moved[1:] = np.diff(mid) != 0
    # 각 시점에서 "가장 최근에 움직인 인덱스"
    last = np.maximum.accumulate(np.where(moved, np.arange(n), 0))
    gap = (np.arange(n) - last) * spec.BUCKET_MS / 1000.0
    gap = np.clip(gap, spec.BUCKET_MS / 1000.0, MAX_GAP_SEC)
    return np.log10(gap).astype(np.float32)


def build_day(cache: str, symbol: str, date: str, tick: float, raw_root: str):
    """22 + 상태 2 + 체결 3 = 27 열.

    체결은 원본 tar 를 다시 읽어야 한다. 무거운 호가창 복원은 다시 하지 않고
    @trade 줄만 훑으므로 전처리 전체를 다시 도는 것보다 훨씬 싸다.
    체결이 없거나 원본이 없으면 그 세 열은 0 으로 둔다.
    """
    base = os.path.join(cache, symbol, date)
    need = ["_x.npy", "_px.npy", "_ts.npy"]
    if not all(os.path.exists(base + e) for e in need):
        return None
    x = np.load(base + "_x.npy", mmap_mode="r")
    px = np.asarray(np.load(base + "_px.npy", mmap_mode="r"), dtype=np.float64)
    ts = np.asarray(np.load(base + "_ts.npy"), dtype=np.int64)
    n = min(x.shape[0], px.shape[0], ts.size)
    px, ts = px[:n], ts[:n]

    tar = os.path.join(raw_root, symbol, f"{symbol}USDT_{date}.tar")
    if os.path.exists(tar):
        rows = [(t, sq) for t, _, _, sq in iter_trades(tar)]
        T = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 2))
        tf = trade_features(ts, T[:, 0], T[:, 1])
    else:
        tf = np.zeros((n, 3), dtype=np.float32)

    extra = np.concatenate(
        [np.stack([spread_ticks(px, tick), secs_since_move(px[:, 0])], axis=1),
         tf[:n]], axis=1)
    out = np.concatenate([np.asarray(x[:n], dtype=np.float32), extra], axis=1)
    assert out.shape[1] == spec.INPUT_DIM, (out.shape, spec.INPUT_DIM)
    np.save(base + "_x2.npy", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--raw", default="data/raw",
                    help="체결을 읽을 원본 tar 위치")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=spec.N_VAL)
    ap.add_argument("--n-test", type=int, default=spec.N_TEST)
    args = ap.parse_args()

    train, val, test = split_dates(args.dates, args.n_val, args.n_test)
    print(f"틱 산출: 학습 {len(train)}일   (검증 {val} / 시험 {test} 미사용)\n")
    print(f"추가 특징 {EXTRA_NAMES}  ->  {spec.INPUT_DIM} + 2 = "
          f"{spec.INPUT_DIM + 2}개\n")

    meta = {}
    for sym in args.symbols:
        tick = infer_tick(args.cache, sym, train)
        if not tick:
            print(f"{sym}: 데이터 없음, 건너뜀")
            continue
        made, samples = 0, None
        for date in args.dates:
            out = build_day(args.cache, sym, date, tick, args.raw)
            if out is not None:
                made += 1
                if samples is None:
                    samples = out
        if samples is None:
            continue
        sp, gp = samples[:, -2], samples[:, -1]
        meta[sym] = dict(tick=tick, n_days=made)
        print(f"[{sym}] 틱 {tick:g}   {made}일 저장")
        print(f"   스프레드(틱)      중앙 {10**np.median(sp):6.2f}   "
              f"p5 {10**np.percentile(sp,5):5.2f}  p95 {10**np.percentile(sp,95):6.2f}")
        print(f"   마지막 변동 후(초) 중앙 {10**np.median(gp):6.3f}   "
              f"p5 {10**np.percentile(gp,5):5.3f}  p95 {10**np.percentile(gp,95):6.3f}")
        print(f"   두 열의 표준편차   {sp.std():.3f}  {gp.std():.3f}"
              f"    (다른 특징과 견줄 만한가)\n")

    if not meta:
        print("만든 것이 없습니다.", file=sys.stderr)
        return 1
    with open(os.path.join(args.cache, "features_v2.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(extra_names=EXTRA_NAMES, symbols=meta), fh,
                  ensure_ascii=False, indent=2)
    print("저장: _x2.npy (종목·날짜별) + features_v2.json")
    print("\n두 열은 **종목별 z-score 를 씌우면 안 된다** - 종목 구분이 지워진다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
