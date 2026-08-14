"""Order Flow (OF) / Order Flow Imbalance (OFI) 특징 생성 엔진.

사양 §3 ~ §9 를 그대로 구현한다. 데이터 소스에 의존하지 않는다 —
입력은 (timestamp_ms, bid 변경목록, ask 변경목록) 이벤트 스트림이고,
거래소별 어댑터가 이 형태로 변환해서 넣어준다.

핵심 규칙 (사양에서 특히 틀리기 쉬운 지점):
  * 버킷은 반열린 구간 [0,50), [50,100) ... 경계 시각은 다음 버킷 (§3)
  * 버킷 안의 이벤트는 B0->B1, B1->B2 로 **연쇄** 비교. 직전 버킷 최종상태와
    각 이벤트를 개별 비교하면 안 된다 (§3)
  * bid OF 는 bid 끼리, ask OF 는 ask 끼리 합산하고 끝까지 분리 유지 (§4,§5)
  * 이벤트별 OF 를 먼저 전부 합산한 뒤 **버킷당 한 번만** 정규화 (§6)
  * rolling window 는 이벤트가 아니라 50ms 버킷마다 한 칸씩 이동, 현재 버킷 제외 (§6)
  * raw_ofi 는 정규화 이전의 원시 OF 로 계산 (§7)
  * OFI 는 최근 10/20 버킷 raw_ofi 를 **먼저 합산**한 뒤 과거 100버킷 전체
    평균잔량으로 **한 번만** 나눈다. 두 OFI 의 분모는 동일 (§9)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

import ofi_spec as spec


# ----------------------------------------------------------------------------
# 호가창 재구성
# ----------------------------------------------------------------------------
class OrderBook:
    """차분(diff) 업데이트를 누적해 전체 호가창 상태를 유지한다.

    가격을 키로 하는 dict 와 정렬된 가격 리스트를 병행 관리한다. 상위 N단계만
    필요하므로 정렬 리스트의 양 끝만 읽으면 된다.
    수량 0 은 해당 호가 삭제를 의미한다 (바이낸스 depth 스트림 규약).
    """

    __slots__ = ("_bid", "_ask", "_bid_px", "_ask_px")

    def __init__(self) -> None:
        self._bid: dict[float, float] = {}
        self._ask: dict[float, float] = {}
        self._bid_px: list[float] = []   # 오름차순. 최우선 매수는 맨 뒤
        self._ask_px: list[float] = []   # 오름차순. 최우선 매도는 맨 앞

    @staticmethod
    def _apply_side(book: dict, prices: list, updates) -> None:
        for px, qty in updates:
            if qty > 0:
                if px not in book:
                    bisect.insort(prices, px)
                book[px] = qty
            elif px in book:
                del book[px]
                i = bisect.bisect_left(prices, px)
                if i < len(prices) and prices[i] == px:
                    prices.pop(i)

    def apply(self, bid_updates, ask_updates) -> None:
        self._apply_side(self._bid, self._bid_px, bid_updates)
        self._apply_side(self._ask, self._ask_px, ask_updates)

    def top(self, levels: int):
        """상위 `levels` 단계를 (bid_px, bid_qty, ask_px, ask_qty) 로 반환.

        단계가 모자라면 가격·수량 모두 0.0 으로 채운다. 0 패딩은 warm-up 구간
        에서만 발생하며, 그 구간은 §11 에 따라 어차피 폐기된다.
        """
        bp = np.zeros(levels, dtype=np.float64)
        bq = np.zeros(levels, dtype=np.float64)
        ap = np.zeros(levels, dtype=np.float64)
        aq = np.zeros(levels, dtype=np.float64)

        n = len(self._bid_px)
        for i in range(min(levels, n)):
            px = self._bid_px[n - 1 - i]        # 높은 가격부터
            bp[i] = px
            bq[i] = self._bid[px]

        for i in range(min(levels, len(self._ask_px))):
            px = self._ask_px[i]                # 낮은 가격부터
            ap[i] = px
            aq[i] = self._ask[px]

        return bp, bq, ap, aq


# ----------------------------------------------------------------------------
# 이벤트별 OF (§4, §5)
# ----------------------------------------------------------------------------
def event_of_bid(prev_px, prev_qty, cur_px, cur_qty):
    """매수 OF. 사양 §4 의 piecewise 정의를 그대로 벡터화한 것.

        p[i] > p[i-1]  ->   q[i]
        p[i] == p[i-1] ->   q[i] - q[i-1]
        p[i] < p[i-1]  ->  -q[i-1]
    """
    return np.where(
        cur_px > prev_px,
        cur_qty,
        np.where(cur_px == prev_px, cur_qty - prev_qty, -prev_qty),
    )


def event_of_ask(prev_px, prev_qty, cur_px, cur_qty):
    """매도 OF. 사양 §5. 부등호 방향이 매수와 반대다.

        p[i] < p[i-1]  ->   q[i]
        p[i] == p[i-1] ->   q[i] - q[i-1]
        p[i] > p[i-1]  ->  -q[i-1]
    """
    return np.where(
        cur_px < prev_px,
        cur_qty,
        np.where(cur_px == prev_px, cur_qty - prev_qty, -prev_qty),
    )


# ----------------------------------------------------------------------------
# 결과 묶음
# ----------------------------------------------------------------------------
@dataclass
class OFIResult:
    """버킷 단위 산출물. K = 버킷 개수, L = 호가 단계 수."""

    first_bucket: int            # 절대 버킷 인덱스 (= ts_ms // BUCKET_MS)
    bid_of: np.ndarray           # [K, L] 원시 매수 OF (버킷 합산, 정규화 전)
    ask_of: np.ndarray           # [K, L] 원시 매도 OF
    q_bid_end: np.ndarray        # [K, L] 버킷 종료 시점 매수 잔량
    q_ask_end: np.ndarray        # [K, L] 버킷 종료 시점 매도 잔량
    raw_ofi: np.ndarray          # [K]    §7
    features: np.ndarray         # [K, 22] §10 최종 입력 특징
    mid: np.ndarray              # [K]    §12 미드프라이스
    best_bid: np.ndarray         # [K]    버킷 종료 시점 최우선 매수호가
    best_ask: np.ndarray         # [K]    버킷 종료 시점 최우선 매도호가
    mid_valid: np.ndarray        # [K] bool, 양쪽 호가가 모두 존재하는지
    n_events: np.ndarray         # [K] int, 버킷별 실제 이벤트 수 (검증·진단용)
    n_repairs: np.ndarray        # [K] int, 버킷별 장부 정정 횟수 (OF 미반영)
    feature_valid: np.ndarray    # [K] bool, §6/§9 warm-up 을 만족하는 버킷

    @property
    def n_buckets(self) -> int:
        return self.bid_of.shape[0]


# ----------------------------------------------------------------------------
# 메인 파이프라인
# ----------------------------------------------------------------------------
def build_features(
    events,
    levels: int = spec.LEVELS,
    bucket_ms: int = spec.BUCKET_MS,
    depth_window: int = spec.DEPTH_ROLLING_WINDOW,
    ofi_windows=tuple(spec.OFI_WINDOWS),
    eps: float = spec.EPS,
) -> OFIResult:
    """이벤트 스트림 -> 버킷별 22개 특징.

    Parameters
    ----------
    events : iterable of (ts_ms:int, bid_updates, ask_updates[, is_repair])
        `bid_updates` / `ask_updates` 는 (price, qty) 의 순회 가능 객체.
        qty == 0 은 해당 호가 삭제. 이벤트는 **거래소 순서대로** 들어와야 한다.

        `is_repair=True` 인 이벤트는 호가창 상태만 고치고 **OF 에는 반영되지
        않는다**. 캡처 유실로 생긴 유령 호가를 bookTicker 기준으로 지우는 등의
        장부 정정용이다. 실제 주문이 들어오거나 빠진 게 아니므로 주문흐름으로
        세면 가짜 OF 가 만들어진다. 정정 후 상태가 다음 실제 이벤트의 비교
        기준이 된다.
    """
    book = OrderBook()

    bid_of_rows: list[np.ndarray] = []
    ask_of_rows: list[np.ndarray] = []
    qb_rows: list[np.ndarray] = []
    qa_rows: list[np.ndarray] = []
    bid_px_rows: list[float] = []
    ask_px_rows: list[float] = []
    nev_rows: list[int] = []
    nrep_rows: list[int] = []

    first_bucket = None
    cur_k = None
    acc_bid = np.zeros(levels)
    acc_ask = np.zeros(levels)
    n_ev = 0
    n_rep = 0
    prev_top = None          # 직전 이벤트 적용 후 상태
    last_top = None          # 직전 버킷 최종 상태 (빈 버킷 forward-fill 용)

    def close_bucket():
        """현재 버킷을 확정하고 결과 리스트에 추가."""
        bp, bq, ap, aq = last_top
        bid_of_rows.append(acc_bid.copy())
        ask_of_rows.append(acc_ask.copy())
        qb_rows.append(bq.copy())
        qa_rows.append(aq.copy())
        bid_px_rows.append(bp[0])
        ask_px_rows.append(ap[0])
        nev_rows.append(n_ev)
        nrep_rows.append(n_rep)

    for event in events:
        ts_ms, bid_updates, ask_updates = event[0], event[1], event[2]
        is_repair = len(event) > 3 and event[3]
        k = ts_ms // bucket_ms          # 반열린 구간. 경계 시각은 다음 버킷 (§3)

        if first_bucket is None:
            first_bucket = k
            cur_k = k
        elif k != cur_k:
            if k < cur_k:
                raise ValueError(
                    f"이벤트 시각이 뒤로 갔다: bucket {k} < {cur_k}. "
                    "입력 스트림이 시간순으로 정렬되어 있어야 한다."
                )
            close_bucket()
            # 이벤트가 없는 버킷: 호가창은 forward-fill, OF 는 0 (§3)
            for _ in range(cur_k + 1, k):
                bid_of_rows.append(np.zeros(levels))
                ask_of_rows.append(np.zeros(levels))
                bp, bq, ap, aq = last_top
                qb_rows.append(bq.copy())
                qa_rows.append(aq.copy())
                bid_px_rows.append(bp[0])
                ask_px_rows.append(ap[0])
                nev_rows.append(0)
                nrep_rows.append(0)
            cur_k = k
            acc_bid = np.zeros(levels)
            acc_ask = np.zeros(levels)
            n_ev = 0
            n_rep = 0

        book.apply(bid_updates, ask_updates)
        cur_top = book.top(levels)

        if is_repair:
            # 장부 정정. 주문흐름이 아니므로 OF 에 넣지 않는다. 정정 후 상태를
            # 기준으로 삼아야 다음 실제 이벤트의 OF 가 올바르게 계산된다.
            n_rep += 1
        elif prev_top is None:
            # 첫 이벤트에는 비교 대상이 없다. OF 를 0 으로 두고 상태만 잡는다.
            # 이 버킷은 warm-up 구간이라 §11 에서 폐기된다.
            n_ev += 1
        else:
            acc_bid += event_of_bid(prev_top[0], prev_top[1], cur_top[0], cur_top[1])
            acc_ask += event_of_ask(prev_top[2], prev_top[3], cur_top[2], cur_top[3])
            n_ev += 1

        prev_top = cur_top
        last_top = cur_top

    if first_bucket is None:
        raise ValueError("이벤트가 하나도 없다.")
    close_bucket()

    bid_of = np.asarray(bid_of_rows)                 # [K, L]
    ask_of = np.asarray(ask_of_rows)
    q_bid_end = np.asarray(qb_rows)
    q_ask_end = np.asarray(qa_rows)
    best_bid = np.asarray(bid_px_rows)
    best_ask = np.asarray(ask_px_rows)
    n_events = np.asarray(nev_rows, dtype=np.int64)
    n_repairs = np.asarray(nrep_rows, dtype=np.int64)
    K = bid_of.shape[0]

    # ---- §6 OF 정규화: 같은 방향, 같은 단계, 과거 depth_window 버킷 평균잔량 ----
    mean_bid_depth = _trailing_mean(q_bid_end, depth_window)   # [K, L], 현재 버킷 제외
    mean_ask_depth = _trailing_mean(q_ask_end, depth_window)

    norm_bid_of = bid_of / (mean_bid_depth + eps)
    norm_ask_of = ask_of / (mean_ask_depth + eps)

    # ---- §7 원시 OFI: 정규화 이전 값으로 계산 ----
    raw_ofi = (bid_of - ask_of).sum(axis=1)                    # [K]

    # ---- §8 OFI 정규화용 전체 평균잔량 ----
    total_depth = (q_bid_end + q_ask_end).sum(axis=1) / (2 * q_bid_end.shape[1])
    mean_total_depth = _trailing_mean(total_depth[:, None], depth_window)[:, 0]

    # ---- §9 최근 W개 raw_ofi 를 먼저 합산 후 한 번만 나눈다 ----
    ofi_cols = []
    for w in ofi_windows:
        ofi_cols.append(_trailing_sum_inclusive(raw_ofi, w) / (mean_total_depth + eps))

    # ---- §10 특징 조립. 순서 고정 ----
    features = np.concatenate(
        [norm_bid_of, norm_ask_of] + [c[:, None] for c in ofi_cols], axis=1
    )
    assert features.shape[1] == spec.OF_DIM, features.shape

    # ---- §12 미드프라이스 ----
    mid_valid = (best_bid > 0) & (best_ask > 0) & (best_ask >= best_bid)
    mid = np.where(mid_valid, (best_bid + best_ask) / 2.0, np.nan)

    # ---- §11 warm-up: 필요한 과거가 모두 갖춰진 버킷만 유효 ----
    feature_valid = np.zeros(K, dtype=bool)
    warmup = max(depth_window, max(ofi_windows) - 1)
    feature_valid[warmup:] = True
    feature_valid &= np.isfinite(features).all(axis=1)
    feature_valid &= mid_valid

    return OFIResult(
        first_bucket=first_bucket,
        bid_of=bid_of,
        ask_of=ask_of,
        q_bid_end=q_bid_end,
        q_ask_end=q_ask_end,
        raw_ofi=raw_ofi,
        features=features,
        mid=mid,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_valid=mid_valid,
        n_events=n_events,
        n_repairs=n_repairs,
        feature_valid=feature_valid,
    )


# ----------------------------------------------------------------------------
# rolling 헬퍼
# ----------------------------------------------------------------------------
def _trailing_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """arr[k-window : k] 의 평균. **현재 행 k 는 제외** (§6, §8).

    과거가 모자란 앞부분은 NaN. 0 으로 채우지 않는다 (§11).
    """
    csum = np.concatenate([np.zeros((1,) + arr.shape[1:]), np.cumsum(arr, axis=0)])
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if arr.shape[0] > window:
        out[window:] = (csum[window:-1] - csum[:-window - 1]) / window
    return out


def _trailing_sum_inclusive(arr: np.ndarray, window: int) -> np.ndarray:
    """arr[k-window+1 : k+1] 의 합. **현재 행 k 를 포함** (§9).

    ofi_sum_10[k] = sum(raw_ofi[k-r] for r in range(10)) 과 동일하다.
    """
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    out[window - 1:] = csum[window:] - csum[:-window]
    return out
