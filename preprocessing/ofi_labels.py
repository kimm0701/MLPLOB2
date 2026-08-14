"""미래 수익률 정답값 생성 (사양 §12, §13) 과 유효 샘플 선별 (§11, §17).

    return[k, h] = (mid[k + offset[h]] - mid[k]) / mid[k]

offset 은 50ms 버킷 개수다: 1초 = 20버킷 ... 10초 = 200버킷.
수익률은 소수 그대로 저장한다 (0.001 = +0.1%). 100 을 곱하지 않는다.

미래 데이터가 없는 마지막 구간은 **제거한다**. 마지막 값을 끌어다 채우면
(forward-fill) 존재하지 않는 미래를 지어내는 것이고, 그 구간의 수익률이 0 으로
깔려 모델이 "곧 아무 일도 없다" 를 학습해 버린다.
"""

from __future__ import annotations

import numpy as np

import ofi_spec as spec


def build_targets(mid: np.ndarray, mid_valid: np.ndarray,
                  horizon_buckets=tuple(spec.TARGET_HORIZON_BUCKETS)):   # 구 방식
    """미드프라이스 배열 -> (targets [K, H], target_valid [K]).

    target_valid[k] 는 다음을 모두 만족할 때만 True.
      * 현재 시점 mid 가 유효하고 0 이 아님
      * 모든 horizon 의 미래 mid 가 존재하고 유효함
    """
    mid = np.asarray(mid, dtype=np.float64)
    mid_valid = np.asarray(mid_valid, dtype=bool)
    K = mid.shape[0]
    H = len(horizon_buckets)

    targets = np.full((K, H), np.nan, dtype=np.float64)
    valid = mid_valid & np.isfinite(mid) & (mid > 0)

    for j, off in enumerate(horizon_buckets):
        if off >= K:
            valid[:] = False
            continue
        future = np.full(K, np.nan)
        future[:K - off] = mid[off:]

        future_ok = np.zeros(K, dtype=bool)
        future_ok[:K - off] = mid_valid[off:] & np.isfinite(mid[off:])

        with np.errstate(invalid="ignore", divide="ignore"):
            targets[:, j] = (future - mid) / mid
        valid &= future_ok

    # 미래가 부족한 꼬리 구간은 명시적으로 제거 (§13)
    valid[K - max(horizon_buckets):] = False
    targets[~valid] = np.nan
    return targets, valid


def rolling_all(mask: np.ndarray, window: int) -> np.ndarray:
    """out[k] = mask[k-window+1 : k+1].all(). 앞부분은 False."""
    mask = np.asarray(mask, dtype=bool)
    K = mask.shape[0]
    out = np.zeros(K, dtype=bool)
    if K < window:
        return out
    csum = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    out[window - 1:] = (csum[window:] - csum[:-window]) == window
    return out


def valid_sample_indices(feature_valid: np.ndarray, target_valid: np.ndarray,
                         seq_len: int = spec.SEQ_LEN) -> np.ndarray:
    """모델에 넣을 수 있는 시점 k 의 목록.

    샘플 하나는 x[k-seq_len+1 : k+1] 과 y[k] 로 이루어진다. 따라서 입력 구간
    seq_len 개가 **전부** 유효하고 정답도 유효해야 한다 (§11).
    """
    ok = rolling_all(feature_valid, seq_len) & np.asarray(target_valid, dtype=bool)
    return np.flatnonzero(ok)


def split_indices(indices: np.ndarray, n_buckets: int, split_rates=(0.8, 0.1, 0.1),
                  seq_len: int = spec.SEQ_LEN,
                  max_horizon: int = max(spec.TARGET_HORIZON_BUCKETS)):
    """시간순 3분할 (§17). 섞지 않는다.

    한 샘플은 과거 `seq_len` 버킷을 입력으로, 미래 `max_horizon` 버킷을 정답으로
    쓴다. 경계에 걸친 샘플은 입력이나 정답이 옆 구간을 침범하므로 버린다.
    그래서 각 구간의 앞뒤로 여유를 두고 잘라낸다.
    """
    assert abs(sum(split_rates) - 1.0) < 1e-9, split_rates
    b1 = int(n_buckets * split_rates[0])
    b2 = b1 + int(n_buckets * split_rates[1])
    bounds = [(0, b1), (b1, b2), (b2, n_buckets)]

    out = []
    for lo, hi in bounds:
        # 입력 시작이 lo 이상, 정답 끝이 hi 미만이어야 구간 안에 온전히 들어간다
        keep = indices[(indices - seq_len + 1 >= lo) & (indices + max_horizon < hi)]
        out.append(keep)
    return out
