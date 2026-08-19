"""체결 피처 3개. _ts.npy 의 버킷 경계에 체결을 배정해 만든다."""
import numpy as np

# 정규화 분모의 하한 = 그날 체결량 중앙값 x 이 비율.
# 너무 작으면 체결 없는 구간에서 값이 폭발하고, 너무 크면 신호가 눌린다.
MIN_SCALE_RATIO = 0.1


def _trailing_sum(x, w):
    """현재 버킷 포함, 과거 w 개 합. 앞쪽은 있는 만큼만."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    k = np.arange(x.size)
    return c[k + 1] - c[np.maximum(k - w + 1, 0)]


def _trailing_mean(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    k = np.arange(x.size)
    lo = np.maximum(k - w, 0)
    n = np.maximum(k - lo, 1)
    return (c[k] - c[lo]) / n          # 현재 버킷 **제외** (인과)


def trade_features(ts_bucket, tr_ts, tr_signed, windows=(10, 20), depth_window=100,
                   eps=1e-8):
    """[K,3]  체결흐름(w1), 체결흐름(w2), log 체결강도.

    ts_bucket : [K] 버킷 종료 시각(ms).  체결은 이 경계로 배정한다.
    tr_signed : 부호 있는 체결량 (+ 매수공격, − 매도공격)

    정규화 분모는 **과거 depth_window 버킷의 평균 체결량**이다. 호가창 잔량을
    쓰지 않으므로 전처리 산출물만으로 계산된다. 현재 버킷을 빼서 인과성을
    지킨다 - 지금 체결량으로 지금을 나누면 값이 항상 1 근처가 되어 정보가 없다.
    """
    K = ts_bucket.size
    if tr_ts.size == 0:
        return np.zeros((K, 3), dtype=np.float32)

    # 체결 시각이 어느 버킷에 속하는지. 버킷 종료 시각 기준 오른쪽 경계 포함.
    idx = np.searchsorted(ts_bucket, tr_ts, side="left")
    ok = idx < K
    idx, sgn = idx[ok], tr_signed[ok]

    signed = np.bincount(idx, weights=sgn, minlength=K)
    volume = np.bincount(idx, weights=np.abs(sgn), minlength=K)
    count = np.bincount(idx, minlength=K).astype(np.float64)

    # 분모에 하한을 둔다.
    #
    # 버킷의 88% 에는 체결이 없어서, 100버킷 창이 통째로 비는 구간이 생긴다.
    # eps 만 더하면 거기서 1e-8 로 나누게 되어 값이 폭발한다 - 실측으로
    # 표준편차가 100만을 넘었다(다른 피처는 0.17~0.68). sigma 하한에서 겪은
    # 것과 같은 형태의 실수다.
    #
    # 하한은 **그날 체결량의 중앙값**으로 둔다. 체결이 아예 없던 구간에서는
    # "평소 한 번 체결될 만한 양" 을 기준으로 재게 되어 뜻이 분명하다.
    nz = volume[volume > 0]
    floor = float(np.median(nz)) * MIN_SCALE_RATIO if nz.size else 1.0
    scale = np.maximum(_trailing_mean(volume, depth_window), max(floor, eps))
    cols = [_trailing_sum(signed, w) / (scale * w) for w in windows]
    cols.append(np.log10(1.0 + _trailing_sum(count, max(windows))))
    return np.stack(cols, axis=1).astype(np.float32)
