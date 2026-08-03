"""사양 §12, §13, §17, §18-24 검증."""

import numpy as np
import pytest

import ofi_spec as spec
from preprocessing.ofi_labels import (
    build_targets,
    rolling_all,
    split_indices,
    valid_sample_indices,
)

H = spec.TARGET_HORIZON_BUCKETS          # [20, 40, ... 200]


def test_horizon_offsets_are_seconds_times_20():
    assert H == [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    assert len(H) == spec.OUTPUT_DIM == 10


def test_13_return_formula_and_decimal_form():
    """수익률 = (미래mid - 현재mid) / 현재mid, 소수 그대로."""
    K = 500
    mid = np.full(K, 100.0)
    mid[20:] = 100.1          # 1초 뒤부터 +0.1%
    t, v = build_targets(mid, np.ones(K, bool))

    assert t[0, 0] == pytest.approx(0.001), "0.1% 는 0.001 로 저장 (100 곱하지 않음)"
    assert t[0, 0] != pytest.approx(0.1)


def test_13_negative_and_zero_returns():
    K = 500
    mid = np.full(K, 200.0)
    mid[40:] = 199.6                       # 2초 뒤 -0.2%
    t, v = build_targets(mid, np.ones(K, bool))
    assert t[0, 1] == pytest.approx(-0.002)
    assert t[0, 0] == pytest.approx(0.0)   # 1초 뒤는 변화 없음


def test_13_each_horizon_uses_its_own_offset():
    K = 600
    mid = np.arange(K, dtype=float) + 1000.0     # 버킷당 +1
    t, v = build_targets(mid, np.ones(K, bool))
    k = 100
    for j, off in enumerate(H):
        expected = (mid[k + off] - mid[k]) / mid[k]
        assert t[k, j] == pytest.approx(expected), f"horizon {off}"


def test_18_24_tail_without_future_is_dropped():
    """미래 10초가 없는 마지막 구간은 제거된다. 채워넣지 않는다."""
    K = 400
    mid = np.linspace(100, 101, K)
    t, v = build_targets(mid, np.ones(K, bool))

    assert not v[K - max(H):].any(), "미래가 모자란 꼬리는 전부 무효"
    assert np.isnan(t[K - max(H):]).all(), "무효 구간은 NaN. 0 이나 마지막 값으로 채우지 않는다"
    assert v[:K - max(H)].all()


def test_invalid_mid_propagates_to_targets():
    K = 600
    mid = np.linspace(100, 101, K)
    mv = np.ones(K, bool)
    mv[250] = False                       # 미래 한 지점이 비정상
    t, v = build_targets(mid, mv)

    # 250 을 미래로 참조하는 시점들은 전부 무효가 되어야 한다
    for off in H:
        k = 250 - off
        if 0 <= k < K:
            assert not v[k], f"offset {off} 로 무효 지점을 참조하는 k={k}"
    assert not v[250]


# ---------------------------------------------------------------------------
# 유효 샘플 선별 (§11)
# ---------------------------------------------------------------------------
def test_rolling_all():
    m = np.array([1, 1, 0, 1, 1, 1, 1], bool)
    got = rolling_all(m, 3)
    assert got.tolist() == [False, False, False, False, False, True, True]


def test_sample_requires_full_input_window_valid():
    K = 1000
    fv = np.ones(K, bool)
    fv[500] = False                       # 한 시점만 깨져도
    tv = np.ones(K, bool)
    idx = valid_sample_indices(fv, tv, seq_len=100)

    # 500 을 입력 구간에 포함하는 k (500..599) 는 전부 빠져야 한다
    assert not any(500 <= k <= 599 for k in idx)
    assert 499 in idx and 600 in idx


# ---------------------------------------------------------------------------
# 시간순 분할 (§17)
# ---------------------------------------------------------------------------
def test_17_splits_are_time_ordered_and_disjoint():
    K = 100_000
    idx = np.arange(K)
    tr, va, te = split_indices(idx, K, seq_len=100, max_horizon=200)

    assert tr.max() < va.min() < te.min()
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(va, te)) == 0


def test_17_sample_window_never_crosses_a_split_boundary():
    """입력 구간과 정답 구간이 통째로 한 split 안에 들어와야 한다."""
    K = 100_000
    seq, hor = 100, 200
    idx = np.arange(K)
    splits = split_indices(idx, K, seq_len=seq, max_horizon=hor)
    b1 = int(K * 0.8)
    b2 = b1 + int(K * 0.1)
    bounds = [(0, b1), (b1, b2), (b2, K)]

    for keep, (lo, hi) in zip(splits, bounds):
        assert (keep - seq + 1 >= lo).all(), "입력이 앞 구간을 침범"
        assert (keep + hor < hi).all(), "정답이 뒤 구간을 침범"


def test_17_no_shuffling():
    K = 50_000
    idx = np.arange(K)
    tr, va, te = split_indices(idx, K)
    for part in (tr, va, te):
        assert (np.diff(part) > 0).all(), "분할 결과는 시간순을 유지해야 한다"
