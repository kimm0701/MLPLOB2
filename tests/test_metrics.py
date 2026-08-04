"""성적 지표 검증.

지표가 틀리면 모델이 좋은지 나쁜지 판단 자체가 어긋나므로, 답을 아는 경우를
넣어서 확인한다.
"""

import numpy as np
import pytest

from utils.metrics import (
    directional_accuracy,
    information_coefficient,
    mae,
    mse,
    r2,
    summarise,
)


def test_perfect_prediction():
    t = np.random.default_rng(0).normal(size=(500, 3))
    assert np.allclose(mse(t, t), 0)
    assert np.allclose(mae(t, t), 0)
    assert np.allclose(directional_accuracy(t, t), 1.0)
    assert np.allclose(information_coefficient(t, t), 1.0)
    assert np.allclose(r2(t, t), 1.0)


def test_inverted_prediction():
    t = np.random.default_rng(1).normal(size=(500, 2))
    assert np.allclose(directional_accuracy(-t, t), 0.0)
    assert np.allclose(information_coefficient(-t, t), -1.0)


def test_mse_and_mae_are_per_horizon():
    t = np.zeros((4, 2))
    p = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert mse(p, t) == pytest.approx([1.0, 4.0])
    assert mae(p, t) == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# 방향 정확도
# ---------------------------------------------------------------------------
def test_direction_ignores_zero_targets():
    """정답이 0 인 표본은 방향이 없으므로 세지 않는다."""
    t = np.array([[1.0], [-1.0], [0.0], [0.0]])
    p = np.array([[1.0], [-1.0], [5.0], [-5.0]])
    assert directional_accuracy(p, t) == pytest.approx([1.0]), \
        "0 인 표본 2개를 틀림으로 세면 0.5 가 된다"


def test_direction_all_zero_targets_gives_nan():
    t = np.zeros((5, 1))
    p = np.ones((5, 1))
    assert np.isnan(directional_accuracy(p, t)[0])


def test_direction_half_right_is_a_coin_flip():
    t = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    p = np.array([[1.0], [-1.0], [-1.0], [1.0]])
    assert directional_accuracy(p, t) == pytest.approx([0.5])


# ---------------------------------------------------------------------------
# IC
# ---------------------------------------------------------------------------
def test_ic_is_rank_based_so_outliers_do_not_dominate():
    """값 상관은 극단값 하나에 흔들리지만 순위 상관은 버틴다."""
    rng = np.random.default_rng(2)
    t = rng.normal(size=(300, 1))
    p = t.copy()
    p[0] = 1e6                      # 극단값 하나를 심는다
    t[0] = -1e-6

    rank_ic = information_coefficient(p, t, rank=True)[0]
    value_ic = information_coefficient(p, t, rank=False)[0]
    assert rank_ic > 0.9, "순위 상관은 나머지 299개가 완벽하므로 높아야 한다"
    assert value_ic < rank_ic


def test_ic_of_noise_is_near_zero():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(20000, 1))
    p = rng.normal(size=(20000, 1))
    assert abs(information_coefficient(p, t)[0]) < 0.03


def test_ic_of_constant_prediction_is_nan():
    t = np.random.default_rng(4).normal(size=(100, 1))
    p = np.zeros((100, 1))
    assert np.isnan(information_coefficient(p, t)[0])


def test_ic_handles_ties():
    t = np.array([[1.0], [1.0], [2.0], [3.0]])
    p = np.array([[1.0], [1.0], [2.0], [3.0]])
    assert information_coefficient(p, t)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------
def test_r2_zero_when_predicting_the_mean():
    t = np.random.default_rng(5).normal(size=(1000, 1))
    p = np.full_like(t, t.mean())
    assert r2(p, t)[0] == pytest.approx(0.0, abs=1e-9)


def test_r2_negative_when_worse_than_the_mean():
    t = np.random.default_rng(6).normal(size=(500, 1))
    p = t + 5.0
    assert r2(p, t)[0] < 0


# ---------------------------------------------------------------------------
# 묶음
# ---------------------------------------------------------------------------
def test_summarise_shapes_and_horizons():
    rng = np.random.default_rng(7)
    t = rng.normal(scale=1e-4, size=(1000, 10))
    p = t * 0.3 + rng.normal(scale=1e-4, size=(1000, 10))
    s = summarise(p, t, horizons=list(range(1, 11)))

    for key in ("mse", "mae", "dir_acc", "ic", "r2"):
        assert s[key].shape == (10,), key
    assert s["n"] == 1000
    assert s["horizons"][0] == 1 and s["horizons"][-1] == 10
    assert (s["ic"] > 0).all(), "예측에 신호를 섞었으므로 IC 는 양수여야 한다"


# ---------------------------------------------------------------------------
# 배율 보정 (scripts/calibrate.py)
# ---------------------------------------------------------------------------
def test_calibration_recovers_r2_up_to_ic_squared():
    """크기가 어긋난 예측은 R2 가 IC^2 보다 낮고, 배율을 맞추면 회복된다."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.calibrate import fit_scale

    rng = np.random.default_rng(11)
    t = rng.normal(scale=2e-4, size=(50_000, 3))
    signal = t * 0.05 + rng.normal(scale=2e-4, size=t.shape) * 0.999
    pred = signal * 7.0                      # 크기를 7배 부풀린 예측

    ic = information_coefficient(pred, t, rank=False)
    before = r2(pred, t)
    assert (before < 0).all(), "부풀린 예측은 평균값 답하기보다도 못하다"

    alpha, beta = fit_scale(pred, t)
    after = r2(pred * alpha + beta, t)

    assert (after > before).all()
    assert np.allclose(after, ic ** 2, atol=1e-4), "보정 후 R2 는 IC^2 에 도달한다"


def test_calibration_leaves_a_well_scaled_prediction_alone():
    rng = np.random.default_rng(12)
    t = rng.normal(scale=2e-4, size=(20_000, 2))
    pred = t * 0.05 + rng.normal(scale=2e-4, size=t.shape)

    from scripts.calibrate import fit_scale
    alpha, _ = fit_scale(pred, t)
    r2_before, r2_after = r2(pred, t), r2(pred * alpha, t)
    assert (r2_after >= r2_before - 1e-12).all(), "보정이 성능을 깎으면 안 된다"
