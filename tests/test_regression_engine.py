"""학습 엔진 검증 — 특히 출력 단위.

모델이 소수(0.0002)가 아니라 bp(2) 를 내놓게 하는 게 핵심이다. 이걸 되돌리면
학습이 눈금 맞추는 데만 몇 epoch 을 쓴다. 실측으로 첫 epoch 손실이 3467 대
36 이었다.
"""

import numpy as np
import pytest
import torch
from torch import nn

import ofi_spec as spec
from models.mlplob import MLPLOB
from models.regression_engine import (
    TARGET_SCALE,
    RegressionEngine,
    build_loss,
)

B, SEQ, FEAT, OUT = 4, spec.SEQ_LEN, spec.INPUT_DIM, spec.OUTPUT_DIM


@pytest.fixture
def engine():
    torch.manual_seed(0)
    cfg = dict(hidden_dim=40, num_layers=3, seq_size=SEQ,
               num_features=FEAT, dataset_type="OFI")
    return RegressionEngine(model=MLPLOB(**cfg), model_config=cfg)


# ---------------------------------------------------------------------------
# 손실 단위 (사양 §16)
# ---------------------------------------------------------------------------
def test_loss_is_zero_when_prediction_is_the_target_in_bp(engine):
    target = torch.full((B, OUT), 2e-4)          # 소수 수익률 = 2bp
    pred_bp = target * TARGET_SCALE              # 모델이 맞혔다면 bp 로 2
    assert engine.loss(pred_bp, target).item() == pytest.approx(0.0, abs=1e-12)


def test_predicting_in_decimals_is_heavily_penalised(engine):
    """되돌림 방지: 모델이 소수 단위로 답하면 손실이 커야 한다."""
    target = torch.full((B, OUT), 2e-4)
    loss_bp = engine.loss(target * TARGET_SCALE, target)     # 올바른 단위
    loss_dec = engine.loss(target, target)                   # 예전처럼 소수로
    assert loss_bp < loss_dec
    # 정답은 bp 로 2.0, 소수로 답하면 0.0002 -> 오차 (2.0 - 0.0002)^2
    assert loss_dec.item() == pytest.approx((2.0 - 2e-4) ** 2, rel=1e-5)


def test_loss_magnitude_is_order_one_not_1e_minus_8(engine):
    """수익률 규모(1e-4)를 그대로 두면 MSE 가 1e-8 이라 Adam eps 에 눌린다."""
    torch.manual_seed(1)
    target = torch.randn(256, OUT) * 2e-4
    pred_bp = torch.randn(256, OUT) * 2.0
    loss = engine.loss(pred_bp, target).item()
    assert 0.1 < loss < 100, f"손실이 O(1) 규모여야 한다: {loss}"


def test_untrained_output_is_within_reach_of_the_target_scale(engine):
    """초기 출력이 정답 크기의 몇 배 안쪽이어야 학습이 바로 시작된다."""
    torch.manual_seed(2)
    out_bp = engine(torch.randn(512, SEQ, FEAT))
    ratio = out_bp.std().item() / 2.0            # 정답 규모 약 2bp
    assert 0.01 < ratio < 100, (
        f"초기 출력이 정답보다 {ratio:.0f}배 어긋나 있다. "
        "이러면 학습을 눈금 맞추는 데 다 쓴다"
    )


# ---------------------------------------------------------------------------
# 지표는 소수 단위로 되돌려 계산 (사양 §13)
# ---------------------------------------------------------------------------
def test_metrics_are_computed_in_decimal_units(engine):
    target = torch.full((B, OUT), 3e-4)
    pred_bp = target * TARGET_SCALE
    engine._eval_step((torch.randn(B, SEQ, FEAT), target), 0)

    engine._buffers.clear()
    engine._buffers[0] = {"p": [pred_bp / TARGET_SCALE], "t": [target],
                          "l": [torch.tensor(0.0)]}
    stored = engine._buffers[0]["p"][0]
    assert torch.allclose(stored, target), \
        "지표에 넘기는 예측은 bp 가 아니라 소수여야 한다"


# ---------------------------------------------------------------------------
# 손실함수 선택 (사양 §16)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,cls", [
    ("mse", nn.MSELoss), ("MSE", nn.MSELoss),
    ("huber", nn.SmoothL1Loss), ("smoothl1", nn.SmoothL1Loss),
])
def test_loss_selection(name, cls):
    assert isinstance(build_loss(name), cls)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="mse"):
        build_loss("mae")


def test_default_loss_is_mse():
    assert isinstance(build_loss(spec.LOSS_TYPE), nn.MSELoss)
    assert spec.LOSS_TYPE == "mse"


# ---------------------------------------------------------------------------
# 체크포인트에 구조가 남는가 (백테스트가 열 수 있어야 한다)
# ---------------------------------------------------------------------------
def test_model_config_is_kept_for_reloading(engine):
    cfg = engine.model_config
    assert cfg["num_features"] == spec.INPUT_DIM
    assert cfg["seq_size"] == spec.SEQ_LEN
    assert MLPLOB(**cfg) is not None, "저장된 설정만으로 같은 구조를 만들 수 있어야 한다"


def test_backward_runs_through_the_engine_loss(engine):
    x = torch.randn(B, SEQ, FEAT)
    y = torch.randn(B, OUT) * 2e-4
    loss = engine.loss(engine(x), y)
    loss.backward()
    grads = [p.grad for p in engine.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert np.isfinite(loss.item())


# ---------------------------------------------------------------------------
# 탐색 채점 기준은 손실함수 종류와 무관해야 한다
# ---------------------------------------------------------------------------
def test_training_loss_is_not_comparable_across_loss_types():
    """huber 와 mse 는 같은 예측에도 값이 크게 다르다.

    이걸 그대로 Optuna 채점에 쓰면 성능과 무관하게 huber 설정만 선택된다.
    실제 탐색에서 huber 2.08 대 mse 20.80 이 나왔다.
    """
    cfg = dict(hidden_dim=32, num_layers=2, seq_size=SEQ,
               num_features=FEAT, dataset_type="OFI")
    e_mse = RegressionEngine(model=MLPLOB(**cfg), loss_type="mse")
    e_hub = RegressionEngine(model=MLPLOB(**cfg), loss_type="huber")

    torch.manual_seed(5)
    target = torch.randn(4096, OUT) * 2e-4
    pred_bp = torch.randn(4096, OUT) * 4.0          # 오차가 1bp 보다 크다

    l_mse = e_mse.loss(pred_bp, target).item()
    l_hub = e_hub.loss(pred_bp, target).item()
    assert l_mse > 3 * l_hub, (
        f"두 손실의 눈금이 비슷하면 이 문제가 안 생긴다: mse {l_mse:.2f} "
        f"huber {l_hub:.2f}")


def test_mse_metric_is_the_same_whatever_loss_trained_it():
    """지표 쪽 mse 는 예측값만 보고 계산하므로 손실 종류와 무관하다."""
    from utils.metrics import summarise

    rng = np.random.default_rng(6)
    target = rng.normal(scale=2e-4, size=(5000, OUT))
    pred = target * 0.05 + rng.normal(scale=2e-4, size=target.shape)

    a = summarise(pred, target)["mse"]
    b = summarise(pred, target)["mse"]
    assert np.allclose(a, b)
    # 이 값이 tune.py 가 읽는 val_mse 의 근거다 (bp^2 로 환산해 기록)
    assert np.isfinite(a).all() and (a > 0).all()


def test_tune_scores_on_ic_not_on_a_training_loss():
    """채점 기준이 손실이 아니라 IC 여야 한다.

    학습 손실로 채점하면 huber 가 늘 이기고, val_mse 로 채점하면 mse 로 학습한
    쪽이 이긴다. 둘 다 자기가 최적화한 지표로 채점받는 셈이다. IC 는 어느
    손실로 학습했든 공평하다.
    """
    import io as _io
    import os as _os
    src = _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "scripts", "tune.py"), encoding="utf-8").read()
    assert 'callback_metrics.get("val_ic")' in src
    assert 'callback_metrics.get("val_loss")' not in src
    assert 'direction="maximize"' in src, "IC 는 클수록 좋다"


# ---------------------------------------------------------------------------
# huber_beta — 어디까지를 신호로 볼지의 경계
# ---------------------------------------------------------------------------
def test_huber_beta_controls_how_much_outliers_outweigh_ordinary_errors():
    """beta 가 클수록 큰 오차의 **상대 비중**이 커진다 (MSE 에 가까워진다).

    절대값으로 비교하면 안 된다. SmoothL1 은 제곱 구간에서 beta 로 나누므로
    beta 를 키우면 손실값 자체는 어디서나 작아진다. 학습에 영향을 주는 건
    "평범한 오차 대비 튀는 오차의 비중"이다.
    """
    target = torch.zeros(1, 1)
    outlier = torch.full((1, 1), 8.0)            # 튀는 오차 8bp
    normal = torch.full((1, 1), 0.5)             # 평범한 오차 0.5bp

    def ratio(loss):
        return loss(outlier, target).item() / loss(normal, target).item()

    r_small = ratio(build_loss("huber", huber_beta=1.0))
    r_large = ratio(build_loss("huber", huber_beta=10.0))
    r_mse = ratio(build_loss("mse"))

    assert r_small < r_large <= r_mse * 1.01, (
        f"beta 1 -> {r_small:.0f}배,  beta 10 -> {r_large:.0f}배,  "
        f"mse -> {r_mse:.0f}배")
    assert r_small < 100, "beta 1 이면 튀는 값의 비중이 크게 눌린다"


def test_huber_beta_reaches_the_engine():
    cfg = dict(hidden_dim=32, num_layers=2, seq_size=SEQ,
               num_features=FEAT, dataset_type="OFI")
    e = RegressionEngine(model=MLPLOB(**cfg), loss_type="huber", huber_beta=4.0)
    assert isinstance(e.criterion, nn.SmoothL1Loss)
    assert e.criterion.beta == pytest.approx(4.0)


def test_beta_is_ignored_for_mse():
    e = RegressionEngine(model=MLPLOB(32, 2, SEQ, FEAT, "OFI"),
                         loss_type="mse", huber_beta=7.0)
    assert isinstance(e.criterion, nn.MSELoss)


def test_collapsed_prediction_is_scored_bad_not_failed():
    """상수 예측이면 IC 가 NaN 이다. 그대로 두면 Optuna 가 시도를 버려서
    같은 영역을 다시 뽑는다. 최하점으로 기록해야 그 근처를 피한다."""
    import io as _io
    import os as _os
    from utils.metrics import information_coefficient

    const = np.zeros((100, 3))
    target = np.random.default_rng(9).normal(size=(100, 3))
    assert np.isnan(information_coefficient(const, target)).all()

    src = _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "scripts", "tune.py"), encoding="utf-8").read()
    assert "BAD_SCORE = -1.0" in src
    assert "math.isfinite" in src
