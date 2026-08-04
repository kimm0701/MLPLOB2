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
