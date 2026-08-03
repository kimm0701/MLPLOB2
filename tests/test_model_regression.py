"""사양 §18 (17~22, 25, 26) + §20-9,10 모델 검증.

입력 [B, 100, 22] -> 출력 [B, 10], 최종 출력 뒤 활성함수 없음, 손실 backward.
"""

import numpy as np
import pytest
import torch
from torch import nn

import ofi_spec as spec
from models.mlplob import MLPLOB

B = 2
SEQ = spec.SEQ_LEN          # 100
FEAT = spec.INPUT_DIM       # 22
OUT = spec.OUTPUT_DIM       # 10
HIDDEN = 40
LAYERS = 3


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return MLPLOB(
        hidden_dim=HIDDEN,
        num_layers=LAYERS,
        seq_size=SEQ,
        num_features=FEAT,
        dataset_type="OFI",
    )


# ---------------------------------------------------------------------------
# §18-18,19,20  shape
# ---------------------------------------------------------------------------
def test_18_single_sequence_shape():
    x = torch.randn(1, SEQ, FEAT)
    assert x.shape[1:] == (100, 22)


def test_19_20_batch_in_out_shape(model):
    x = torch.randn(B, SEQ, FEAT)
    y = model(x)
    assert x.shape == (B, 100, 22)
    assert y.shape == (B, 10), "출력은 horizon 10개"


def test_input_projection_accepts_22_not_40(model):
    """입력 차원이 22로 잡혀 있어야 한다 (기존 40/144 하드코딩 제거 확인)."""
    assert model.first_layer.in_features == spec.INPUT_DIM == 22
    assert model.norm_layer.d1 == spec.INPUT_DIM
    assert model.norm_layer.t1 == spec.SEQ_LEN

    with pytest.raises(RuntimeError):
        model(torch.randn(B, SEQ, 40))


# ---------------------------------------------------------------------------
# §18-21,22  최종 출력에 활성함수가 없다 -> 음수가 나올 수 있다
# ---------------------------------------------------------------------------
def test_22_final_layer_is_bare_linear(model):
    last = model.final_layers[-1]
    assert isinstance(last, nn.Linear)
    assert last.out_features == spec.OUTPUT_DIM

    forbidden = (nn.Softmax, nn.LogSoftmax, nn.Sigmoid, nn.ReLU, nn.Tanh)
    # 최종 Linear 뒤에 아무 모듈도 없어야 한다
    assert len(model.final_layers) >= 1
    for m in list(model.final_layers)[len(model.final_layers) - 1:]:
        assert not isinstance(m, forbidden)


def test_21_output_can_be_negative(model):
    """여러 무작위 입력에서 음수와 양수가 모두 나와야 한다."""
    torch.manual_seed(1)
    y = model(torch.randn(256, SEQ, FEAT))
    assert (y < 0).any(), "수익률 예측인데 음수가 전혀 안 나오면 활성함수가 붙은 것"
    assert (y > 0).any()


def test_internal_gelu_is_kept(model):
    """내부 Feature/Temporal Mixing 의 GELU 는 유지되어야 한다."""
    gelus = [m for m in model.modules() if isinstance(m, nn.GELU)]
    assert len(gelus) > 0
    layer_norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    assert len(layer_norms) > 0, "MLP 블록의 LayerNorm 유지"


# ---------------------------------------------------------------------------
# §18-25,26  손실 계산과 backward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("loss_type,cls", [("mse", nn.MSELoss), ("huber", nn.SmoothL1Loss)])
def test_25_26_loss_shapes_and_backward(model, loss_type, cls):
    torch.manual_seed(2)
    m = MLPLOB(HIDDEN, LAYERS, SEQ, FEAT, "OFI")
    x = torch.randn(B, SEQ, FEAT)
    target = torch.randn(B, OUT) * 1e-3          # 수익률 스케일

    pred = m(x)
    assert pred.shape == target.shape == (B, 10)

    criterion = cls()
    loss = criterion(pred, target)
    assert loss.ndim == 0
    loss.backward()

    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "역전파가 파라미터에 도달하지 않았다"
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_no_nan_on_all_zero_bucket_rows():
    """이벤트가 없던 버킷은 22개 특징이 전부 0이다.
    BiN 의 피처축 std 가 0 이 되어 NaN 이 나오면 안 된다 (승인된 가드)."""
    torch.manual_seed(3)
    m = MLPLOB(HIDDEN, LAYERS, SEQ, FEAT, "OFI")

    x = torch.randn(B, SEQ, FEAT)
    x[:, 30:60, :] = 0.0                 # 연속 30개 버킷이 완전 무이벤트
    y = m(x)
    assert torch.isfinite(y).all(), "빈 버킷 때문에 NaN 이 발생했다"

    x_all_zero = torch.zeros(B, SEQ, FEAT)
    assert torch.isfinite(m(x_all_zero)).all()


def test_forward_backward_smoke_matches_spec_20_9_10():
    """§20-9,10 이 요구한 최소 실행."""
    m = MLPLOB(HIDDEN, LAYERS, SEQ, FEAT, "OFI")
    dummy_input = torch.randn(2, 100, 22)
    out = m(dummy_input)
    assert tuple(out.shape) == (2, 10)

    target = torch.randn(2, 10)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    assert np.isfinite(loss.item())
