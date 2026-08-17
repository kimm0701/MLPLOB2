"""LSTM 구조가 MLPLOB 자리에 그대로 들어가는지, 인과성이 지켜지는지 확인한다."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                   # noqa: E402
from models.lstmlob import LSTMLOB                        # noqa: E402
from models.mlplob import MLPLOB                          # noqa: E402
from models.registry import build_model                   # noqa: E402

HIDDEN, LAYERS, SEQ, FEAT = 32, 2, spec.SEQ_LEN, spec.INPUT_DIM


def cfg(**kw):
    base = dict(arch="lstm", hidden_dim=HIDDEN, num_layers=LAYERS,
                seq_size=SEQ, num_features=FEAT, dataset_type="OFI")
    base.update(kw)
    return base


def test_출력모양이_사양과_같다():
    m = build_model(cfg())
    out = m(torch.randn(8, SEQ, FEAT))
    assert out.shape == (8, spec.OUTPUT_DIM)


def test_MLPLOB_과_생성인자가_호환된다():
    """같은 자리에 끼울 수 있어야 학습 스크립트를 안 건드린다."""
    args = (HIDDEN, LAYERS, SEQ, FEAT, "OFI")
    a, b = MLPLOB(*args), LSTMLOB(*args)
    x = torch.randn(4, SEQ, FEAT)
    assert a(x).shape == b(x).shape


def test_마지막층에_활성함수가_없다():
    """수익률은 음수도 된다. 출력이 한쪽으로 잘리면 안 된다.

    가중치만 무작위로 흔들고 부호를 보면 불안정하다. 초기 LSTM 은 은닉값이
    거의 0 이라 출력이 사실상 편향값만 남는데, 편향 4개가 우연히 모두 양수일
    확률이 1/16 이다. 실제로 그 확률로 실패했다. 그래서 시드를 고정하고,
    은닉값이 0 이 아니도록 가중치를 함께 키운다.
    """
    torch.manual_seed(0)
    m = build_model(cfg())
    with torch.no_grad():
        for p in m.lstm.parameters():
            p.uniform_(-0.5, 0.5)
        for p in m.head.parameters():
            p.uniform_(-3, 3)
        out = m(torch.randn(256, SEQ, FEAT))
    assert (out < 0).any() and (out > 0).any(), "부호 양쪽이 다 나와야 한다"
    # 활성함수가 붙었다면 한쪽이 정확히 0 에 눌리거나 범위가 잘린다
    assert out.min() < -0.1 and out.max() > 0.1


def test_미래를_보지_않는다():
    """마지막 시점 출력은 그 시점까지의 입력에만 의존해야 한다.

    LSTM 은 단방향이므로 t 시점 은닉값은 t 이후 입력과 무관하다. 우리는
    마지막 시점만 쓰므로 이 검사는 '창 끝을 잘라도 앞쪽 출력이 같은가' 로 본다.
    """
    m = build_model(cfg()).eval()
    x = torch.randn(2, SEQ, FEAT)
    with torch.no_grad():
        # BiN 은 창 전체를 보고 정규화하므로 이 검사에서는 끈다
        m2 = build_model(cfg(use_bin=False)).eval()
        m2.load_state_dict({k: v for k, v in m.state_dict().items()
                            if not k.startswith("norm_layer.")}, strict=False)
        full, _ = m2.lstm(x)
        half, _ = m2.lstm(x[:, :SEQ // 2])
    assert torch.allclose(full[:, SEQ // 2 - 1], half[:, -1], atol=1e-5), \
        "앞쪽 시점의 은닉값이 뒤쪽 입력에 영향을 받으면 미래 참조다"


def test_체크포인트로_구조가_복원된다(tmp_path):
    from models.regression_engine import (RegressionEngine,
                                          load_model_from_checkpoint)

    c = cfg()
    eng = RegressionEngine(model=build_model(c), model_config=c)
    path = str(tmp_path / "m.ckpt")
    torch.save({"state_dict": eng.state_dict(),
                "hyper_parameters": {"model_config": c}}, path)
    m, _ = load_model_from_checkpoint(path)
    assert isinstance(m, LSTMLOB)
    assert m(torch.randn(2, SEQ, FEAT)).shape == (2, spec.OUTPUT_DIM)


def test_arch_없는_예전_체크포인트는_MLPLOB_로_본다():
    old = dict(hidden_dim=HIDDEN, num_layers=LAYERS, seq_size=SEQ,
               num_features=FEAT, dataset_type="OFI")
    assert isinstance(build_model(old), MLPLOB)


def test_모르는_구조는_거부한다():
    with pytest.raises(KeyError):
        build_model(cfg(arch="transformer"))


def test_단층이면_dropout_경고가_없다():
    """PyTorch 는 num_layers=1 에 dropout 을 주면 경고를 낸다."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_model(cfg(num_layers=1, dropout=0.3))


def test_손실을_고정하면_탐색하지_않는다():
    """--loss 로 고정했는데도 탐색공간에 남아 있으면 시도를 나눠 먹는다.

    실제로 손실을 열어둔 채 돌렸더니 은닉 64 시도가 우연히 전부 huber 라서
    huber 가 이긴 것처럼 보였다. 같은 은닉으로 붙은 유일한 비교(128)에서는
    mse 가 이겼다 — 손실이 아니라 크기의 효과였다.
    """
    import io as _io
    import os as _os
    src = _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "scripts", "tune.py"), encoding="utf-8").read()
    assert 'loss = args.loss or trial.suggest_categorical' in src
    assert '"--loss", default=None, choices=["mse", "huber"]' in src
