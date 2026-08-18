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


def test_날짜분할이_한곳에서만_정해진다():
    """분할 기본값이 스크립트마다 흩어져 있으면 조용한 누출이 생긴다.

    make_targets 가 n_val=2 로 level·winsorize 경계를 뽑았는데 train.py 가
    n_val=4 로 돌면, 그 경계는 지금은 검증일인 날짜에서 나온 값이 된다.
    성적은 좋아 보이고 원인은 안 보인다.
    """
    import glob as _glob
    import io as _io
    import os as _os

    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    bad = []
    for f in _glob.glob(_os.path.join(root, "scripts", "*.py")):
        src = _io.open(f, encoding="utf-8").read()
        for flag in ("--n-val", "--n-test"):
            if f'"{flag}", type=int, default=' in src:
                line = src.split(f'"{flag}", type=int, default=')[1][:20]
                if not line.startswith("spec.N_"):
                    bad.append(f"{_os.path.basename(f)} {flag}")
    assert not bad, f"spec.N_VAL/N_TEST 를 쓰지 않는 곳: {bad}"

    assert spec.N_VAL >= 1 and spec.N_TEST >= 1


def test_split_dates가_spec을_따른다():
    from preprocessing.ofi_dataset import split_dates

    days = [f"2026{m:02d}{d:02d}" for m in (7, 8) for d in range(1, 16)]
    tr, va, te = split_dates(days)
    assert len(va) == spec.N_VAL and len(te) == spec.N_TEST
    assert len(tr) + len(va) + len(te) == len(days)
    # 시험은 가장 최근이어야 한다 — 과거로 미래를 시험하면 의미가 없다
    assert max(tr) < min(va) < max(va) < min(te)


def test_sigma가_어느_지점_이후도_보지_않는다():
    """기존 테스트는 '뒤쪽 절반'만 봐서 하한 누출을 놓쳤다.

    하한을 그날 앞쪽 구간에서 뽑던 판본은 시점 200 의 sigma 가 시점 3000 의
    데이터에 좌우됐다. 하루 앞쪽 0.3% 안에서 벌어진 일이라 '절반' 검사의
    사각지대였다. 이제 **여러 절단점**에서 확인한다.
    """
    import numpy as np
    from scripts.make_targets import causal_sigma_sec

    rng = np.random.default_rng(0)
    n = 20000
    ts = np.cumsum(rng.integers(20, 120, n)).astype(float)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    base = causal_sigma_sec(mid, ts, 0.01, 0.5, 1500, floor=0.5)

    for cut in (200, 1000, 3000, 8000, n // 2):
        m2 = mid.copy()
        m2[cut:] += np.cumsum(rng.normal(0, 0.5, n - cut))
        s2 = causal_sigma_sec(m2, ts, 0.01, 0.5, 1500, floor=0.5)
        assert np.allclose(base[:cut], s2[:cut]), \
            f"{cut} 이후를 바꿨는데 그 앞 sigma 가 변했다 = 미래 참조"


def test_하한은_전날에서_온다():
    """같은 날 데이터로 하한을 만들면 그 자체가 미래 참조다."""
    import inspect

    from scripts import make_targets as mt

    src = inspect.getsource(mt._apply_floor)
    assert "percentile" not in src, \
        "_apply_floor 안에서 분위수를 구하면 그날 데이터를 쓰는 것이다"
    assert "floor" in inspect.signature(mt.causal_sigma_sec).parameters

    # 첫날은 전날이 없으므로 절대 하한만
    import numpy as np
    sig = np.array([1e-9, 0.5, 2.0])
    assert mt._apply_floor(sig, None)[0] == mt.ABS_MIN_SIGMA
    assert np.allclose(mt._apply_floor(sig, 0.3), [0.3, 0.5, 2.0])
    # 전날 하한이 물리적 하한보다 작아도 물리적 하한이 이긴다
    assert mt._apply_floor(sig, 1e-9)[0] == mt.ABS_MIN_SIGMA
    assert mt.ABS_MIN_SIGMA >= 0.001, "너무 작으면 조용한 구간에서 z 가 폭발한다"


def test_깨진_호가창_스냅샷을_무효로_본다():
    """한쪽만 갱신된 스냅샷은 시장 상태가 아니라 캡처 결함이다.

    실측: META 20260714 idx4 에서 매수가 521.76 에 멈춰 있고 매도만 653.19 로
    들어와 스프레드가 13,143틱이 됐다. 0.02초 뒤 652.19 로 정상화된다.
    그 한 지점이 |z| 를 7.79e5 까지 밀어 올렸다.

    진짜 넓은 스프레드(p99.99 = 32~36틱)는 걸리면 안 된다 - 그건 학습해야 할
    시장 상태이고, log_spread_ticks 로 이미 모델에 들어간다.
    """
    from scripts.make_targets import MAX_SPREAD_TICKS

    assert MAX_SPREAD_TICKS >= 50, "실제 넓은 스프레드(p99.99 32~36틱)를 자르면 안 된다"
    assert MAX_SPREAD_TICKS <= 500, "이보다 크면 깨진 스냅샷을 못 거른다"


def test_일중계수는_과거_날짜에서만_온다():
    """계수표에 그날 데이터가 들어가면 정답이 미래를 본다.

    실측(1분 해상도, 학습 17일): 개장 13:30 UTC 가 하루 평균의 6~10배다.
    EWMA 반감기가 개장 기준 27초라 6배 점프를 따라잡는 데 90초가 걸리는데
    급첨은 1분 만에 끝난다. 그래서 계수로 미리 반영한다 - 다만 그 계수는
    반드시 지난 날들에서만 나와야 한다.
    """
    import numpy as np

    from scripts.make_targets import (DIURNAL_MIN_DAYS, back_returns,
                                      causal_sigma_sec, day_profile,
                                      diurnal_coef)

    rng = np.random.default_rng(3)
    n = 20000
    ts = np.cumsum(rng.integers(20, 120, n)).astype(float)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))

    # 과거가 부족하면 계수를 쓰지 않는다
    prof = day_profile(back_returns(mid, ts, 0.01, 0.5), ts)
    assert np.allclose(diurnal_coef([prof] * (DIURNAL_MIN_DAYS - 1), ts), 1.0)
    assert not np.allclose(diurnal_coef([prof] * DIURNAL_MIN_DAYS, ts), 1.0)

    # 계수를 써도 인과성이 깨지지 않는다
    coef = diurnal_coef([prof] * 5, ts)
    base = causal_sigma_sec(mid, ts, 0.01, 0.5, 1500, floor=0.5, coef=coef)
    for cut in (200, 1000, 3000, 8000):
        m2 = mid.copy()
        m2[cut:] += np.cumsum(rng.normal(0, 0.5, n - cut))
        s2 = causal_sigma_sec(m2, ts, 0.01, 0.5, 1500, floor=0.5, coef=coef)
        assert np.allclose(base[:cut], s2[:cut]), f"{cut} 이후가 그 앞에 영향"

    # 계수가 클수록 sigma 가 크다 (모양이 반영된다)
    c2 = np.where(np.arange(n) > n // 2, 4.0, 1.0)
    s = causal_sigma_sec(mid, ts, 0.01, 0.5, 1500, floor=1e-6, coef=c2)
    assert np.median(s[n // 2 + 2000:]) > 2 * np.median(s[:n // 2])
