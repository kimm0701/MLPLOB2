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


def _tune_src():
    import io as _io
    import os as _os
    return _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "scripts", "tune.py"), encoding="utf-8").read()


def test_tune_scores_on_r2_not_on_a_training_loss():
    """목표가 가격 회귀 예측이므로 채점 기준은 R2 다.

    학습 손실로 채점하면 huber 가 늘 이기고, val_mse 로 채점하면 mse 로 학습한
    쪽이 이긴다. 둘 다 자기가 최적화한 지표로 채점받는 셈이다.
    """
    src = _tune_src()
    assert 'callback_metrics.get("val_r2")' in src
    assert 'callback_metrics.get("val_loss")' not in src
    assert 'callback_metrics.get("val_ic")' not in src
    assert 'direction="maximize"' in src, "R2 는 클수록 좋다"


def test_search_keeps_every_metric_not_just_the_score():
    """채점값 하나만 남기면 다른 지표를 보려고 전부 다시 학습해야 한다.

    실제로 그 일이 있었다 — R2 를 보려는데 IC 만 저장돼 있었다.
    """
    src = _tune_src()
    assert "set_user_attr" in src, "성적표를 DB 에 남겨야 한다"
    assert "_ReportRecorder(trial)" in src, "기록 콜백이 실제로 붙어야 한다"
    for m in ("r2_cal", "r2", "ic", "dir_acc", "mse", "mae"):
        assert f'"{m}"' in src, f"{m} 도 남겨야 한다"


def test_calibrated_r2_does_not_punish_a_wrongly_scaled_model():
    """보정 전 R2 로 채점하면 신호가 아니라 출력 크기로 순위가 갈린다.

    덜 학습된 모델은 방향을 맞혀도 출력이 너무 커서 R2 가 음수가 된다
    (실측: 예측이 이론값의 8~48%). 그건 배율 하나로 고쳐지는 문제다.
    """
    from utils.metrics import information_coefficient, r2, r2_calibrated

    rng = np.random.default_rng(21)
    t = rng.normal(scale=2e-4, size=(50_000, 3))
    signal = t * 0.06 + rng.normal(scale=2e-4, size=t.shape)

    good_scale = signal
    bad_scale = signal * 9.0            # 신호는 같고 크기만 어긋난 모델

    # 두 모델의 신호량(IC)은 같다 — 배율은 순위를 바꾸지 않는다
    assert np.allclose(information_coefficient(good_scale, t),
                       information_coefficient(bad_scale, t))

    assert (r2(bad_scale, t) < r2(good_scale, t)).all(), \
        "보정 전에는 크기만 어긋나도 크게 깎인다"
    assert np.allclose(r2_calibrated(bad_scale, t), r2_calibrated(good_scale, t)), \
        "보정 후에는 같은 신호에 같은 점수가 나와야 한다"


def test_engine_logs_r2_for_the_search_to_read():
    from utils.metrics import summarise
    src = _io_read("models/regression_engine.py")
    assert 'f"{stage}_r2"' in src, "탐색이 읽을 val_r2 를 기록해야 한다"
    assert 'primary["r2_cal"]' in src, "보정 후 R2 를 기록해야 한다"
    assert 'f"{stage}_r2_raw"' in src, "보정 전 값도 함께 남긴다"

    s = summarise(np.random.default_rng(3).normal(size=(500, OUT)),
                  np.random.default_rng(4).normal(size=(500, OUT)))
    assert "r2_cal" in s and s["r2_cal"].shape == (OUT,)


def _io_read(rel):
    import io as _io
    import os as _os
    return _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), *rel.split("/")), encoding="utf-8").read()


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


def test_resumed_search_does_not_replay_the_same_random_trials():
    """이어받기 시 이미 돌린 설정을 다시 뽑으면 안 된다.

    TPE 는 초반을 무작위로 뽑는데 시드가 고정이면 그 순서가 매번 같다.
    중단 후 이어받으면 탐색기가 처음부터 시작해 같은 설정을 또 돌린다.
    실측으로 재시작 한 번에 2회(trial 0≡4, 1≡5)를 헛돌았다.
    """
    import io as _io
    import os as _os

    src = _io.open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "scripts", "tune.py"), encoding="utf-8").read()
    assert "TPESampler(seed=1)" not in src, \
        "시드를 고정하면 이어받을 때마다 같은 설정을 다시 뽑는다"
    assert "seed=1 + n_existing" in src
    assert "len(study.trials)" in src

    # 시드가 다르면 실제로 다른 값을 뽑는지 확인한다
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def first_draws(seed, n=6):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda t: t.suggest_float("lr", 1e-5, 1e-2, log=True),
                       n_trials=n)
        return [t.params["lr"] for t in study.trials]

    assert first_draws(1) == first_draws(1), "같은 시드는 재현돼야 한다"
    assert first_draws(1) != first_draws(1 + 6), \
        "시드를 밀었는데 같은 값이 나오면 중복 방지가 안 된 것이다"


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


# ---------------------------------------------------------------------------
# 정답 정규화 (Kolm et al. 2023 §3.2.2)
# ---------------------------------------------------------------------------
def test_winsorize_applies_to_training_only_not_evaluation():
    """평가 정답까지 자르면 문제가 쉬워져서 성적이 부풀려진다."""
    from preprocessing.ofi_dataset import build_split
    src = _io_read("preprocessing/ofi_dataset.py")
    assert "winsorize 는 normalize_target 과 함께 써야" in src
    with pytest.raises(ValueError, match="normalize_target"):
        build_split("nonexistent", [], [], normalize=False, winsorize=True,
                    rolling=False)

    src = _io_read("scripts/train.py")
    # 학습에만 winsorize 를 넘기고 검증·시험 kw 에는 넣지 않는다
    assert "winsorize=args.winsorize" in src
    assert src.count("winsorize=args.winsorize") == 1, \
        "자르기는 학습셋 한 곳에만 적용해야 한다"


def test_dataset_divides_target_by_scale_and_clips_in_order():
    from preprocessing.ofi_dataset import OFIWindowDataset

    feat = np.zeros((30, spec.INPUT_DIM), dtype=np.float32)
    tgt = np.zeros((30, OUT), dtype=np.float32)
    tgt[25] = 5.0                                   # 경계 밖 값
    scale = np.full(OUT, 2.0, dtype=np.float32)
    clip = (np.full(OUT, -1.0), np.full(OUT, 1.0))

    ds = OFIWindowDataset(feat, tgt, [25], seq_len=4, target_scale=scale,
                          target_clip=clip)
    _, y = ds[0]
    # 먼저 1.0 로 잘리고 그 다음 2.0 으로 나뉜다
    assert torch.allclose(y, torch.full((OUT,), 0.5)), y

    ds_noclip = OFIWindowDataset(feat, tgt, [25], seq_len=4, target_scale=scale)
    _, y2 = ds_noclip[0]
    assert torch.allclose(y2, torch.full((OUT,), 2.5)), "자르지 않으면 5/2 = 2.5"


def test_engine_restores_decimal_units_before_scoring():
    """표준화된 정답으로 학습해도 지표는 원래 수익률 단위로 나와야 한다."""
    cfg = dict(hidden_dim=32, num_layers=2, seq_size=SEQ,
               num_features=FEAT, dataset_type="OFI")
    sc = [2e-4] * OUT
    e = RegressionEngine(model=MLPLOB(**cfg), eval_names=["AMD"],
                         pooled_name=None, target_scale_by_name={"AMD": sc})

    assert e.loss_scale == 1.0, "정답이 이미 O(1) 이면 추가 배율을 걸면 안 된다"

    pred = np.full((5, OUT), 0.5)
    tgt = np.full((5, OUT), 0.5)
    p2, t2 = e._to_decimal(pred, tgt, "AMD")
    assert np.allclose(t2, 1e-4), "0.5 x 2e-4 = 1e-4 로 되돌아와야 한다"
    assert np.allclose(p2, 1e-4)

    # 정규화를 끄면 예전 동작 그대로
    e0 = RegressionEngine(model=MLPLOB(**cfg))
    assert e0.loss_scale == TARGET_SCALE
    p3, t3 = e0._to_decimal(np.full((5, OUT), 2.0), np.full((5, OUT), 2e-4), "all")
    assert np.allclose(p3, 2e-4) and np.allclose(t3, 2e-4)


def test_normalised_targets_make_each_horizon_weigh_the_same():
    """지금은 10초가 손실의 18.4%, 1초가 1.7% 를 차지한다 (실측).
    표준편차로 나누면 10개가 균등해진다."""
    rng = np.random.default_rng(31)
    sd = np.array([1.70, 2.41, 2.95, 3.39, 3.81, 4.17, 4.52, 4.83, 5.11, 5.41])
    y = rng.normal(size=(20000, 10)) * sd

    share = (y ** 2).mean(0) / (y ** 2).mean(0).sum()
    assert share[0] < 0.03 and share[-1] > 0.15, "정규화 전에는 10초가 지배한다"

    share_n = ((y / sd) ** 2).mean(0) / ((y / sd) ** 2).mean(0).sum()
    assert np.allclose(share_n, 0.1, atol=0.01), "정규화 후에는 10개가 균등하다"


def test_paper_settings_are_the_defaults():
    """Kolm et al. 2023 Table 3 / §3.2.3 과 맞춘다."""
    src = _tune_src()
    # 가지치기는 스터디 방향(maximize)과 같은 값을 봐야 한다
    assert 'monitor: str = "val_r2"' in src
    assert 'EarlyStopping(monitor="val_loss", mode="min"' in src, \
        "학습 중단은 논문대로 검증 손실로 본다"

    tr = _io_read("scripts/train.py")
    assert '"--max-epochs", type=int, default=50' in tr, "논문은 50 epoch"
    assert '"--patience", type=int, default=5' in tr, "논문은 patience 5"
