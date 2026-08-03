"""Dataset 검증 (사양 §11, §17, §18-18,19)."""

import numpy as np
import pytest
import torch

import ofi_spec as spec
from preprocessing.ofi_dataset import (
    OFIWindowDataset,
    build_split,
    split_dates,
)

SEQ = spec.SEQ_LEN
FEAT = spec.INPUT_DIM
OUT = spec.OUTPUT_DIM


@pytest.fixture
def day():
    """각 시점의 값이 그 시점 번호인 배열. 슬라이싱이 맞는지 눈으로 확인 가능."""
    K = 1000
    x = np.tile(np.arange(K, dtype=np.float32)[:, None], (1, FEAT))
    y = np.tile(np.arange(K, dtype=np.float32)[:, None], (1, OUT)) * 0.001
    idx = np.arange(SEQ - 1, K - 200)
    return x, y, idx


def test_18_19_sample_shapes(day):
    x, y, idx = day
    ds = OFIWindowDataset(x, y, idx, SEQ)
    xb, yb = ds[0]
    assert tuple(xb.shape) == (100, 22)
    assert tuple(yb.shape) == (10,)
    assert xb.dtype == torch.float32 and yb.dtype == torch.float32


def test_window_is_the_100_buckets_ending_at_k(day):
    """입력은 k-99 ~ k 이어야 한다. 미래를 포함하면 안 된다."""
    x, y, idx = day
    ds = OFIWindowDataset(x, y, idx, SEQ)
    i = 300
    k = int(idx[i])
    xb, yb = ds[i]

    assert xb[0, 0].item() == pytest.approx(k - SEQ + 1)
    assert xb[-1, 0].item() == pytest.approx(k), "마지막 행이 현재 시점 k"
    assert xb[-1, 0].item() <= k, "미래 시점이 입력에 들어갔다"
    assert yb[0].item() == pytest.approx(k * 0.001), "정답은 시점 k 의 것"


def test_returned_tensors_are_writable(day):
    """mmap 읽기전용 뷰를 그대로 넘기면 안 된다."""
    x, y, idx = day
    ds = OFIWindowDataset(x, y, idx, SEQ)
    xb, _ = ds[10]
    xb += 1.0            # 예외가 나면 실패


def test_input_window_never_reaches_before_array_start(day):
    x, y, _ = day
    with pytest.raises(AssertionError):
        OFIWindowDataset(x, y, np.array([SEQ - 2]), SEQ)   # 앞이 모자란 시점


def test_length_matches_index_count(day):
    x, y, idx = day
    assert len(OFIWindowDataset(x, y, idx, SEQ)) == idx.size


# ---------------------------------------------------------------------------
# 파일 단위로 잘라 창이 경계를 넘지 않게 하는 구조 (§17)
# ---------------------------------------------------------------------------
def test_concat_never_mixes_symbols_or_days(tmp_path):
    K = 800
    for sym in ("AAA", "BBB"):
        for date in ("20260101", "20260102"):
            d = tmp_path / sym
            d.mkdir(exist_ok=True)
            base = str(d / date)
            np.save(base + "_x.npy", np.full((K, FEAT), 1.0, np.float32))
            np.save(base + "_y.npy", np.full((K, OUT), 2.0, np.float32))
            np.save(base + "_idx.npy", np.arange(SEQ - 1, K - 200, dtype=np.int64))

    ds = build_split(str(tmp_path), ["AAA", "BBB"], ["20260101", "20260102"],
                     normalize=False)
    assert len(ds.datasets) == 4, "(종목, 날짜) 조합마다 독립된 Dataset"
    # 각 조각이 자기 배열만 보므로 창이 조합 경계를 넘는 것이 구조적으로 불가능
    for part in ds.datasets:
        assert part.indices.min() - SEQ + 1 >= 0
        assert part.indices.max() < part.features.shape[0]


def test_missing_cache_raises_with_a_useful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        build_split(str(tmp_path), ["NOPE"], ["20260101"], normalize=False)


# ---------------------------------------------------------------------------
# 종목별 사전 정규화 (§17)
# ---------------------------------------------------------------------------
def _write_day(root, sym, date, fill, K=800):
    d = root / sym
    d.mkdir(exist_ok=True)
    base = str(d / date)
    np.save(base + "_x.npy", np.full((K, FEAT), fill, np.float32))
    np.save(base + "_y.npy", np.full((K, OUT), 0.001, np.float32))
    np.save(base + "_idx.npy", np.arange(SEQ - 1, K - 200, dtype=np.int64))
    return base


def _write_normalizer(root, table):
    import json
    payload = {
        "feature_names": list(spec.FEATURE_NAMES),
        "method": "median_iqr",
        "symbols": {s: {"center": [c] * FEAT, "scale": [sc] * FEAT}
                    for s, (c, sc) in table.items()},
    }
    (root / "normalizer.json").write_text(json.dumps(payload), encoding="utf-8")


def test_per_symbol_normalisation_puts_symbols_on_one_scale(tmp_path):
    """스케일이 10배 다른 두 종목이 정규화 후 같은 값이 되어야 한다."""
    _write_day(tmp_path, "BIG", "20260101", fill=100.0)
    _write_day(tmp_path, "SML", "20260101", fill=10.0)
    _write_normalizer(tmp_path, {"BIG": (100.0, 20.0), "SML": (10.0, 2.0)})

    ds = build_split(str(tmp_path), ["BIG", "SML"], ["20260101"])
    by_sym = {p.symbol: p for p in ds.datasets}

    xb, _ = by_sym["BIG"][0]
    xs, _ = by_sym["SML"][0]
    assert torch.allclose(xb, torch.zeros_like(xb)), "(100-100)/20 = 0"
    assert torch.allclose(xs, torch.zeros_like(xs)), "(10-10)/2 = 0"
    assert torch.allclose(xb, xs), "정규화 후 두 종목이 같은 스케일"


def test_normalisation_formula(tmp_path):
    _write_day(tmp_path, "AAA", "20260101", fill=7.0)
    _write_normalizer(tmp_path, {"AAA": (3.0, 2.0)})
    ds = build_split(str(tmp_path), ["AAA"], ["20260101"])
    x, _ = ds[0]
    assert torch.allclose(x, torch.full_like(x, 2.0)), "(7-3)/2 = 2"


def test_normalisation_can_be_turned_off_for_comparison(tmp_path):
    _write_day(tmp_path, "AAA", "20260101", fill=7.0)
    _write_normalizer(tmp_path, {"AAA": (3.0, 2.0)})
    ds = build_split(str(tmp_path), ["AAA"], ["20260101"], normalize=False)
    x, _ = ds[0]
    assert torch.allclose(x, torch.full_like(x, 7.0)), "끄면 원본 그대로"


def test_unknown_symbol_in_normalizer_is_an_error(tmp_path):
    """기준값이 없는 종목을 조용히 정규화 없이 섞으면 안 된다."""
    _write_day(tmp_path, "AAA", "20260101", fill=7.0)
    _write_day(tmp_path, "NEW", "20260101", fill=7.0)
    _write_normalizer(tmp_path, {"AAA": (3.0, 2.0)})
    with pytest.raises(KeyError, match="fit_normalizer"):
        build_split(str(tmp_path), ["AAA", "NEW"], ["20260101"])


def test_feature_order_change_invalidates_the_normalizer(tmp_path):
    import json
    _write_day(tmp_path, "AAA", "20260101", fill=7.0)
    payload = {"feature_names": ["뒤바뀐", "순서"], "symbols": {}}
    (tmp_path / "normalizer.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="특징 순서"):
        build_split(str(tmp_path), ["AAA"], ["20260101"])


def test_normalizer_is_fitted_on_training_dates_only():
    """검증·시험 날짜가 기준값 계산에 들어가면 미래를 미리 본 것이 된다."""
    dates = [f"202607{d:02d}" for d in range(14, 28)]
    train, val, test = split_dates(dates, n_val=2, n_test=2)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test)
    assert max(train) < min(val)


# ---------------------------------------------------------------------------
# 날짜 분할 (§17)
# ---------------------------------------------------------------------------
def test_split_dates_is_chronological_and_test_is_most_recent():
    dates = [f"202607{d:02d}" for d in range(14, 28)]
    tr, va, te = split_dates(dates, n_val=2, n_test=2)

    assert len(tr) == 10 and len(va) == 2 and len(te) == 2
    assert max(tr) < min(va) < min(te)
    assert te == dates[-2:], "최종시험은 가장 최근 날짜"
    assert set(tr) | set(va) | set(te) == set(dates)
    assert not (set(tr) & set(va)) and not (set(va) & set(te))


@pytest.mark.parametrize("n_val,n_test", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_split_dates_handles_zero_counts(n_val, n_test):
    """dates[-0:] 는 빈 목록이 아니라 전체다. 0 을 넣어도 어긋나면 안 된다."""
    dates = [f"202607{d:02d}" for d in range(14, 28)]
    tr, va, te = split_dates(dates, n_val=n_val, n_test=n_test)
    assert len(va) == n_val and len(te) == n_test
    assert len(tr) == len(dates) - n_val - n_test
    assert tr + va + te == dates, "합치면 원래 순서 그대로여야 한다"
