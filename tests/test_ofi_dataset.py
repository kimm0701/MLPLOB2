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

    ds = build_split(str(tmp_path), ["AAA", "BBB"], ["20260101", "20260102"])
    assert len(ds.datasets) == 4, "(종목, 날짜) 조합마다 독립된 Dataset"
    # 각 조각이 자기 배열만 보므로 창이 조합 경계를 넘는 것이 구조적으로 불가능
    for part in ds.datasets:
        assert part.indices.min() - SEQ + 1 >= 0
        assert part.indices.max() < part.features.shape[0]


def test_missing_cache_raises_with_a_useful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        build_split(str(tmp_path), ["NOPE"], ["20260101"])


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
