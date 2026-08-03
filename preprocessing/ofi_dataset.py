"""회귀용 Dataset (사양 §11, §14, §17).

한 샘플 = 입력 [SEQ_LEN, 22] + 정답 [10].

(종목, 날짜) 조합마다 별도의 Dataset 을 만들고 ConcatDataset 으로 잇는다.
이렇게 하면 입력 구간이 날짜 경계나 종목 경계를 넘는 일이 **구조적으로**
불가능하다 — 경계를 넘으면 다른 종목의 호가나 하루 건너뛴 시각이 한 창 안에
섞여 들어간다.

전처리 결과는 .npy 로 저장하고 mmap 으로 연다. 평일 14일 x 3종목이면 특징만
6 GB 가 넘어서 전부 메모리에 올리면 24 GB 인스턴스가 버겁다. mmap 은 실제로
읽는 부분만 메모리에 올린다.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

import ofi_spec as spec


class OFIWindowDataset(Dataset):
    """하나의 (종목, 날짜) 에 대한 슬라이딩 윈도우 Dataset."""

    def __init__(self, features, targets, indices, seq_len: int = spec.SEQ_LEN,
                 symbol: str = "", date: str = "", center=None, scale=None):
        self.features = features          # [K, 22]
        self.targets = targets            # [K, 10]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = seq_len
        self.symbol = symbol
        self.date = date
        # 종목별 사전 정규화. 저장된 배열은 건드리지 않고 읽을 때만 적용해서,
        # 껐다 켠 비교와 종목 추가가 재전처리 없이 가능하게 한다.
        self.center = None if center is None else np.asarray(center, np.float32)
        self.scale = None if scale is None else np.asarray(scale, np.float32)

        if self.indices.size:
            assert self.indices.min() - seq_len + 1 >= 0, "입력 구간이 배열 앞을 벗어난다"
            assert self.indices.max() < features.shape[0]

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, i: int):
        k = int(self.indices[i])
        # np.array 로 복사한다. mmap 슬라이스는 읽기 전용이라
        # torch.from_numpy 가 경고를 내고, 이후 어떤 in-place 연산도 위험해진다.
        x = np.array(self.features[k - self.seq_len + 1: k + 1], dtype=np.float32)
        y = np.array(self.targets[k], dtype=np.float32)
        if self.center is not None:
            x -= self.center
            x /= self.scale
        return torch.from_numpy(x), torch.from_numpy(y)


def load_normalizer(cache_dir: str, path: str | None = None) -> dict:
    """scripts/fit_normalizer.py 가 저장한 종목별 기준값. 없으면 빈 dict."""
    path = path or os.path.join(cache_dir, "normalizer.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("feature_names") != list(spec.FEATURE_NAMES):
        raise ValueError(
            f"{path} 의 특징 순서가 현재 사양과 다릅니다. "
            "fit_normalizer.py 를 다시 실행하세요."
        )
    return payload.get("symbols", {})


def load_day(cache_dir: str, symbol: str, date: str, seq_len: int = spec.SEQ_LEN,
             mmap: bool = True, norm: dict | None = None) -> OFIWindowDataset | None:
    """전처리해 둔 하루치를 읽어 Dataset 으로 만든다. 없으면 None."""
    base = os.path.join(cache_dir, symbol, date)
    fx, fy, fi = base + "_x.npy", base + "_y.npy", base + "_idx.npy"
    if not (os.path.exists(fx) and os.path.exists(fy) and os.path.exists(fi)):
        return None

    mode = "r" if mmap else None
    x = np.load(fx, mmap_mode=mode)
    y = np.load(fy, mmap_mode=mode)
    idx = np.load(fi)
    if idx.size == 0:
        return None

    center = scale = None
    if norm:
        st = norm.get(symbol)
        if st is None:
            raise KeyError(
                f"{symbol} 의 정규화 기준값이 없습니다. "
                f"scripts/fit_normalizer.py --symbols {symbol} 를 실행하세요."
            )
        center, scale = st["center"], st["scale"]

    return OFIWindowDataset(x, y, idx, seq_len, symbol, date, center, scale)


def build_split(cache_dir: str, symbols, dates, seq_len: int = spec.SEQ_LEN,
                mmap: bool = True, normalize: bool = True,
                normalizer_path: str | None = None) -> ConcatDataset:
    """여러 종목 x 여러 날짜를 하나의 Dataset 으로 잇는다.

    normalize=True 면 종목별 사전 정규화를 적용한다. 깊이 정규화만으로는
    종목 간 스케일이 최대 68배까지 벌어져서, 그대로 합치면 모델이 흐름 패턴
    대신 종목 정체성을 학습한다.
    """
    norm = load_normalizer(cache_dir, normalizer_path) if normalize else {}
    if normalize and not norm:
        raise FileNotFoundError(
            f"{cache_dir}/normalizer.json 이 없습니다.\n"
            "scripts/fit_normalizer.py 를 먼저 실행하거나 normalize=False 로 끄세요."
        )

    parts = []
    for sym in symbols:
        for date in dates:
            ds = load_day(cache_dir, sym, date, seq_len, mmap, norm)
            if ds is not None:
                parts.append(ds)
    if not parts:
        raise FileNotFoundError(
            f"전처리 결과가 없습니다: {cache_dir} / {symbols} / {dates}\n"
            "scripts/preprocess.py 를 먼저 실행하세요."
        )
    return ConcatDataset(parts)


def describe(ds: ConcatDataset) -> str:
    parts = ds.datasets
    n = sum(len(p) for p in parts)
    syms = sorted({p.symbol for p in parts})
    dates = sorted({p.date for p in parts})
    return (f"샘플 {n:,}개  ({len(parts)}개 파일, 종목 {','.join(syms)}, "
            f"{dates[0]}~{dates[-1]})")


# ----------------------------------------------------------------------------
# 날짜 단위 시간순 분할 (§17)
# ----------------------------------------------------------------------------
def split_dates(dates, n_val: int = 2, n_test: int = 2):
    """날짜 목록을 학습/검증/최종시험으로 시간순 분할.

    날짜 단위로 자르면 한 샘플의 입력·정답 구간이 통째로 하루 안에 들어가므로
    구간 사이 침범이 원천적으로 없다. 최종시험은 **가장 최근** 날짜를 쓴다.
    """
    dates = sorted(dates)
    assert n_val >= 0 and n_test >= 0, (n_val, n_test)
    assert len(dates) > n_val + n_test, f"날짜가 너무 적다: {len(dates)}"

    # 음수 인덱스 슬라이싱은 0 에서 뒤집힌다. dates[-0:] 은 빈 목록이 아니라
    # 전체 목록이라, n_test=0 이면 시험 구간이 전부가 되어 버린다.
    n = len(dates)
    train = dates[: n - n_val - n_test]
    val = dates[n - n_val - n_test: n - n_test]
    test = dates[n - n_test:] if n_test else []
    return train, val, test
