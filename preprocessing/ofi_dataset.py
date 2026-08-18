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
                 symbol: str = "", date: str = "", center=None, scale=None,
                 target_scale=None, target_clip=None,
                 sigma=None, price=None, tick=1.0, level=1.0):
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

        # 정답 정규화 (Kolm et al. 2023 §3.2.2). 나눗셈만 하고 평균은 빼지 않는다.
        self.target_scale = (None if target_scale is None
                             else np.asarray(target_scale, np.float32))
        # winsorize 경계. **학습 구간에만** 준다. 검증·시험에서 자르면 문제
        # 자체가 쉬워져서 성적이 부풀려지므로, 평가는 자르지 않은 정답으로 한다.
        self.target_clip = (None if target_clip is None
                            else (np.asarray(target_clip[0], np.float32),
                                  np.asarray(target_clip[1], np.float32)))
        # 롤링 정규화용. sigma 가 있으면 정답은 틱 단위로 들어온 것으로 본다.
        self.sigma = sigma
        self.price = price
        self.tick = float(tick)
        self.level = float(level)

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
            # 뒤쪽 상태 변수(스프레드·경과시간)는 제외한다. 종목별 z-score 를
            # 씌우면 종목 간 차이가 정확히 0 으로 지워져 넣은 의미가 없어진다.
            m = x.shape[1] - spec.STATE_FEATURES
            x[:, :m] -= self.center[:m]
            x[:, :m] /= self.scale[:m]

        if self.sigma is not None:
            # 틱 단위 정답 -> 무차원.  sigma 는 인과적(과거만 봄), level 은
            # 학습 구간에서 구한 종목 상수.
            denom = float(self.sigma[k]) * self.level
            y = y / denom
            if self.target_clip is not None:          # 학습에만 적용
                np.clip(y, self.target_clip[0], self.target_clip[1], out=y)
            # 예측을 원래 수익률로 되돌릴 배율:  z * scale = 소수 수익률
            scale = np.float32(denom * self.tick / max(float(self.price[k]), 1e-9))
            return (torch.from_numpy(x), torch.from_numpy(y),
                    torch.tensor([scale], dtype=torch.float32))

        if self.target_clip is not None:
            np.clip(y, self.target_clip[0], self.target_clip[1], out=y)
        if self.target_scale is not None:
            y /= self.target_scale
        return torch.from_numpy(x), torch.from_numpy(y)


def load_normalizer(cache_dir: str, path: str | None = None) -> dict:
    """scripts/fit_normalizer.py 가 저장한 종목별 기준값. 없으면 빈 dict."""
    path = path or os.path.join(cache_dir, "normalizer.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    # 정규화 대상은 주문흐름 22개뿐이다. 상태 변수 2개는 일부러 제외하므로
    # 기준값 파일에도 들어 있지 않다.
    if payload.get("feature_names") != list(spec.OF_FEATURE_NAMES):
        raise ValueError(
            f"{path} 의 특징 순서가 현재 사양과 다릅니다. "
            "fit_normalizer.py 를 다시 실행하세요."
        )
    return payload.get("symbols", {})


def load_targets_meta(cache_dir: str) -> dict:
    """scripts/make_targets.py 가 남긴 종목별 틱·Delta_t·수준 보정."""
    p = os.path.join(cache_dir, "targets.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_day(cache_dir: str, symbol: str, date: str, seq_len: int = spec.SEQ_LEN,
             mmap: bool = True, norm: dict | None = None,
             normalize_target: bool = False,
             winsorize: bool = False,
             tmeta: dict | None = None) -> OFIWindowDataset | None:
    """전처리해 둔 하루치를 읽어 Dataset 으로 만든다. 없으면 None.

    tmeta 가 있으면 새 방식이다 - 입력은 _x2.npy(24개), 정답은 _yt.npy(틱)를
    _sg.npy(인과적 sigma)로 나눈 무차원 값.
    """
    base = os.path.join(cache_dir, symbol, date)
    if tmeta and symbol in tmeta:
        need = [base + s for s in ("_x2.npy", "_yt.npy", "_sg.npy",
                                   "_px.npy", "_idx.npy")]
        if not all(os.path.exists(f) for f in need):
            return None
        mode = "r" if mmap else None
        st = tmeta[symbol]
        idx = np.load(base + "_idx.npy")
        if idx.size == 0:
            return None
        x = np.load(base + "_x2.npy", mmap_mode=mode)
        yt = np.load(base + "_yt.npy", mmap_mode=mode)
        # 하루 끝에서는 t+h 초가 배열을 넘어 정답이 NaN 이다. 몇 칸인지는
        # 이벤트 밀도에 따라 다르므로(초 단위 horizon 을 칸 수로 환산할 수
        # 없다) **정답을 직접 보고** 걸러낸다. 예전에는 horizon 값을 버킷
        # 수로 오해해 끝 2 칸만 잘랐고, 남은 NaN 이 학습 손실을 통째로
        # NaN 으로 만들었다.
        idx = idx[idx < min(x.shape[0], yt.shape[0])]
        if idx.size:
            idx = idx[np.isfinite(np.asarray(yt[idx])).all(axis=1)]
        if idx.size == 0:
            return None
        center = scale = None
        if norm:
            sy = norm.get(symbol)
            if sy is None:
                raise KeyError(f"{symbol} 의 정규화 기준값이 없습니다.")
            center, scale = sy["center"], sy["scale"]
        clip = None
        if winsorize:
            # 경계는 targets.json 에 z 공간으로 들어 있다. 없으면 **에러를
            # 낸다** - 예전에 조용히 건너뛰어서 --winsorize 가 아무 일도 하지
            # 않은 채 학습이 끝난 적이 있다.
            if st.get("clip_lo") is None:
                raise KeyError(
                    f"{symbol} 의 winsorize 경계가 없습니다. "
                    "scripts/make_targets.py 를 다시 실행하세요.")
            clip = (st["clip_lo"], st["clip_hi"])
        return OFIWindowDataset(
            x, yt, idx, seq_len,
            symbol, date, center, scale, None, clip,
            sigma=np.load(base + "_sg.npy", mmap_mode=mode),
            price=np.load(base + "_px.npy", mmap_mode=mode)[:, 0],
            tick=st["tick"], level=st.get("level", 1.0))

    fx, fy, fi = base + "_x.npy", base + "_y.npy", base + "_idx.npy"
    if not (os.path.exists(fx) and os.path.exists(fy) and os.path.exists(fi)):
        return None

    mode = "r" if mmap else None
    x = np.load(fx, mmap_mode=mode)
    y = np.load(fy, mmap_mode=mode)
    idx = np.load(fi)
    if idx.size == 0:
        return None

    center = scale = t_scale = t_clip = None
    if norm:
        st = norm.get(symbol)
        if st is None:
            raise KeyError(
                f"{symbol} 의 정규화 기준값이 없습니다. "
                f"scripts/fit_normalizer.py --symbols {symbol} 를 실행하세요."
            )
        center, scale = st["center"], st["scale"]
        if normalize_target:
            if "target_scale" not in st:
                raise KeyError(
                    f"{symbol} 에 정답 기준값이 없습니다. "
                    "scripts/fit_normalizer.py 를 다시 실행하세요."
                )
            t_scale = st["target_scale"]
            if winsorize:
                t_clip = (st["target_clip_lo"], st["target_clip_hi"])

    return OFIWindowDataset(x, y, idx, seq_len, symbol, date, center, scale,
                            t_scale, t_clip)


def target_scales(cache_dir: str, symbols, normalizer_path: str | None = None) -> dict:
    """종목별 정답 표준편차. 평가 때 예측을 원래 단위로 되돌리는 데 쓴다."""
    norm = load_normalizer(cache_dir, normalizer_path)
    return {s: norm[s]["target_scale"] for s in symbols
            if s in norm and "target_scale" in norm[s]}


def build_split(cache_dir: str, symbols, dates, seq_len: int = spec.SEQ_LEN,
                mmap: bool = True, normalize: bool = True,
                normalizer_path: str | None = None,
                normalize_target: bool = False,
                winsorize: bool = False,
                rolling: bool = True) -> ConcatDataset:
    """여러 종목 x 여러 날짜를 하나의 Dataset 으로 잇는다.

    normalize=True 면 종목별 사전 정규화를 적용한다. 깊이 정규화만으로는
    종목 간 스케일이 최대 68배까지 벌어져서, 그대로 합치면 모델이 흐름 패턴
    대신 종목 정체성을 학습한다.

    normalize_target=True 면 정답을 horizon·종목별 표준편차로 나눈다.
    winsorize=True 는 **학습 구간에서만** 켠다 — 평가 정답을 자르면 문제가
    쉬워져서 성적이 부풀려진다.
    """
    norm = load_normalizer(cache_dir, normalizer_path) if normalize else {}
    if normalize and not norm:
        raise FileNotFoundError(
            f"{cache_dir}/normalizer.json 이 없습니다.\n"
            "scripts/fit_normalizer.py 를 먼저 실행하거나 normalize=False 로 끄세요."
        )
    if winsorize and not normalize_target and not rolling:
        raise ValueError("winsorize 는 normalize_target 과 함께 써야 합니다")

    tmeta = load_targets_meta(cache_dir) if rolling else {}
    parts = []
    for sym in symbols:
        for date in dates:
            ds = load_day(cache_dir, sym, date, seq_len, mmap, norm,
                          normalize_target, winsorize, tmeta)
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
def split_dates(dates, n_val: int | None = None, n_test: int | None = None):
    """날짜 목록을 학습/검증/최종시험으로 시간순 분할.

    날짜 단위로 자르면 한 샘플의 입력·정답 구간이 통째로 하루 안에 들어가므로
    구간 사이 침범이 원천적으로 없다. 최종시험은 **가장 최근** 날짜를 쓴다.
    """
    n_val = spec.N_VAL if n_val is None else n_val
    n_test = spec.N_TEST if n_test is None else n_test
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
