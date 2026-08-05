"""회귀 성적 지표.

분류(상승·보합·하락)에서 회귀로 바뀌면서 정확도·F1 대신 쓸 것들이다.
Lightning 모듈과 분리해 둔 이유는 이 계산 자체를 단위 테스트로 검증하기
위해서다 — 지표가 틀리면 모델이 좋은지 나쁜지 판단 자체가 어긋난다.

모든 함수는 [N, H] 예측과 [N, H] 정답을 받아 horizon 별로 [H] 를 돌려준다.
"""

from __future__ import annotations

import numpy as np


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def mse(pred, target) -> np.ndarray:
    p, t = _as2d(pred), _as2d(target)
    return np.mean((p - t) ** 2, axis=0)


def mae(pred, target) -> np.ndarray:
    p, t = _as2d(pred), _as2d(target)
    return np.mean(np.abs(p - t), axis=0)


def directional_accuracy(pred, target) -> np.ndarray:
    """오를지 내릴지를 맞춘 비율.

    정답이 정확히 0 인 표본은 제외한다. 미드가격은 틱 단위라 짧은 horizon 에서
    변화 없음이 흔한데, 그걸 '틀림' 으로 세면 0.5 를 향해 눌리고 '맞음' 으로
    세면 부풀려진다. 방향을 물었으면 방향이 있는 표본만 세는 게 맞다.
    0.5 는 동전 던지기와 같다는 뜻이다.
    """
    p, t = _as2d(pred), _as2d(target)
    out = np.full(p.shape[1], np.nan)
    for j in range(p.shape[1]):
        m = t[:, j] != 0
        if m.sum() == 0:
            continue
        out[j] = np.mean(np.sign(p[m, j]) == np.sign(t[m, j]))
    return out


def information_coefficient(pred, target, rank: bool = True) -> np.ndarray:
    """예측과 실제의 상관계수. 이 분야에서 IC 라고 부른다.

    rank=True 면 순위상관(스피어만)이다. 수익률은 꼬리가 두꺼워서 값 그대로
    상관을 내면 극단값 몇 개가 결과를 좌우한다. 순위로 바꾸면 그 영향이 사라진다.

    0 이면 무관, 0.02~0.05 면 초단타에서는 쓸만한 수준으로 본다.
    """
    p, t = _as2d(pred), _as2d(target)
    out = np.full(p.shape[1], np.nan)
    for j in range(p.shape[1]):
        a, b = p[:, j], t[:, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            continue
        a, b = a[m], b[m]
        if rank:
            a, b = _rankdata(a), _rankdata(b)
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            continue
        out[j] = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    """동점은 평균 순위를 준다 (scipy.stats.rankdata 와 같은 규칙)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def r2(pred, target) -> np.ndarray:
    """1 - (예측 오차) / (평균으로 찍었을 때 오차).

    0 이면 '그냥 평균값을 답하는 것' 과 같다. 음수면 그보다도 못하다는 뜻이라,
    금융 데이터에서는 흔히 음수가 나온다.
    """
    p, t = _as2d(pred), _as2d(target)
    ss_res = np.sum((t - p) ** 2, axis=0)
    ss_tot = np.sum((t - t.mean(axis=0)) ** 2, axis=0)
    return np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-30), np.nan)


def fit_scale(pred, target, with_intercept: bool = False):
    """horizon 마다 최소제곱 배율(과 절편). pred*alpha + beta 가 target 에 가장 가깝게."""
    p, t = _as2d(pred), _as2d(target)
    H = p.shape[1]
    alpha, beta = np.ones(H), np.zeros(H)
    for j in range(H):
        pj, tj = p[:, j], t[:, j]
        m = np.isfinite(pj) & np.isfinite(tj)
        pj, tj = pj[m], tj[m]
        if pj.size < 10 or pj.std() == 0:
            continue
        if with_intercept:
            A = np.vstack([pj, np.ones_like(pj)]).T
            sol, *_ = np.linalg.lstsq(A, tj, rcond=None)
            alpha[j], beta[j] = sol
        else:
            denom = float(np.dot(pj, pj))
            alpha[j] = float(np.dot(pj, tj) / denom) if denom > 0 else 1.0
    return alpha, beta


def r2_calibrated(pred, target) -> np.ndarray:
    """배율을 맞춘 뒤의 R2 — 이 모델이 실제로 낼 수 있는 R2.

    보정 전 R2 는 방향과 크기를 함께 벌한다. 덜 학습된 모델은 방향을 맞혀도
    출력이 너무 크거나 작아서 R2 가 음수로 떨어진다 (실측: 예측이 이론값의
    8~48%). 그 크기는 학습 후 배율 하나로 고치면 되는 문제라, 설정끼리
    비교할 때 크기 차이로 순위가 뒤집히면 안 된다.

    배율을 같은 데이터에서 구하므로 값이 약간 낙관적이다. 다만 horizon 당
    자유도가 1 개뿐이고 표본이 수백만이라 그 영향은 무시할 수준이다. 최종
    성적은 scripts/calibrate.py 로 검증→시험 분리해서 다시 낸다.
    """
    p, t = _as2d(pred), _as2d(target)
    alpha, beta = fit_scale(p, t)
    return r2(p * alpha + beta, t)


def summarise(pred, target, horizons=None) -> dict:
    """지표를 한 번에 계산해 dict 로 돌려준다."""
    p, t = _as2d(pred), _as2d(target)
    horizons = list(horizons) if horizons is not None else list(range(1, p.shape[1] + 1))
    return dict(
        horizons=horizons,
        mse=mse(p, t),
        mae=mae(p, t),
        dir_acc=directional_accuracy(p, t),
        ic=information_coefficient(p, t),
        r2=r2(p, t),
        r2_cal=r2_calibrated(p, t),
        n=int(p.shape[0]),
    )


def format_table(s: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'horizon':>8}{'MSE(bp^2)':>12}{'MAE(bp)':>10}"
                 f"{'방향정확도':>11}{'IC':>9}{'R2':>9}{'R2(보정)':>11}")
    lines.append("-" * 71)
    for j, h in enumerate(s["horizons"]):
        cal = s.get("r2_cal", s["r2"])[j]
        lines.append(
            f"{h:>6}초{s['mse'][j]*1e8:>12.2f}{s['mae'][j]*1e4:>10.3f}"
            f"{s['dir_acc'][j]*100:>10.2f}%{s['ic'][j]:>9.4f}{s['r2'][j]:>9.4f}"
            f"{cal:>11.5f}"
        )
    lines.append(f"표본 {s['n']:,}개")
    return "\n".join(lines)
