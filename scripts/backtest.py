"""예측을 실제 매매로 환산해 스프레드 비용까지 빼고 수익을 본다.

    python scripts/backtest.py --ckpt data/checkpoints/.../epoch3_val6.12345.ckpt

예측이 맞는 것과 돈이 되는 것은 다르다. 미드프라이스 기준 수익률은 사고파는
가격 차이(스프레드)를 건너뛰므로 실제보다 좋아 보인다. 여기서는 전처리 때
저장해 둔 _px.npy 의 실제 최우선 호가로 체결을 흉내낸다.

    매수 진입 -> 매도호가에 산다 (비싸게)
    매수 청산 -> 매수호가에 판다 (싸게)

그래서 진입 즉시 스프레드만큼 손해로 시작한다. 예측이 그걸 넘어야 수익이다.
MRVL 실측으로 10초 변동성 5.06bp 대 스프레드 1.09bp 였으니 여지는 있지만,
예측이 그만큼 맞아야 한다.

겹치는 거래
-----------
0.05초마다 신호가 나오고 보유는 1~10초라 거래가 겹친다. 기본은 신호 하나하나를
독립 표본으로 보는 방식(건당 평균 수익)이고, --non-overlap 을 주면 청산 전에는
새로 진입하지 않는 방식으로 잰다. 앞은 신호의 질을, 뒤는 실제로 굴렸을 때를
본다.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                        # noqa: E402
from models.regression_engine import (                          # noqa: E402
    TARGET_SCALE,
    load_model_from_checkpoint,
)
from preprocessing.ofi_dataset import load_normalizer, split_dates  # noqa: E402
from scripts.download_data import FILE_IDS, WEEKDAYS           # noqa: E402


@torch.no_grad()
def predict_day(model, cache, symbol, date, norm, device, batch_size=4096):
    """하루치 예측. (버킷인덱스, 예측[N,10], 가격[K,3]) 를 돌려준다."""
    base = os.path.join(cache, symbol, date)
    if not os.path.exists(base + "_idx.npy"):
        return None
    x = np.load(base + "_x.npy", mmap_mode="r")
    px = np.load(base + "_px.npy", mmap_mode="r")
    idx = np.load(base + "_idx.npy")
    if idx.size == 0:
        return None

    center = scale = None
    if norm:
        st = norm[symbol]
        center = np.asarray(st["center"], np.float32)
        scale = np.asarray(st["scale"], np.float32)

    preds = np.empty((idx.size, spec.OUTPUT_DIM), np.float32)
    seq = spec.SEQ_LEN
    for s in range(0, idx.size, batch_size):
        ks = idx[s:s + batch_size]
        win = np.stack([np.asarray(x[k - seq + 1:k + 1], np.float32) for k in ks])
        if center is not None:
            win = (win - center) / scale
        out = model(torch.from_numpy(win).to(device))
        # 모델은 bp 로 예측한다. 가격 계산은 소수 수익률로 한다.
        preds[s:s + len(ks)] = out.float().cpu().numpy() / TARGET_SCALE
    return idx, preds, np.asarray(px)


def trade_stats(idx, preds, px, horizon_j, threshold, fee_bp=0.0,
                non_overlap=False):
    """한 horizon 에 대해 거래를 흉내내고 결과를 돌려준다."""
    offset = spec.TARGET_HORIZON_BUCKETS[horizon_j]
    mid, bid, ask = px[:, 0], px[:, 1], px[:, 2]

    sig = preds[:, horizon_j]
    take = np.abs(sig) > threshold
    exit_k = idx + offset
    ok = take & (exit_k < len(mid)) & np.isfinite(bid[idx]) & np.isfinite(ask[idx])
    ok &= np.isfinite(bid[np.clip(exit_k, 0, len(mid) - 1)])
    ok &= np.isfinite(ask[np.clip(exit_k, 0, len(mid) - 1)])
    if not ok.any():
        return None

    k_in = idx[ok]
    k_out = exit_k[ok]
    side = np.sign(sig[ok])

    if non_overlap:
        keep = np.zeros(len(k_in), bool)
        free_at = -1
        for i, (a, b) in enumerate(zip(k_in, k_out)):
            if a >= free_at:
                keep[i] = True
                free_at = b
        k_in, k_out, side = k_in[keep], k_out[keep], side[keep]
        if not len(k_in):
            return None

    # 실제 체결가: 살 때는 매도호가, 팔 때는 매수호가를 친다
    entry = np.where(side > 0, ask[k_in], bid[k_in])
    exit_ = np.where(side > 0, bid[k_out], ask[k_out])
    net = side * (exit_ - entry) / entry - 2 * fee_bp * 1e-4

    gross = side * (mid[k_out] - mid[k_in]) / mid[k_in]      # 스프레드 무시
    spread = (ask[k_in] - bid[k_in]) / mid[k_in]

    return dict(
        n=len(k_in),
        gross_bp=float(np.mean(gross) * 1e4),
        net_bp=float(np.mean(net) * 1e4),
        spread_bp=float(np.mean(spread) * 1e4),
        hit=float(np.mean(net > 0)),
        total_bp=float(np.sum(net) * 1e4),
        sharpe=float(np.mean(net) / np.std(net) * np.sqrt(len(k_in)))
        if np.std(net) > 0 else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.0, 0.5e-4, 1e-4, 2e-4],
                    help="이 값보다 예측 절대값이 커야 진입한다 (소수 단위)")
    ap.add_argument("--fee-bp", type=float, default=0.0,
                    help="편도 수수료 (bp). 왕복이므로 2배가 빠진다")
    ap.add_argument("--non-overlap", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    _, _, test_dates = split_dates(args.dates, args.n_val, args.n_test)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"체크포인트 {args.ckpt}")
    print(f"시험 날짜 {test_dates}   장치 {device}")
    print("겹침 처리: " + ("청산 후 재진입" if args.non_overlap else "신호 건당 독립"))
    print(f"수수료 편도 {args.fee_bp}bp\n")

    model, hp = load_model_from_checkpoint(args.ckpt, map_location=device)
    model = model.to(device).eval()
    print(f"모델 구조 {hp.get('model_config')}\n")
    norm = {} if args.no_normalize else load_normalizer(args.cache)

    per_symbol = {}
    for sym in args.symbols:
        chunks = []
        for date in test_dates:
            got = predict_day(model, args.cache, sym, date, norm, device)
            if got is not None:
                chunks.append(got)
        if chunks:
            per_symbol[sym] = chunks

    if not per_symbol:
        print("시험할 데이터가 없습니다.", file=sys.stderr)
        return 1

    for thr in args.thresholds:
        print("=" * 78)
        print(f"진입 기준 |예측| > {thr*1e4:.2f}bp")
        print("=" * 78)
        print(f"{'종목':<7}{'horizon':>8}{'거래수':>10}{'스프레드':>9}"
              f"{'총수익(무시)':>13}{'실수익':>10}{'승률':>8}{'누적bp':>11}")
        print("-" * 78)
        for sym, chunks in per_symbol.items():
            for j, h in enumerate(spec.TARGET_HORIZONS_SEC):
                agg = [trade_stats(i, p, x, j, thr, args.fee_bp, args.non_overlap)
                       for i, p, x in chunks]
                agg = [a for a in agg if a]
                if not agg:
                    continue
                n = sum(a["n"] for a in agg)
                w = np.array([a["n"] for a in agg], float) / n
                row = {k: float(np.sum([a[k] for a in agg] * w))
                       for k in ("gross_bp", "net_bp", "spread_bp", "hit")}
                total = sum(a["total_bp"] for a in agg)
                mark = " <-" if row["net_bp"] > 0 else ""
                print(f"{sym:<7}{h:>6}초{n:>10,}{row['spread_bp']:>9.2f}"
                      f"{row['gross_bp']:>13.3f}{row['net_bp']:>10.3f}"
                      f"{row['hit']*100:>7.1f}%{total:>11.0f}{mark}")
        print()

    print("총수익(무시) = 미드 기준, 스프레드를 안 뺀 값")
    print("실수익      = 실제 호가로 체결했을 때. 이게 양수여야 의미가 있다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
