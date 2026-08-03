"""원본 캡처(.tar) -> 학습용 배열(.npy).

    python scripts/preprocess.py                       # 전 종목 x 평일 14일
    python scripts/preprocess.py --symbols MRVL
    python scripts/preprocess.py --dates 20260730

(종목, 날짜) 하나당 파일 3개를 만든다.

    <cache>/<SYM>/<DATE>_x.npy     [K, 22]  float32   입력 특징
    <cache>/<SYM>/<DATE>_y.npy     [K, 10]  float32   미래 수익률
    <cache>/<SYM>/<DATE>_px.npy    [K, 3]   float64   mid, 최우선매수, 최우선매도
    <cache>/<SYM>/<DATE>_idx.npy   [N]      int64     학습에 쓸 수 있는 시점

_px 는 학습에 쓰지 않는다. 백테스트에서 "이 예측대로 매매했으면 얼마를 벌었나"
를 계산하려면 실제 호가가 필요한데, 특징(_x)은 정규화된 주문흐름이라 가격
정보가 남아 있지 않다. 스프레드 비용도 최우선 매수/매도 차이로 계산한다.
float64 인 이유는 float32 가 소수 7자리까지만 정확해서, 435.58 같은 가격의
1틱(0.01) 차이를 다루기에 여유가 부족하기 때문이다.

x/y 는 버킷 전체를 담고, idx 가 그중 쓸 수 있는 시점만 가리킨다. 이렇게 두면
슬라이딩 윈도우를 만들 때 배열을 복사하지 않아도 되고 (100배 중복 저장 회피),
warm-up 이나 미래 부족 구간을 나중에 다시 계산할 필요도 없다.

이미 만들어진 날짜는 건너뛴다. --force 로 다시 만들 수 있다.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ofi_spec as spec                                        # noqa: E402
from preprocessing.binance_capture import iter_anchored_events  # noqa: E402
from preprocessing.ofi_features import build_features           # noqa: E402
from preprocessing.ofi_labels import build_targets, valid_sample_indices  # noqa: E402
from scripts.download_data import WEEKDAYS, FILE_IDS            # noqa: E402


def process_day(tar_path: str, out_base: str) -> dict:
    t0 = time.time()
    res = build_features(iter_anchored_events(tar_path))
    targets, target_valid = build_targets(res.mid, res.mid_valid)
    idx = valid_sample_indices(res.feature_valid, target_valid, spec.SEQ_LEN)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    np.save(out_base + "_x.npy", res.features.astype(np.float32))
    np.save(out_base + "_y.npy", targets.astype(np.float32))
    np.save(out_base + "_px.npy",
            np.stack([res.mid, res.best_bid, res.best_ask], axis=1).astype(np.float64))
    np.save(out_base + "_idx.npy", idx.astype(np.int64))

    return dict(
        buckets=res.n_buckets,
        samples=int(idx.size),
        filled=float((res.n_events > 0).mean()),
        multi=float((res.n_events > 1).mean()),
        repairs=int(res.n_repairs.sum()),
        hours=res.n_buckets * spec.BUCKET_MS / 3.6e6,
        secs=time.time() - t0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="원본 tar 위치")
    ap.add_argument("--cache", default="data/processed", help="결과 저장 위치")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--force", action="store_true", help="이미 만든 날짜도 다시 만든다")
    args = ap.parse_args()

    jobs = []
    for sym in args.symbols:
        for date in args.dates:
            tar = os.path.join(args.raw, sym, f"{sym}USDT_{date}.tar")
            if not os.path.exists(tar):
                print(f"  원본 없음, 건너뜀: {tar}")
                continue
            jobs.append((sym, date, tar))

    if not jobs:
        print("처리할 파일이 없습니다.", file=sys.stderr)
        return 1

    print(f"대상 {len(jobs)}개 (종목 {len(args.symbols)} x 날짜 {len(args.dates)})\n")
    print(f"{'':>5}{'종목':<6}{'날짜':<10}{'버킷':>10}{'학습샘플':>11}"
          f"{'이벤트있음':>10}{'2건이상':>9}{'소요':>8}")
    print("-" * 72)

    done = skipped = 0
    total_samples = 0
    for i, (sym, date, tar) in enumerate(jobs, 1):
        out_base = os.path.join(args.cache, sym, date)
        # _px 까지 있어야 완료로 본다. 이 파일이 없던 시절에 처리한 날짜는
        # 다시 돌려서 가격 배열을 채운다.
        complete = all(os.path.exists(out_base + s)
                       for s in ("_x.npy", "_y.npy", "_px.npy", "_idx.npy"))
        if not args.force and complete:
            n = int(np.load(out_base + "_idx.npy").size)
            total_samples += n
            skipped += 1
            print(f"{i:>4} {sym:<6}{date:<10}{'':>10}{n:>11,}{'  (이미 있음)':>20}")
            continue

        try:
            st = process_day(tar, out_base)
        except Exception as exc:                                # noqa: BLE001
            print(f"{i:>4} {sym:<6}{date:<10}  실패: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            continue

        total_samples += st["samples"]
        done += 1
        print(f"{i:>4} {sym:<6}{date:<10}{st['buckets']:>10,}{st['samples']:>11,}"
              f"{100*st['filled']:>9.1f}%{100*st['multi']:>8.1f}%{st['secs']:>7.0f}초",
              flush=True)

    print(f"\n새로 처리 {done} / 이미 있음 {skipped}")
    print(f"학습에 쓸 수 있는 샘플 총 {total_samples:,}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
