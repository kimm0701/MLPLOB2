"""전처리를 여러 프로세스로 나눠 돌린다 (로컬용).

    python scripts/preprocess_parallel.py --cache data/processed_ev --bucket-events 1

scripts/preprocess.py 는 한 파일씩 순서대로 처리한다. 서버는 CPU 가 2 코어라
그게 맞지만, 코어가 많은 기계에서는 파일 단위로 나누면 코어 수만큼 빨라진다.
날짜별 파일은 서로 완전히 독립이라 나눠도 결과가 달라지지 않는다.

실측: 파일 하나에 약 200 초. 2 코어 순차면 42 개에 2.3 시간, 8 코어에서 6 개씩
병렬이면 약 25 분.

이미 만들어 둔 날짜는 건너뛴다 - 중간에 끊겨도 다시 돌리면 남은 것만 한다.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_data import FILE_IDS, WEEKDAYS      # noqa: E402
from scripts.preprocess import process_day                # noqa: E402


def one(job):
    """자식 프로세스에서 도는 단위 작업."""
    sym, date, tar, out_base, bev = job
    t0 = time.time()
    try:
        st = process_day(tar, out_base, bev)
        return sym, date, st, None, time.time() - t0
    except Exception as exc:                              # noqa: BLE001
        return sym, date, None, f"{type(exc).__name__}: {exc}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/processed_ev")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument("--bucket-events", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="동시에 돌릴 프로세스 수. 기본은 코어 수 - 2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = []
    for sym in args.symbols:
        for date in args.dates:
            tar = os.path.join(args.raw, sym, f"{sym}USDT_{date}.tar")
            if not os.path.exists(tar):
                continue
            out_base = os.path.join(args.cache, sym, date)
            if not args.force and os.path.exists(out_base + "_x.npy"):
                continue
            os.makedirs(os.path.dirname(out_base), exist_ok=True)
            jobs.append((sym, date, tar, out_base, args.bucket_events))

    if not jobs:
        print("할 일이 없습니다 (이미 다 만들었거나 원본이 없습니다).")
        return 0

    mode = (f"이벤트 {args.bucket_events}건/칸" if args.bucket_events
            else "시간 격자 50ms")
    print(f"대상 {len(jobs)}개   버킷 기준: {mode}   동시 {args.workers}개\n")
    print(f"{'':>4} {'종목':<6}{'날짜':<10}{'버킷':>12}{'학습샘플':>12}"
          f"{'이벤트있음':>10}{'소요':>8}")
    print("-" * 66)

    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for fu in as_completed(futs):
            sym, date, st, err, el = fu.result()
            done += 1
            if err:
                fail += 1
                print(f"{done:>4} {sym:<6}{date:<10}  실패: {err}")
            else:
                print(f"{done:>4} {sym:<6}{date:<10}{st['buckets']:>12,}"
                      f"{st['samples']:>12,}{st['filled']*100:>9.1f}%{el:>7.0f}초")
            sys.stdout.flush()

    el = time.time() - t0
    print(f"\n완료 {done - fail} / 실패 {fail}   총 {el/60:.1f}분 "
          f"(순차였다면 약 {sum(1 for _ in jobs) * 200 / 60:.0f}분)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
