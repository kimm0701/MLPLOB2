"""구글 드라이브에서 바이낸스 호가 캡처 원본을 내려받는다.

    pip install gdown
    python scripts/download_data.py                 # 평일 14일 x 3종목
    python scripts/download_data.py --symbols AMD   # 한 종목만
    python scripts/download_data.py --dates 20260730 20260731

이미 받은 파일은 크기를 확인해서 건너뛴다. 중간에 끊겨도 다시 실행하면
받다 만 것부터 이어서 진행한다.

파일 하나는 하루치이고, 내부에 분당 1개씩 .gz 가 들어있다. 압축을 풀 필요는
없다 — 전처리 코드가 tar 를 직접 읽는다.
"""

import argparse
import os
import sys
import time

# 2026-07-14 ~ 2026-07-31 중 평일 14일. 주말은 미국 본장이 닫혀 있어 제외한다.
WEEKDAYS = [
    "20260714", "20260715", "20260716", "20260717",
    "20260720", "20260721", "20260722", "20260723", "20260724",
    "20260727", "20260728", "20260729", "20260730", "20260731",
]

# {심볼: {날짜: 드라이브 파일 ID}}
FILE_IDS = {
    "MRVL": {
        "20260714": "1qYt2nLSVp4azqB8eL7RvOutsl5b8Ecvd",
        "20260715": "1upuLqsPtCV9rvgvoM84w2Cz-49sVYZuq",
        "20260716": "1SfZf-H7fTN6AGsXOtiqpgxOqm62k7-sy",
        "20260717": "1IlxI3qm8UxVDTlqUfHDsYdF-kaId_TlC",
        "20260720": "1WpWYKHNijh-llaUlOiMKGAOSJ5sxzBJP",
        "20260721": "1LO8-PILF4mzpwCfaSxR4HzOzwmSBHOdZ",
        "20260722": "1dtyS3NRhEcXM-LBL_zNkGCMxlYLaJUhd",
        "20260723": "1g__BQ3eQVw1O44BE_oK_BGC6iZbBuKQw",
        "20260724": "1JsAzfH9f5H2d7YCMMouJGVZINsSVgPY0",
        "20260727": "1k_xplH3eJi3BSj8x4U14ofn_M1T9LkaK",
        "20260728": "18Ui9puWVY-oBSO_6OnPMZTzmvK4FXV4S",
        "20260729": "1pz7g5TeXDQG-IWyW4D5riHqK_xovJFCK",
        "20260730": "1-hSt_UNW_87K-KDDaGP3eYwPlOSBzrtk",
        "20260731": "12EmlowwUx5ixotzCC3cqLsipwKHFu-ab",
    },
    "AMD": {
        "20260714": "1dJmvoTX3sNi9xcPlEMmI_E3MD7Kq3Yb0",
        "20260715": "1LRhjTsi5zlCW6AwyDhgY2_toRWzGDzO2",
        "20260716": "1Tw0Tv0k11gAMty3OZatNS16S8yIOtRCt",
        "20260717": "1HhBYjvE9XPEWJiSUJXk21P_9ubsNggS4",
        "20260720": "14MK2x5zX4ZHacEx7drfPSzuzT6gS4GQE",
        "20260721": "1E9z-U00ZbRRQ5BYif1zV5FJmNstc6Ydu",
        "20260722": "1-2mBa4BeVJgMMGPueQwcwUeB24NaO7wV",
        "20260723": "1HZSTuIvtlbea5uQ-X71MWa8JipB8BeML",
        "20260724": "1_OVR_mGcokNNn6cy53JLJBR3TDEQ4xoi",
        "20260727": "1Hxt3drB7kgk9ulz4cTkzY1zwSD_jNAdn",
        "20260728": "1z6Z7yS95gz8boTz9WPX50u6ZUgxonKkq",
        "20260729": "1C6A7mSh9OLrLhSYbdGrOt16T12SSqRU2",
        "20260730": "158RMILXb20hRxcya2eM-HSCpfxkpIj9A",
        "20260731": "1Df7UCf39_uxQmenGZbCltkvTGsnQj4e0",
    },
    "META": {
        "20260714": "1lcYEWDx8Y-Gz-cg9ZxFUJRCVAkWZqGlL",
        "20260715": "1BjhNs4Kxg5kAn8bDatxS1OawsPF7mjJC",
        "20260716": "1wJKEaInH7WZV8FMlIHIE86uyrLqB1bhW",
        "20260717": "1Xs13PLv6mQgrA-plvDV80zdIQ1DHrBRm",
        "20260720": "1uBOFE9NNo1TEcTZq3rm77lPwBKk2W4Mx",
        "20260721": "1S78sIalTCV_6vSmBPn2XKhfSuBEbFToc",
        "20260722": "1bEAM0waoAnOHxWvOeAXnd_SWV25fSgJ5",
        "20260723": "1-LxHHkXGOsF9C0yChNt_IZPRIJEehw6e",
        "20260724": "1vQ9T_uzDecjxquXL2qQPBibEJxxH1Rrn",
        "20260727": "1UVxUHz7dYJjMI-2Tm1CZRdntA_dJDsA_",
        "20260728": "1vykSUhNUvHzbQ7ZadxajjulTWkd_ETXd",
        "20260729": "1et171v2f4ZgEQl_aQYpVrUWbaNAhTL9b",
        "20260730": "1r42hwZssDmrVj6nhdecwuz7M-g-Ean8k",
        "20260731": "1yR1kevzNWi3AGk1LWFtCpHf_W3iHCL18",
    },
}

MIN_BYTES = 1_000_000     # 이보다 작으면 받다 만 것으로 보고 다시 받는다


def is_complete_tar(path: str) -> bool:
    """받다 만 파일을 완성본으로 착각하지 않도록 검사한다.

    크기만 보면 중간에 끊긴 파일도 통과한다. tar 를 열어 첫 멤버만 읽는 것도
    부족하다 — 앞부분이 멀쩡하면 잘린 파일도 통과해 버린다.
    tar 는 512바이트 0 블록 두 개로 끝나므로, 그 종료 표식까지 확인한다.
    """
    import tarfile

    if not os.path.exists(path) or os.path.getsize(path) <= MIN_BYTES:
        return False

    # 1) 끝에 종료 표식(0 으로 채운 1024바이트)이 있는가 -> 잘림 탐지
    try:
        with open(path, "rb") as fh:
            fh.seek(-1024, os.SEEK_END)
            if fh.read(1024) != b"\0" * 1024:
                return False
    except OSError:
        return False

    # 2) 헤더 체인이 끝까지 이어지는가 -> 내용 손상 탐지
    try:
        with tarfile.open(path) as tf:
            return len(tf.getmembers()) > 0
    except Exception:                                   # noqa: BLE001
        return False


def deep_verify_tar(path: str) -> tuple[bool, str]:
    """압축을 실제로 풀어보는 정밀 검사.

    is_complete_tar 는 헤더 체인과 종료 표식만 본다. 그것만으로는 파일 중간이
    깨진 경우를 못 잡는다 (예: 같은 파일에 두 프로세스가 동시에 쓴 경우).
    여기서는 안에 든 .gz 를 전부 실제로 풀어보므로 확실하다. 대신 느리다.
    """
    import gzip
    import tarfile

    try:
        with tarfile.open(path) as tf:
            members = tf.getmembers()
            if not members:
                return False, "빈 아카이브"
            for mem in members:
                if not mem.isfile():
                    continue
                fh = tf.extractfile(mem)
                if fh is None:
                    return False, f"{mem.name}: 읽을 수 없음"
                with gzip.GzipFile(fileobj=fh) as gz:
                    while gz.read(1 << 20):
                        pass
        return True, f"{len(members)}개 조각 정상"
    except Exception as exc:                            # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _pid_alive(pid: int) -> bool:
    """그 PID 가 아직 살아 있는가."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)            # 신호 0 은 존재 확인만 한다
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                # 남의 프로세스지만 살아 있다
    except OSError:
        return False
    return True


def acquire_lock(out_dir: str):
    """같은 스크립트를 두 번 띄워 서로의 파일을 덮어쓰는 사고를 막는다.

    강제 종료되면 잠금 파일이 남는다. 그 상태로 다음 실행이 영영 막히면
    자리를 비운 사이 아무것도 진행되지 않으므로, 기록된 PID 가 죽어 있으면
    남은 잠금으로 보고 넘겨받는다.
    """
    os.makedirs(out_dir, exist_ok=True)
    lock_path = os.path.join(out_dir, ".download.lock")

    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock_path) as fh:
                    pid = int(fh.read().strip() or 0)
            except (OSError, ValueError):
                pid = 0

            if _pid_alive(pid):
                print(f"이미 실행 중입니다 (PID {pid}). 두 개가 같은 파일을 "
                      f"건드리지 않도록 이번 실행은 멈춥니다.", file=sys.stderr)
                return None

            print(f"죽은 프로세스(PID {pid}) 의 잠금이 남아 있어 정리합니다.",
                  file=sys.stderr)
            try:
                os.remove(lock_path)
            except OSError:
                pass
            continue                       # 지웠으니 한 번 더 시도
        else:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock_path

    print(f"잠금을 얻지 못했습니다: {lock_path}", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="저장 위치")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="받지 않고, 이미 있는 파일의 압축을 실제로 풀어 손상 여부만 검사한다",
    )
    ap.add_argument(
        "--delete-corrupt",
        action="store_true",
        help="--verify 에서 깨진 파일을 지운다. 이후 다시 실행하면 그것만 새로 받는다",
    )
    ap.add_argument("--retries", type=int, default=4,
                    help="파일당 재시도 횟수 (기본 4)")
    ap.add_argument("--backoff", type=float, default=30.0,
                    help="첫 재시도 전 대기 초. 실패할수록 2배씩 늘어난다 (기본 30)")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="파일 사이 대기 초. 드라이브 속도 제한 회피용 (기본 3)")
    args = ap.parse_args()

    if not args.verify:
        try:
            import gdown
        except ImportError:
            print("gdown 이 없습니다.  pip install gdown", file=sys.stderr)
            return 1

    jobs = []
    for sym in args.symbols:
        if sym not in FILE_IDS:
            print(f"모르는 심볼: {sym}. 가능한 값: {sorted(FILE_IDS)}", file=sys.stderr)
            return 1
        for date in args.dates:
            fid = FILE_IDS[sym].get(date)
            if fid is None:
                print(f"  건너뜀 {sym} {date} (해당 날짜 파일 없음)")
                continue
            jobs.append((sym, date, fid))

    if args.verify:
        print(f"정밀 검사 {len(jobs)}개 파일 (압축을 실제로 풀어봅니다)\n")
        bad = missing = 0
        for i, (sym, date, _) in enumerate(jobs, 1):
            path = os.path.join(args.out, sym, f"{sym}USDT_{date}.tar")
            if not os.path.exists(path):
                print(f"[{i}/{len(jobs)}] 없음   {sym} {date}")
                missing += 1
                continue
            good, detail = deep_verify_tar(path)
            print(f"[{i}/{len(jobs)}] {'정상' if good else '손상'} {sym} {date}  {detail}",
                  flush=True)
            if not good:
                bad += 1
                if args.delete_corrupt:
                    os.remove(path)
                    print("        지웠습니다. 다시 실행하면 새로 받습니다.")
        print(f"\n손상 {bad} / 없음 {missing} / 정상 {len(jobs) - bad - missing}")
        if bad and not args.delete_corrupt:
            print("--delete-corrupt 를 붙여 다시 실행하면 깨진 파일을 지웁니다.")
        return 1 if (bad or missing) else 0

    lock_path = acquire_lock(args.out)
    if lock_path is None:
        return 1

    try:
        print(f"대상 {len(jobs)}개 파일 -> {args.out}/\n")
        done = skipped = failed = 0

        for i, (sym, date, fid) in enumerate(jobs, 1):
            sym_dir = os.path.join(args.out, sym)
            os.makedirs(sym_dir, exist_ok=True)
            path = os.path.join(sym_dir, f"{sym}USDT_{date}.tar")

            if is_complete_tar(path):
                print(f"[{i}/{len(jobs)}] 이미 있음  {sym} {date}"
                      f"  ({os.path.getsize(path)/1e6:.0f} MB)")
                skipped += 1
                continue

            if os.path.exists(path):
                print(f"    받다 만 파일 발견, 지우고 다시 받습니다: {path}")
                os.remove(path)

            print(f"[{i}/{len(jobs)}] 받는 중   {sym} {date} ...", flush=True)

            # 구글 드라이브는 짧은 시간에 여러 파일을 받으면 일시적으로 막는다.
            # 권한 문제가 아니라 속도 제한이므로, 간격을 늘려가며 다시 시도한다.
            ok = False
            for attempt in range(1, args.retries + 1):
                try:
                    gdown.download(id=fid, output=path, quiet=True)
                except Exception as exc:                   # noqa: BLE001
                    print(f"    시도 {attempt}/{args.retries} 실패: {exc}",
                          file=sys.stderr, flush=True)
                else:
                    if is_complete_tar(path):
                        ok = True
                        break
                    print(f"    시도 {attempt}/{args.retries} 실패: "
                          "받다 만 파일", file=sys.stderr, flush=True)

                if os.path.exists(path):
                    os.remove(path)
                if attempt < args.retries:
                    wait = args.backoff * (2 ** (attempt - 1))
                    print(f"    {wait:.0f}초 쉬었다가 다시 시도합니다", flush=True)
                    time.sleep(wait)

            if not ok:
                print(f"    포기: {sym} {date}", file=sys.stderr, flush=True)
                failed += 1
                continue

            print(f"    완료 {os.path.getsize(path)/1e6:.0f} MB")
            done += 1
            time.sleep(args.delay)      # 다음 파일 전 잠깐 쉬어 제한을 피한다
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.out)
        for f in fs
        if f.endswith(".tar")
    )
    print(f"\n새로 받음 {done} / 이미 있음 {skipped} / 실패 {failed}")
    print(f"총 용량 {total/1e9:.2f} GB")
    if failed:
        print("실패한 파일은 이 스크립트를 다시 실행하면 재시도합니다.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
