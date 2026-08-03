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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="저장 위치")
    ap.add_argument("--symbols", nargs="*", default=sorted(FILE_IDS))
    ap.add_argument("--dates", nargs="*", default=WEEKDAYS)
    args = ap.parse_args()

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

    print(f"대상 {len(jobs)}개 파일 -> {args.out}/\n")
    done = skipped = failed = 0

    for i, (sym, date, fid) in enumerate(jobs, 1):
        sym_dir = os.path.join(args.out, sym)
        os.makedirs(sym_dir, exist_ok=True)
        path = os.path.join(sym_dir, f"{sym}USDT_{date}.tar")

        if os.path.exists(path) and os.path.getsize(path) > MIN_BYTES:
            print(f"[{i}/{len(jobs)}] 이미 있음  {sym} {date}"
                  f"  ({os.path.getsize(path)/1e6:.0f} MB)")
            skipped += 1
            continue

        print(f"[{i}/{len(jobs)}] 받는 중   {sym} {date} ...", flush=True)
        try:
            gdown.download(id=fid, output=path, quiet=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"    실패: {exc}", file=sys.stderr)
            failed += 1
            continue

        if not os.path.exists(path) or os.path.getsize(path) <= MIN_BYTES:
            print("    실패: 파일이 비었거나 너무 작습니다", file=sys.stderr)
            failed += 1
            continue
        print(f"    완료 {os.path.getsize(path)/1e6:.0f} MB")
        done += 1

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.out) for f in fs
    )
    print(f"\n새로 받음 {done} / 이미 있음 {skipped} / 실패 {failed}")
    print(f"총 용량 {total/1e9:.2f} GB")
    if failed:
        print("실패한 파일은 이 스크립트를 다시 실행하면 재시도합니다.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
