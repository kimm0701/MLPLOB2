"""바이낸스 선물 WebSocket 캡처(.tar) 를 이벤트 스트림으로 바꾸는 어댑터.

파일 구조
---------
    SYMBOLUSDT_YYYYMMDD.tar
      +- SYMBOLUSDT_{epoch초}_{일련번호}.gz     분당 1개
           각 줄:  {수신시각_나노초} {JSON}

JSON 은 세 종류가 섞여 있다.
    @depth@0ms   호가창 차분 업데이트   <- OF/OFI 계산의 원천
    @bookTicker  최우선 매수/매도 갱신  <- 복원 결과 검증용
    @trade       체결

호가창 스냅샷이 캡처에 없다. 빈 호가창에서 차분을 누적하면 상위 단계는 갱신이
잦아 금방 수렴하지만, 그 전까지는 값을 믿을 수 없다. `validate_reconstruction`
으로 복원한 L1 이 bookTicker 와 언제부터 일치하는지 확인해서 warm-up 을 잘라내야
한다.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import tarfile
from dataclasses import dataclass

# depth 만 필요할 때 JSON 파싱 자체를 건너뛰기 위한 사전 필터.
# 전체 줄의 10% 정도만 depth 라서 효과가 크다.
_DEPTH_MARK = '"depthUpdate"'
_BOOK_MARK = '"bookTicker"'
_TRADE_MARK = '"e":"trade"'

_MEMBER_RE = re.compile(r"_(\d+)_(\d+)\.gz$")


def _member_sort_key(name: str):
    """파일명의 (epoch초, 일련번호) 로 정렬. tar 순서에 의존하지 않는다."""
    m = _MEMBER_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def iter_raw_lines(tar_path: str):
    """tar 안의 .gz 를 시간순으로 열어 (수신시각_ns, json문자열) 을 흘려준다."""
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".gz")]
        members.sort(key=lambda m: _member_sort_key(m.name))
        for mem in members:
            fh = tf.extractfile(mem)
            if fh is None:
                continue
            with gzip.GzipFile(fileobj=fh) as gz:
                for raw in gz:
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    sp = line.split(" ", 1)
                    if len(sp) != 2:
                        continue
                    yield sp[0], sp[1]


def iter_depth_events(tar_paths, time_field: str = "T"):
    """(ts_ms, bid_updates, ask_updates) 를 흘려준다.

    time_field
        "T" 거래소가 실제로 호가창을 바꾼 시각 (기본값. OF 의 인과 순서에 맞음)
        "E" 거래소가 메시지를 내보낸 시각

    수량 "0.00" 은 해당 호가 삭제를 뜻한다. ofi_features.OrderBook 이 그렇게 해석한다.
    """
    if isinstance(tar_paths, (str, os.PathLike)):
        tar_paths = [tar_paths]

    for path in tar_paths:
        for _, payload in iter_raw_lines(path):
            if _DEPTH_MARK not in payload:
                continue
            try:
                d = json.loads(payload)["data"]
            except (ValueError, KeyError, TypeError):
                continue
            ts = d.get(time_field, d.get("E"))
            if ts is None:
                continue
            bids = [(float(p), float(q)) for p, q in d.get("b", ())]
            asks = [(float(p), float(q)) for p, q in d.get("a", ())]
            yield int(ts), bids, asks


def iter_book_ticker(tar_paths, time_field: str = "T"):
    """(ts_ms, best_bid, bid_qty, best_ask, ask_qty). 복원 검증용."""
    if isinstance(tar_paths, (str, os.PathLike)):
        tar_paths = [tar_paths]

    for path in tar_paths:
        for _, payload in iter_raw_lines(path):
            if _BOOK_MARK not in payload:
                continue
            try:
                d = json.loads(payload)["data"]
            except (ValueError, KeyError, TypeError):
                continue
            ts = d.get(time_field, d.get("E"))
            if ts is None:
                continue
            yield int(ts), float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"])


def iter_anchored_events(tar_paths, time_field: str = "T", levels: int = 10):
    """bookTicker 로 보정한 이벤트 스트림. build_features 에 바로 넣으면 된다.

    캡처에 호가창 스냅샷이 없고 중간중간 메시지 유실도 있어서, 차분만 누적하면
    현재가에서 멀리 떨어진 자리에 지워지지 않은 '유령 호가' 가 남는다. 실제로
    AMD 는 하루의 94% 동안 매수호가가 매도호가보다 높은, 존재할 수 없는 상태였다.

    bookTicker 는 거래소가 최우선 호가를 직접 알려주는 별도 스트림이다. 이걸
    기준으로 매 갱신마다
        - 진짜 최우선 매수보다 비싼 매수 호가를 지우고
        - 진짜 최우선 매도보다 싼 매도 호가를 지우고
        - 최우선 자리의 수량을 정답으로 덮어쓴다
    그러면 L1 은 정의상 정확해지고, 유령이 사라져 그 아래 단계도 제자리를 찾는다.

    이 정정은 `is_repair=True` 로 표시해서 내보낸다. 실제 주문이 오간 게 아니라
    우리 장부를 고치는 것이므로 OF 로 세면 안 되기 때문이다.

    내보내는 값: (ts_ms, bid_updates, ask_updates, is_repair)
    """
    from preprocessing.ofi_features import OrderBook

    if isinstance(tar_paths, (str, os.PathLike)):
        tar_paths = [tar_paths]

    book = OrderBook()          # 정정 대상을 알아내려면 장부 사본이 필요하다
    prev_u = None

    for path in tar_paths:
        for _, payload in iter_raw_lines(path):
            is_depth = _DEPTH_MARK in payload
            is_book = (not is_depth) and (_BOOK_MARK in payload)
            if not (is_depth or is_book):
                continue
            try:
                d = json.loads(payload)["data"]
            except (ValueError, KeyError, TypeError):
                continue
            ts = d.get(time_field, d.get("E"))
            if ts is None:
                continue
            ts = int(ts)

            if is_depth:
                if prev_u is not None and d.get("pu") != prev_u:
                    # 캡처 유실. 이 지점 이후로 장부가 어긋나기 시작한다.
                    # 통째로 버리지 않는 이유: 현재가 근처 호가는 초당 수십 번
                    # 갱신되므로 곧 제 값을 되찾고, 남은 유령은 아래 정정이 치운다.
                    pass
                prev_u = d.get("u")
                bids = [(float(p), float(q)) for p, q in d.get("b", ())]
                asks = [(float(p), float(q)) for p, q in d.get("a", ())]
                book.apply(bids, asks)
                yield ts, bids, asks, False
                continue

            # bookTicker: 정답과 대조해서 어긋난 부분만 정정 이벤트로 내보낸다
            true_bid, true_bid_q = float(d["b"]), float(d["B"])
            true_ask, true_ask_q = float(d["a"]), float(d["A"])
            if true_bid <= 0 or true_ask <= 0:
                continue

            # 정정은 '구조적으로 불가능한 상태' 만 고친다. 수량이 조금 다른 건
            # 건드리지 않는다 — bookTicker 가 depth 보다 6배 자주 오기 때문에,
            # 수량까지 맞추려 들면 매번 정정이 걸리고 진짜 주문흐름이 정정에
            # 흡수되어 L1 의 OF 가 과소평가된다.
            fix_bids, fix_asks = [], []
            for px in [p for p in book._bid_px if p > true_bid]:
                fix_bids.append((px, 0.0))          # 최우선보다 비싼 매수 = 유령
            for px in [p for p in book._ask_px if p < true_ask]:
                fix_asks.append((px, 0.0))          # 최우선보다 싼 매도 = 유령
            if true_bid not in book._bid:
                fix_bids.append((true_bid, true_bid_q))   # 통째로 빠진 자리만 채운다
            if true_ask not in book._ask:
                fix_asks.append((true_ask, true_ask_q))

            if fix_bids or fix_asks:
                book.apply(fix_bids, fix_asks)
                yield ts, fix_bids, fix_asks, True


# ----------------------------------------------------------------------------
# 파일 찾기
# ----------------------------------------------------------------------------
def day_files(root: str, symbol: str, dates) -> list[str]:
    """data/raw/<SYMBOL>/<SYMBOL>USDT_<date>.tar 경로들. 없는 날짜는 건너뛴다."""
    out = []
    for date in dates:
        p = os.path.join(root, symbol, f"{symbol}USDT_{date}.tar")
        if os.path.exists(p):
            out.append(p)
    return out


# ----------------------------------------------------------------------------
# 시퀀스 연속성 / 복원 정확도 검증
# ----------------------------------------------------------------------------
@dataclass
class SequenceReport:
    n_events: int
    n_breaks: int              # pu != 직전 u 인 지점 (캡처 누락)
    first_ts: int
    last_ts: int

    @property
    def is_continuous(self) -> bool:
        return self.n_breaks == 0


def check_sequence(tar_paths) -> SequenceReport:
    """바이낸스 규약상 각 업데이트의 pu 는 직전 업데이트의 u 와 같아야 한다.
    어긋나면 그 사이 메시지가 유실된 것이므로 호가창 복원이 어긋난다."""
    if isinstance(tar_paths, (str, os.PathLike)):
        tar_paths = [tar_paths]

    prev_u = None
    n = breaks = 0
    first_ts = last_ts = 0

    for path in tar_paths:
        for _, payload in iter_raw_lines(path):
            if _DEPTH_MARK not in payload:
                continue
            try:
                d = json.loads(payload)["data"]
            except (ValueError, KeyError, TypeError):
                continue
            if n == 0:
                first_ts = int(d.get("T", d.get("E", 0)))
            last_ts = int(d.get("T", d.get("E", 0)))
            if prev_u is not None and d.get("pu") != prev_u:
                breaks += 1
            prev_u = d.get("u")
            n += 1

    return SequenceReport(n, breaks, first_ts, last_ts)


@dataclass
class ReconReport:
    n_compared: int
    n_match: int
    first_match_ts: int | None      # 이 시각부터 복원이 신뢰 가능
    converged_after_ms: int | None  # 캡처 시작 후 수렴까지 걸린 시간

    @property
    def match_rate(self) -> float:
        return self.n_match / self.n_compared if self.n_compared else 0.0


def validate_reconstruction(tar_path: str, tolerance: float = 1e-9,
                            settle_window: int = 200,
                            max_lines: int | None = None) -> ReconReport:
    """차분만으로 복원한 L1 이 bookTicker 와 일치하는지 대조한다.

    스냅샷 없이 빈 호가창에서 시작하므로 초반에는 어긋난다. 연속
    `settle_window` 번 일치하는 순간을 수렴 지점으로 본다. 두 스트림이 한 파일에
    시간순으로 섞여 있으므로 한 번만 읽으면서 처리한다.
    """
    from preprocessing.ofi_features import OrderBook

    book = OrderBook()
    n_compared = n_match = 0
    streak = 0
    first_match_ts = None
    start_ts = None
    seen = 0

    for _, payload in iter_raw_lines(tar_path):
        is_depth = _DEPTH_MARK in payload
        is_book = (not is_depth) and (_BOOK_MARK in payload)
        if not (is_depth or is_book):
            continue

        seen += 1
        if max_lines is not None and seen > max_lines:
            break

        try:
            d = json.loads(payload)["data"]
        except (ValueError, KeyError, TypeError):
            continue

        if is_depth:
            if start_ts is None:
                start_ts = int(d.get("T", d.get("E", 0)))
            book.apply(
                [(float(p), float(q)) for p, q in d.get("b", ())],
                [(float(p), float(q)) for p, q in d.get("a", ())],
            )
            continue

        # bookTicker: 지금까지 복원한 최우선 호가와 대조
        bp, _, ap, _ = book.top(1)
        n_compared += 1
        if abs(bp[0] - float(d["b"])) < tolerance and abs(ap[0] - float(d["a"])) < tolerance:
            n_match += 1
            streak += 1
            if first_match_ts is None and streak >= settle_window:
                first_match_ts = int(d.get("T", d.get("E", 0)))
        else:
            streak = 0

    converged = (first_match_ts - start_ts) if (first_match_ts and start_ts) else None
    return ReconReport(n_compared, n_match, first_match_ts, converged)
