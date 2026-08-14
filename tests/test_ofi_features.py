"""사양 §18 (1~16, 23) 과 §19 수작업 검증.

각 테스트는 **틀린 구현이면 반드시 실패하도록** 설계했다. 흔한 오구현을 함께
계산해서 결과가 다름을 확인하는 방식(discriminating assertion)을 쓴다.
"""

import numpy as np
import pytest

import ofi_spec as spec
from preprocessing.ofi_features import (
    build_features,
    event_of_ask,
    event_of_bid,
)

L = spec.LEVELS
BUCKET = spec.BUCKET_MS
W = spec.DEPTH_ROLLING_WINDOW
EPS = spec.EPS

BID_PX = [round(100.00 - 0.01 * i, 4) for i in range(L)]     # 내림차순
ASK_PX = [round(100.01 + 0.01 * i, 4) for i in range(L)]     # 오름차순


def ev(ts, bid_qty=None, ask_qty=None):
    """전 단계 수량을 지정하는 이벤트. 가격은 고정."""
    bq = list(bid_qty) if bid_qty is not None else []
    aq = list(ask_qty) if ask_qty is not None else []
    return (
        ts,
        list(zip(BID_PX, bq)),
        list(zip(ASK_PX, aq)),
    )


def flat(q):
    return [float(q)] * L


# ---------------------------------------------------------------------------
# §18-1  50ms 안에 이벤트가 여러 개여도 timestep 은 1개
# ---------------------------------------------------------------------------
def test_01_multiple_events_collapse_to_one_timestep():
    events = [ev(0, flat(10), flat(10))]
    for i in range(5):                       # 전부 버킷 1 (50~99ms)
        events.append(ev(50 + i, flat(11 + i), flat(10)))

    r = build_features(events)

    assert r.n_buckets == 2, "5개 이벤트가 5개 timestep 이 되면 안 된다"
    assert r.n_events[1] == 5
    assert r.bid_of.shape == (2, L)


# ---------------------------------------------------------------------------
# §18-2,3 + §19  버킷 내부 연쇄 비교 (B0->B1, B1->B2, B2->B3)
# ---------------------------------------------------------------------------
def test_02_03_19_chained_comparison_within_bucket():
    """사양 §19 의 수작업 예제 그대로.

    직전 버킷 최종 잔량 10 -> 15 -> 12 -> 18
    이벤트별 OF: +5, -3, +6  =>  버킷 합계 +8
    """
    events = [
        ev(0, flat(10), flat(10)),      # 버킷 0 최종 상태 B0
        ev(50, flat(15), flat(10)),     # 버킷 1 첫 이벤트  -> +5
        ev(60, flat(12), flat(10)),     # 두 번째           -> -3
        ev(70, flat(18), flat(10)),     # 세 번째           -> +6
    ]
    r = build_features(events)

    assert r.bid_of[1, 0] == pytest.approx(8.0), "버킷 합산 OF 는 +8 이어야 한다"

    # 직전 버킷 최종상태(B0)와 각 이벤트를 개별 비교하는 흔한 오구현은 +15 를 낸다
    wrong = (15 - 10) + (12 - 10) + (18 - 10)
    assert wrong == 15
    assert r.bid_of[1, 0] != wrong


def test_19_ask_side_chained():
    """매도 측 동일 검증. 가격 불변이므로 수량 차분의 합."""
    events = [
        ev(0, flat(10), flat(20)),
        ev(50, flat(10), flat(26)),     # +6
        ev(60, flat(10), flat(21)),     # -5
        ev(70, flat(10), flat(29)),     # +8
    ]
    r = build_features(events)
    assert r.ask_of[1, 0] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# §19  가격이 변하는 경우의 piecewise 공식
# ---------------------------------------------------------------------------
def test_19_piecewise_price_moves_bid():
    prev_px, prev_qty = np.array([100.0]), np.array([10.0])

    # 가격 상승 -> 새 수량 전체가 유입
    assert event_of_bid(prev_px, prev_qty, np.array([100.5]), np.array([7.0]))[0] == 7.0
    # 가격 동일 -> 수량 차분
    assert event_of_bid(prev_px, prev_qty, np.array([100.0]), np.array([7.0]))[0] == -3.0
    # 가격 하락 -> 직전 수량 전체가 이탈
    assert event_of_bid(prev_px, prev_qty, np.array([99.5]), np.array([7.0]))[0] == -10.0


def test_19_piecewise_price_moves_ask():
    prev_px, prev_qty = np.array([100.0]), np.array([10.0])

    # 매도는 부등호가 반대: 가격 하락 -> 새 수량 전체가 유입
    assert event_of_ask(prev_px, prev_qty, np.array([99.5]), np.array([7.0]))[0] == 7.0
    assert event_of_ask(prev_px, prev_qty, np.array([100.0]), np.array([7.0]))[0] == -3.0
    assert event_of_ask(prev_px, prev_qty, np.array([100.5]), np.array([7.0]))[0] == -10.0


def test_19_price_up_through_pipeline():
    """L1 매수 호가가 개선되면 L1 OF = 새 호가의 수량."""
    events = [
        ev(0, flat(10), flat(10)),
        (50, [(100.005, 7.0)], []),      # 기존 100.00 위에 새 호가
    ]
    r = build_features(events)
    assert r.bid_of[1, 0] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# §18-4,5,6  bid 는 bid 끼리, ask 는 ask 끼리. 끝까지 분리 유지
# ---------------------------------------------------------------------------
def test_04_05_06_sides_summed_and_kept_separate():
    events = [
        ev(0, flat(10), flat(20)),
        ev(50, flat(13), flat(17)),     # bid +3, ask -3
        ev(60, flat(15), flat(14)),     # bid +2, ask -3
    ]
    r = build_features(events)

    assert r.bid_of[1, 0] == pytest.approx(5.0)
    assert r.ask_of[1, 0] == pytest.approx(-6.0)

    # 두 방향을 합쳐버리는 오구현이면 5 + (-6) = -1 이 된다
    assert r.bid_of[1, 0] != r.bid_of[1, 0] + r.ask_of[1, 0]

    # 입력 특징에서도 0~9 는 매수, 10~19 는 매도로 분리되어 있어야 한다
    assert spec.OF_FEATURE_NAMES[:L] == [f"bid_of_l{i}" for i in range(1, L + 1)]
    assert spec.OF_FEATURE_NAMES[L:2 * L] == [f"ask_of_l{i}" for i in range(1, L + 1)]
    assert spec.OF_FEATURE_NAMES[2 * L:] == ["ofi_500ms", "ofi_1000ms"]


# ---------------------------------------------------------------------------
# §18-7  이벤트 없는 버킷: 호가창 forward-fill, OF = 0, timestep 유지
# ---------------------------------------------------------------------------
def test_07_empty_bucket_forward_fill_and_zero_of():
    events = [
        ev(0, flat(10), flat(20)),
        ev(50, flat(14), flat(20)),
        # 버킷 2,3,4 는 이벤트 없음
        ev(250, flat(14), flat(20)),    # 버킷 5
    ]
    r = build_features(events)

    assert r.n_buckets == 6, "빈 버킷도 timestep 으로 유지되어야 한다"
    for k in (2, 3, 4):
        assert r.n_events[k] == 0
        assert np.all(r.bid_of[k] == 0.0), "빈 버킷의 OF 는 0"
        assert np.all(r.ask_of[k] == 0.0)
        # 호가창 상태는 직전 버킷 값을 그대로 이어받는다
        assert np.array_equal(r.q_bid_end[k], r.q_bid_end[1])
        assert np.array_equal(r.q_ask_end[k], r.q_ask_end[1])


# ---------------------------------------------------------------------------
# 공용 fixture: 정규화 검증용으로 충분히 긴 스트림
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stream():
    """버킷마다 이벤트 개수가 들쭉날쭉한 300 버킷짜리 스트림."""
    rng = np.random.default_rng(0)
    events = []
    bq = np.full(L, 50.0)
    aq = np.full(L, 60.0)
    for k in range(300):
        n_ev = int(rng.integers(1, 6))          # 버킷당 1~5개 이벤트
        for j in range(n_ev):
            bq = np.clip(bq + rng.normal(0, 3, L), 1, None)
            aq = np.clip(aq + rng.normal(0, 3, L), 1, None)
            ts = k * BUCKET + j                 # 전부 같은 버킷 안
            events.append(ev(ts, bq.tolist(), aq.tolist()))
    return build_features(events)


# ---------------------------------------------------------------------------
# §18-8,9,23  OF 정규화: 같은 방향·같은 단계·과거 100버킷, 현재 제외
# ---------------------------------------------------------------------------
def test_08_09_23_bid_of_normalized_by_past_100_same_level(stream):
    r = stream
    for k in (W, W + 37, r.n_buckets - 1):
        expected_denom = r.q_bid_end[k - W:k, 0].mean()      # 현재 버킷 k 제외
        expected = r.bid_of[k, 0] / (expected_denom + EPS)
        assert r.features[k, 0] == pytest.approx(expected, rel=1e-9)

    # rolling 이 이벤트 수에 따라 움직였다면 이 등식이 깨진다.
    # (버킷별 이벤트 수가 1~5개로 제각각인 스트림이므로 판별력이 있다)
    assert r.n_events[W:].min() != r.n_events[W:].max()


def test_10_ask_of_normalized_by_past_100_same_level(stream):
    r = stream
    for k in (W, W + 51, r.n_buckets - 1):
        expected_denom = r.q_ask_end[k - W:k, 0].mean()
        expected = r.ask_of[k, 0] / (expected_denom + EPS)
        assert r.features[k, L] == pytest.approx(expected, rel=1e-9)


def test_23_current_bucket_excluded_from_rolling_mean(stream):
    """현재 버킷을 포함하는 오구현과 결과가 달라야 한다."""
    r = stream
    k = W + 40
    correct = r.q_bid_end[k - W:k, 0].mean()
    wrong_inclusive = r.q_bid_end[k - W + 1:k + 1, 0].mean()
    assert correct != pytest.approx(wrong_inclusive)
    assert r.features[k, 0] == pytest.approx(r.bid_of[k, 0] / (correct + EPS), rel=1e-9)


def test_09b_each_level_uses_its_own_depth(stream):
    """L2 는 L2 잔량으로, L10 은 L10 잔량으로 나눠야 한다."""
    r = stream
    k = W + 25
    for lev in (1, 5, L - 1):
        expected = r.bid_of[k, lev] / (r.q_bid_end[k - W:k, lev].mean() + EPS)
        assert r.features[k, lev] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# §18-11  raw_ofi 는 원시 OF 로 계산
# ---------------------------------------------------------------------------
def test_11_raw_ofi_uses_raw_not_normalized(stream):
    r = stream
    k = W + 10
    expected = float((r.bid_of[k] - r.ask_of[k]).sum())
    assert r.raw_ofi[k] == pytest.approx(expected, rel=1e-12)

    # 정규화된 값으로 계산한 오구현과는 달라야 한다
    wrong = float((r.features[k, :L] - r.features[k, L:2 * L]).sum())
    assert r.raw_ofi[k] != pytest.approx(wrong)


# ---------------------------------------------------------------------------
# §18-12,13,14,15,16  OFI: 먼저 합산 -> 과거 100버킷 전체 평균잔량으로 한 번 나눔
# ---------------------------------------------------------------------------
def _mean_total_depth(r, k):
    total = (r.q_bid_end + r.q_ask_end).sum(axis=1) / (2 * L)
    return total[k - W:k].mean()


def test_12_ofi10_numerator_is_sum_of_last_10_raw_ofi(stream):
    r = stream
    k = W + 33
    num = r.raw_ofi[k - 9:k + 1].sum()               # 현재 버킷 포함 10개
    expected = num / (_mean_total_depth(r, k) + EPS)
    assert r.features[k, 2 * L] == pytest.approx(expected, rel=1e-9)


def test_13_ofi20_numerator_is_sum_of_last_20_raw_ofi(stream):
    r = stream
    k = W + 33
    num = r.raw_ofi[k - 19:k + 1].sum()
    expected = num / (_mean_total_depth(r, k) + EPS)
    assert r.features[k, 2 * L + 1] == pytest.approx(expected, rel=1e-9)


def test_14_both_ofi_share_the_same_denominator(stream):
    r = stream
    k = W + 77
    ratio_features = r.features[k, 2 * L] / r.features[k, 2 * L + 1]
    ratio_numerators = r.raw_ofi[k - 9:k + 1].sum() / r.raw_ofi[k - 19:k + 1].sum()
    assert ratio_features == pytest.approx(ratio_numerators, rel=1e-9)


def test_15_not_divided_by_recent_10_or_20_depth(stream):
    """최근 10개/20개 depth 합(또는 평균)으로 나누는 오구현과 달라야 한다."""
    r = stream
    k = W + 33
    total = (r.q_bid_end + r.q_ask_end).sum(axis=1) / (2 * L)
    num = r.raw_ofi[k - 9:k + 1].sum()

    wrong_sum10 = num / (total[k - 9:k + 1].sum() + EPS)
    wrong_mean10 = num / (total[k - 9:k + 1].mean() + EPS)
    got = r.features[k, 2 * L]

    assert got != pytest.approx(wrong_sum10)
    assert got != pytest.approx(wrong_mean10)


def test_16_not_individually_normalized_then_averaged(stream):
    """각 raw_ofi 를 따로 정규화한 뒤 평균내는 오구현과 달라야 한다."""
    r = stream
    k = W + 33
    total = (r.q_bid_end + r.q_ask_end).sum(axis=1) / (2 * L)
    wrong = np.mean([
        r.raw_ofi[k - j] / (total[k - j - W:k - j].mean() + EPS) for j in range(10)
    ])
    assert r.features[k, 2 * L] != pytest.approx(wrong)


# ---------------------------------------------------------------------------
# §18-17  한 시점 특징 shape == [22]  /  §11 warm-up 처리
# ---------------------------------------------------------------------------
def test_17_single_timestep_feature_shape(stream):
    assert stream.features.shape[1] == spec.OF_DIM == 22
    assert stream.features[0].shape == (22,)


def test_11_warmup_dropped_not_zero_filled(stream):
    r = stream
    assert not r.feature_valid[:W].any(), "warm-up 구간은 유효하지 않아야 한다"
    assert r.feature_valid[W:].all(), "과거가 갖춰진 뒤로는 전부 유효"
    # NaN 을 0 으로 때워넣지 않았는지 확인
    assert np.isnan(r.features[:W]).any()
    assert np.isfinite(r.features[r.feature_valid]).all()


def test_12_mid_price_from_bucket_end_state(stream):
    r = stream
    assert r.mid_valid[r.feature_valid].all()
    assert np.isfinite(r.mid[r.feature_valid]).all()
    # 고정 호가이므로 미드는 항상 100.005
    assert r.mid[r.feature_valid][0] == pytest.approx((BID_PX[0] + ASK_PX[0]) / 2)


# ---------------------------------------------------------------------------
# 경계 조건: 반열린 구간, 경계 시각은 다음 버킷 (§3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ts,expected_bucket", [(0, 0), (49, 0), (50, 1), (99, 1), (100, 2)])
def test_03_half_open_bucket_boundary(ts, expected_bucket):
    events = [ev(0, flat(10), flat(10)), ev(ts, flat(11), flat(10))]
    r = build_features(events)
    assert r.n_buckets == expected_bucket + 1


def test_events_must_be_time_ordered():
    events = [ev(0, flat(10), flat(10)), ev(200, flat(11), flat(10)), ev(50, flat(12), flat(10))]
    with pytest.raises(ValueError, match="시간순"):
        build_features(events)


# ---------------------------------------------------------------------------
# 이벤트 단위 버킷
# ---------------------------------------------------------------------------
def _stream(n=200, seed=0):
    r = np.random.default_rng(seed)
    ev, t = [], 1000
    for _ in range(n):
        t += int(r.choice([1, 2, 3, 500]))          # 가끔 큰 공백
        ev.append((t,
                   [(100.0 - j * 0.01, float(r.integers(1, 9))) for j in range(10)],
                   [(100.1 + j * 0.01, float(r.integers(1, 9))) for j in range(10)]))
    return ev


def test_event_buckets_have_no_empty_rows():
    """시간 격자는 종목마다 빈 칸 비율이 3.8~32.7% 로 벌어진다 (실측).
    이벤트로 자르면 빈 칸이 구조적으로 0 이 된다."""
    ev = _stream()
    t = build_features(ev, bucket_ms=spec.BUCKET_MS)
    e = build_features(ev, bucket_events=1)

    assert (t.n_events == 0).mean() > 0.5, "시간 격자에는 빈 버킷이 많다"
    assert (e.n_events == 0).mean() == 0.0, "이벤트 버킷에는 빈 칸이 없어야 한다"
    assert e.n_buckets == len(ev), "이벤트 1건 = 1칸"


def test_event_buckets_group_by_count():
    ev = _stream(200)
    for k in (1, 5, 10):
        r = build_features(ev, bucket_events=k)
        assert r.n_buckets == -(-len(ev) // k), (k, r.n_buckets)


def test_event_buckets_carry_timestamps():
    """이벤트 버킷은 간격이 불규칙하므로 시각을 따로 들고 다녀야 한다."""
    ev = _stream()
    r = build_features(ev, bucket_events=1)
    assert r.ts_ms.size == r.n_buckets
    assert np.all(np.diff(r.ts_ms) >= 0), "시각은 단조증가해야 한다"
    assert r.ts_ms[-1] == ev[-1][0]


def test_repairs_do_not_advance_the_event_counter():
    """장부 정정은 실제 주문흐름이 아니므로 칸을 만들지 않는다."""
    base = [(1000 + i, [(100.0, 5.0)], [(101.0, 5.0)]) for i in range(6)]
    with_rep = []
    for i, e in enumerate(base):
        with_rep.append(e)
        if i % 2 == 0:
            with_rep.append((e[0], [(99.0, 0.0)], [], True))     # 정정
    a = build_features([(t, b, k) for t, b, k in base], bucket_events=2)
    c = build_features(with_rep, bucket_events=2)
    assert a.n_buckets == c.n_buckets, "정정이 칸 수를 바꾸면 안 된다"
