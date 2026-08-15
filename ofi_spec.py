"""OF / OFI 회귀 사양의 단일 진실 공급원 (single source of truth).

constants.py 는 torch 와 pytorch_lightning 을 끌어오므로, 전처리 엔진과 단위
테스트가 딥러닝 스택 없이도 돌아가도록 순수 상수만 여기에 둔다.
constants.py 가 이 모듈을 re-export 하므로 `cst.BUCKET_MS` 도 그대로 동작한다.

여기 값들은 사양 고정값이다. 하이퍼파라미터 탐색이 건드리면 안 된다.
"""

BUCKET_MS = 50                  # 원본 이벤트를 묶는 시간 단위
LEVELS = 10                     # L1 ~ L10
SEQ_LEN = 100                   # 모델 입력 timestep 수 (100 x 50ms = 과거 5초)
DEPTH_ROLLING_WINDOW = 100      # OF/OFI 정규화용 평균잔량 구간 (과거 5초)
OFI_WINDOWS = [10, 20]          # 최근 0.5초 / 1초

# horizon 은 **절대 초** 로 고정한다 (종목 공통).
#
# Delta_t 배수(종목별 시간 스케일링)를 검토했다가 접었다. 명분이 "종목 간
# 비교 가능" 인데 실측에서 확인되지 않았다 - 같은 Delta_t 배수에서 종목 간
# 신호 편차가 0.55~0.87 로, 절대 초(0.78~0.93) 와 비슷하게 컸다. Delta_t 는
# "가격이 얼마나 자주 변하나" 만 맞추는데 종목 차이는 "얼마나 예측 가능한가"
# 에서 오기 때문이다 (MRVL 은 Delta_t 가 중간인데 신호가 가장 강했다).
#
# 반면 절대 초는 두 가지를 준다.
#   - 범용 모델의 출력 칸이 종목과 무관하게 같은 뜻을 갖는다. 모델은 지금
#     보는 게 어느 종목인지 모르므로 이게 중요하다.
#   - 레이턴시가 절대 시간이라 실행 제약을 직접 통제할 수 있다.
#
# 길이는 0.5~2.0 초. 더 긴 horizon 을 넣지 않는 이유는 horizon 별 표준화를
# 하지 않기 때문이다 - 분산이 horizon 에 정비례해서 긴 쪽이 손실을 가져간다.
# 실측(AMD): 0.5~3.0 초 6 개면 1 초 이하가 손실의 14.8% 만 받는데, 정작 신호는
# 0.5 초가 0.0562 로 3.0 초(0.0218) 의 2.6 배다. 0.5~2.0 초 4 개로 줄이면
# 1 초 이하가 30.7% 를 받는다.
TARGET_HORIZONS_SEC = [0.5, 1.0, 1.5, 2.0]
INPUT_DIM = 24                  # 주문흐름 22 + 호가창 상태 2
OUTPUT_DIM = 4                  # horizon 별 signed mid-price 변화 (틱)
EPS = 1e-8
LOSS_TYPE = "mse"               # "mse" | "huber"

# 새 정답의 버킷 수는 종목마다 다르므로 여기서 고정하지 않는다 (targets.json).
#
# 아래는 **구 방식** 전용이다. preprocess.py 가 만드는 _y.npy 는 bp 기준
# 1~10초 고정 horizon 이고, 비교용으로 남겨 둔다. 새 학습은 쓰지 않는다.
LEGACY_HORIZONS_SEC = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TARGET_HORIZON_BUCKETS = [s * 1000 // BUCKET_MS for s in LEGACY_HORIZONS_SEC]

# 특징 순서. 데이터 생성 / Dataset / 모델 입력 / 저장 파일 / 추론 전 구간 동일.
# 뒤의 2개는 호가창 **상태** 변수다. 나머지 22개는 평균 잔량으로 나눠 무차원인데,
# 그 과정에서 "지금 호가창이 어떤 상태인가" 가 지워진다. 같은 주문흐름이라도
# 스프레드가 1틱이면 큐 소진 경쟁이고 넓으면 안쪽 진입 경쟁이라 뜻이 다르다.
# 종목 신분증이 아니라 시점마다 바뀌는 값이므로 종목 수와 무관하게 정보를 준다.
# **종목별 z-score 를 씌우면 안 된다** - 그러면 구분이 지워진다.
FEATURE_NAMES = (
    [f"bid_of_l{i}" for i in range(1, LEVELS + 1)]
    + [f"ask_of_l{i}" for i in range(1, LEVELS + 1)]
    + ["ofi_500ms", "ofi_1000ms"]
    + ["log_spread_ticks", "log_secs_since_move"]
)
STATE_FEATURES = 2              # 뒤에서 이만큼은 종목별 정규화에서 제외한다
# 전처리 엔진(ofi_features.py)이 만드는 열 수. 상태 변수 2개는 그 뒤에
# scripts/make_features.py 가 _px.npy 에서 계산해 덧붙인다.
OF_DIM = INPUT_DIM - STATE_FEATURES
OF_FEATURE_NAMES = FEATURE_NAMES[:OF_DIM]

assert len(FEATURE_NAMES) == INPUT_DIM
assert len(TARGET_HORIZONS_SEC) == OUTPUT_DIM
