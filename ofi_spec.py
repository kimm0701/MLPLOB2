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
TARGET_HORIZONS_SEC = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
INPUT_DIM = 22                  # bid_of x10 + ask_of x10 + ofi_500ms + ofi_1000ms
OUTPUT_DIM = 10                 # horizon 별 signed mid-price return
EPS = 1e-8
LOSS_TYPE = "mse"               # "mse" | "huber"

# 미래 수익률 horizon 을 50ms 버킷 개수로 환산: 1초 = 20버킷 ... 10초 = 200버킷
TARGET_HORIZON_BUCKETS = [s * 1000 // BUCKET_MS for s in TARGET_HORIZONS_SEC]

# 특징 순서. 데이터 생성 / Dataset / 모델 입력 / 저장 파일 / 추론 전 구간 동일.
FEATURE_NAMES = (
    [f"bid_of_l{i}" for i in range(1, LEVELS + 1)]
    + [f"ask_of_l{i}" for i in range(1, LEVELS + 1)]
    + ["ofi_500ms", "ofi_1000ms"]
)

assert len(FEATURE_NAMES) == INPUT_DIM
assert len(TARGET_HORIZON_BUCKETS) == OUTPUT_DIM
assert TARGET_HORIZON_BUCKETS[0] == 20 and TARGET_HORIZON_BUCKETS[-1] == 200
