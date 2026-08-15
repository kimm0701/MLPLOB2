"""구조 이름 -> 모델 클래스.

model_config 안에 arch 를 같이 저장해 두면, 체크포인트만 보고도 어떤 구조로
다시 세워야 하는지 알 수 있다. arch 가 없는 예전 체크포인트는 MLPLOB 로 본다
(그때는 그것밖에 없었다).
"""

from models.lstmlob import LSTMLOB
from models.mlplob import MLPLOB

ARCHITECTURES = {
    "mlplob": MLPLOB,
    "lstm": LSTMLOB,
}

DEFAULT_ARCH = "mlplob"


def build_model(cfg: dict):
    """model_config 로 모델을 만든다. cfg 는 바뀌지 않는다."""
    cfg = dict(cfg)
    arch = cfg.pop("arch", DEFAULT_ARCH)
    if arch not in ARCHITECTURES:
        raise KeyError(
            f"모르는 구조 {arch!r} 입니다. 쓸 수 있는 것: "
            f"{', '.join(sorted(ARCHITECTURES))}")
    return ARCHITECTURES[arch](**cfg)
