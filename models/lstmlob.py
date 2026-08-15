"""LSTM 기반 회귀 모델.

Kolm/Turiel/Westray (2023) Table 1 의 LSTM 구성을 따른다.

    입력 [B, 100, 24]  ->  LSTM(hidden, num_layers)  ->  마지막 시점 은닉값
                       ->  Linear(hidden -> OUTPUT_DIM)

논문에서 LSTM 계열은 MLP/ARX 계열보다 확연히 앞섰고, 그중 단순한 LSTM 하나가
LSTM-MLP·LSTM(3)·CNN-LSTM 과 사실상 동등했다. 그래서 기본형을 단층 LSTM 으로
두고 깊이는 num_layers 로 열어 둔다.

MLPLOB 과 __init__ 인자가 같다. 같은 자리에 그대로 끼울 수 있어야 학습·보정·
백테스트 스크립트를 건드리지 않는다.

BiN(표본 안에서의 적응 정규화)은 기본으로 켠다. 논문에는 없지만 우리 파이프라인이
줄곧 쓰던 것이라, 구조 교체의 효과만 따로 보려면 이 부분은 그대로 두는 편이 맞다.
use_bin=False 로 끄고 비교할 수 있다.
"""

from torch import nn

import ofi_spec as spec
from models.bin import BiN


class LSTMLOB(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 dataset_type: str,
                 dropout: float = 0.0,
                 use_bin: bool = True,
                 ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dataset_type = dataset_type
        self.use_bin = use_bin

        self.norm_layer = BiN(num_features, seq_size) if use_bin else None
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch 는 층 사이에만 드롭아웃을 넣는다. 단층이면 무의미하고
            # 경고만 뜨므로 아예 0 으로 준다.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 회귀 머리. 수익률은 음수도 되므로 마지막에 활성함수를 두지 않는다.
        self.head = nn.Linear(hidden_dim, spec.OUTPUT_DIM)

    def forward(self, input):
        x = input
        if self.norm_layer is not None:
            # BiN 은 [B, 피처, 시간] 배치를 받는다
            x = x.permute(0, 2, 1)
            x = self.norm_layer(x)
            x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        # 마지막 시점의 은닉값만 쓴다 = 창 전체를 요약한 상태
        return self.head(out[:, -1])
