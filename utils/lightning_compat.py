"""`lightning` 과 `pytorch_lightning` 중 있는 쪽을 쓴다.

같은 프로젝트를 두 이름으로 배포하고 있어서 환경마다 한쪽만 깔려 있는 일이
흔하다. 이 저장소의 requirements.txt 에는 둘 다 적혀 있지만 실제 인스턴스에는
pytorch_lightning 만 설치돼 있었고, `import lightning` 하나 때문에 학습이
시작도 못 하고 죽었다.

없는 쪽을 pip 로 새로 까는 건 위험하다. `pip install lightning` 은 torch 와
nvidia CUDA 패키지 전체를 함께 끌어와서, 이미 맞춰 놓은 GPU 환경(torch
2.5.0+cu121)을 다른 빌드로 덮어쓸 수 있다.
"""

BACKEND = None

try:
    import lightning as L                                            # noqa: F401
    from lightning import LightningModule, seed_everything, Trainer  # noqa: F401
    from lightning.pytorch.callbacks import (                        # noqa: F401
        Callback,
        EarlyStopping,
        TQDMProgressBar,
    )

    BACKEND = "lightning"
except ImportError:
    try:
        import pytorch_lightning as L                                # noqa: F401
        from pytorch_lightning import (                              # noqa: F401
            LightningModule,
            seed_everything,
            Trainer,
        )
        from pytorch_lightning.callbacks import (                    # noqa: F401
            Callback,
            EarlyStopping,
            TQDMProgressBar,
        )

        BACKEND = "pytorch_lightning"
    except ImportError as exc:                                       # pragma: no cover
        raise ImportError(
            "lightning 도 pytorch_lightning 도 없습니다.\n"
            "이미 동작하는 GPU 환경(torch 2.5.0+cu121)을 건드리지 않으려면\n"
            "    pip install --no-deps pytorch_lightning\n"
            "처럼 의존성 없이 설치하세요. 그냥 pip install 하면 torch 를 다른\n"
            "빌드로 덮어써서 GPU 가 안 잡힐 수 있습니다."
        ) from exc


__all__ = [
    "BACKEND",
    "Callback",
    "EarlyStopping",
    "L",
    "LightningModule",
    "TQDMProgressBar",
    "Trainer",
    "seed_everything",
]
