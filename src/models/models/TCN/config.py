from dataclasses import dataclass

import torch


@dataclass(frozen=False)
class TCNModelConfig:
    """Configuration for a TCN forecaster."""

    in_features: int | None = None  # to be defined
    out_features: int | None = None  # to be defined
    channels: tuple[int, ...] = (128, 256, 128, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    blocks_per_dilation: int = 1


@dataclass(frozen=False)
class TCNTrainConfig:
    """Configuration for a TCN forecaster."""

    epochs: int = 100
    batch_size: int = 16
    eval_proportion: float = 0.15

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float | None = 1.0

    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu"

    # OneCycleLR tends to work well for TCN-like conv nets without validation.
    max_lr: float | None = None  # if None -> uses lr
    pct_start: float = 0.1
    div_factor: float = 25.0
    final_div_factor: float = 1e4
