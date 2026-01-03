import torch
import torch.nn as nn
from torch import Tensor

from src.models.models.TCN.config import TCNModelConfig
from src.models.models.TCN.utils import receptive_field


class Chomp1d(nn.Module):
    """Remove extra right-padding to preserve causality."""

    def __init__(self, chomp_size: int) -> None:
        """Initialize.

        Args:
            chomp_size: Number of timesteps to remove from the right side.
        """
        super().__init__()
        assert chomp_size >= 0
        self._chomp_size: int = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        """Apply chomp.

        Args:
            x: Tensor of shape (batch, channels, length).

        Returns:
            Tensor of shape (batch, channels, length - chomp_size).
        """
        if self._chomp_size == 0:
            return x
        return x[..., : -self._chomp_size]


class CausalConv1d(nn.Module):
    """1D convolution with left padding to ensure causality."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
    ) -> None:
        """Initialize a causal Conv1d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            dilation: Dilation factor.
            bias: Whether to use bias.
        """
        super().__init__()
        assert in_channels >= 1
        assert out_channels >= 1
        assert kernel_size >= 1
        assert dilation >= 1

        self._pad_left: int = (kernel_size - 1) * dilation
        self._conv: nn.Conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, channels, length).

        Returns:
            Tensor of shape (batch, out_channels, length).
        """
        x_padded: Tensor = nn.functional.pad(
            input=x,
            pad=(self._pad_left, 0),
            mode="constant",
            value=0.0,
        )
        return self._conv(x_padded)


class TemporalBlock(nn.Module):
    """Residual block for TCN with two causal convolutions."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        """Initialize a temporal residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            dilation: Dilation factor.
            dropout: Dropout probability.
        """
        super().__init__()
        assert dropout >= 0.0

        self._conv1: CausalConv1d = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self._act1: nn.ReLU = nn.ReLU()
        self._drop1: nn.Dropout = nn.Dropout(p=dropout)

        self._conv2: CausalConv1d = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self._act2: nn.ReLU = nn.ReLU()
        self._drop2: nn.Dropout = nn.Dropout(p=dropout)

        self._downsample: nn.Module
        if in_channels != out_channels:
            self._downsample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            )
        else:
            self._downsample = nn.Identity()

        self._out_act: nn.ReLU = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, in_channels, length).

        Returns:
            Tensor of shape (batch, out_channels, length).
        """
        y: Tensor = self._conv1(x)
        y = self._act1(y)
        y = self._drop1(y)

        y = self._conv2(y)
        y = self._act2(y)
        y = self._drop2(y)

        res: Tensor = self._downsample(x)
        return self._out_act(y + res)


class TCN(nn.Module):
    """TCN for multivariate time series forecasting (horizon = 1)."""

    def __init__(self, *, cfg: TCNModelConfig) -> None:
        """Initialize the model.

        Expected input: x with shape (batch, lookback, in_features).
        Output: y_hat with shape (batch, out_features).

        Args:
            cfg: TCN hyperparameters.
        """
        super().__init__()
        assert cfg.in_features >= 1
        assert cfg.out_features >= 1
        assert len(cfg.channels) >= 1
        assert cfg.kernel_size >= 1
        assert cfg.blocks_per_dilation >= 1
        assert len(cfg.dilations) >= 1

        # Build temporal feature extractor.
        layers: list[nn.Module] = []
        current_channels: int = cfg.in_features
        channel_schedule: list[int] = list(cfg.channels)

        # If channels length doesn't match dilations*blocks_per_dilation, repeat
        # or truncate the channel schedule in a simple, predictable way.
        total_blocks: int = len(cfg.dilations) * cfg.blocks_per_dilation
        if len(channel_schedule) < total_blocks:
            repeats: int = (total_blocks + len(channel_schedule) - 1) // len(
                channel_schedule
            )
            channel_schedule = (channel_schedule * repeats)[:total_blocks]
        else:
            channel_schedule = channel_schedule[:total_blocks]

        idx: int = 0
        for d in cfg.dilations:
            for _ in range(cfg.blocks_per_dilation):
                out_ch: int = channel_schedule[idx]
                idx += 1
                layers.append(
                    TemporalBlock(
                        in_channels=current_channels,
                        out_channels=out_ch,
                        kernel_size=cfg.kernel_size,
                        dilation=d,
                        dropout=cfg.dropout,
                    )
                )
                current_channels = out_ch

        self._tcn: nn.Sequential = nn.Sequential(*layers)

        # Head: use the last timestep representation to predict horizon=1.
        self._head: nn.Linear = nn.Linear(
            in_features=current_channels,
            out_features=cfg.out_features,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor with shape (batch, lookback, in_features).

        Returns:
            Prediction tensor with shape (batch, out_features).
        """
        assert x.ndim == 3, "Expected (batch, lookback, in_features)"
        x_c: Tensor = x.transpose(dim0=1, dim1=2)  # (batch, in_features, lookback)
        z: Tensor = self._tcn(x_c)  # (batch, channels, lookback)
        last: Tensor = z[:, :, -1]  # (batch, channels)
        return self._head(last)


if __name__ == "__main__":
    cfg: TCNModelConfig = TCNModelConfig(
        in_features=31,
        out_features=12,
    )

    print(cfg)
    assert False

    rf: int = receptive_field(
        kernel_size=cfg.kernel_size,
        dilations=cfg.dilations,
        blocks_per_dilation=cfg.blocks_per_dilation,
    )
    print(f"Receptive field: {rf} timesteps")

    lookback: int = 52  # try 26 / 52 / 104
    batch: int = 16
    x: Tensor = torch.randn(batch, lookback, cfg.in_features)

    model: TCN = TCN(cfg=cfg)
    y_hat: Tensor = model(x)
    print(f"y_hat.shape = {tuple(y_hat.shape)}")
