from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch import Tensor

from src.data.load_utils import TorchDataLoader, generate_torch_dataloader
from src.models.metrics.pipelines import complete_metrics


def _init_check(
    feature_size: int,
    output_mask: Tensor,
    temporal_lookback: int,
    temporal_horizon: int,
) -> None:
    assert isinstance(feature_size, int)
    assert isinstance(output_mask, Tensor)
    assert output_mask.dtype is torch.bool
    assert output_mask.ndim == 1, "output mask is not 1-dimensional"
    assert output_mask.shape[0] == feature_size, (
        "output mask size doesnt match feature_size"
    )
    assert int(output_mask.to(dtype=int).sum().item()) > 0, (
        "output_mask cannot be all false"
    )
    assert isinstance(temporal_lookback, int)
    assert isinstance(temporal_horizon, int)

    return None


class ForecastingModel(ABC):
    def __init__(
        self,
        feature_size: int,
        output_mask: Tensor,
        temporal_lookback: int,
        temporal_horizon: int,
    ) -> None:

        _init_check(
            feature_size=feature_size,
            output_mask=output_mask,
            temporal_lookback=temporal_lookback,
            temporal_horizon=temporal_horizon,
        )
        self._feature_size: int = feature_size
        self._output_mask: Tensor = output_mask
        self._temporal_lookback: int = temporal_lookback
        self._temporal_horizon: int = temporal_horizon

        self._init_model(
            feature_size=feature_size,
            output_mask=output_mask,
            temporal_lookback=temporal_lookback,
            temporal_horizon=temporal_horizon,
        )

        return None

    @abstractmethod
    def _init_model(
        self,
        feature_size: int,
        output_mask: Tensor,
        temporal_lookback: int,
        temporal_horizon: int,
    ) -> None:
        """Initialize the model parameters."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_data: pl.DataFrame) -> None:
        """Train the model on the given data."""
        raise NotImplementedError

    def predict(self, x: Tensor) -> Tensor:
        """Predict a single output from input features."""

        squeeze_batch: bool = False
        squeeze_feature: bool = False

        if x.ndim < 3 and x.shape[-1] != 1 and self._feature_size == 1:
            squeeze_feature = True
            assert int(self._output_mask.to(dtype=int).sum().item()) == 1
            x = x.unsqueeze(dim=(-1))
        if (
            x.ndim == 2
            and x.shape[0] == self._temporal_lookback
            and x.shape[1] == self._feature_size
        ):
            squeeze_batch = True
            x = x.unsqueeze(dim=0)

        assert isinstance(x, Tensor)
        assert x.ndim == 3, "x must have shape (batch, tmp_lookback, feature_size)"
        assert x.shape[1] == self._temporal_lookback
        assert x.shape[2] == self._feature_size

        y: Tensor = self._predict(x=x)

        assert x.shape[0] == y.shape[0], "prediction must keep the batch size"
        assert y.shape[1] == self._temporal_horizon
        assert y.shape[2] == int(self._output_mask.to(dtype=int).sum().item())

        if squeeze_batch:
            assert y.shape[0] == 1
            y.squeeze(dim=0)
        if squeeze_feature:
            assert y.shape[-1] == 1
            y.squeeze(dim=(-1))

        return y

    @abstractmethod
    def _predict(self, x: Tensor) -> Tensor:
        """Predict a single output from input features."""
        raise NotImplementedError

    def evaluate(
        self,
        test_data: pl.DataFrame,
        batch_size: int = 32,
        drop_last: bool = False,
        metrics_pipeline: Callable[..., dict[str, Tensor]] | None = None,
        feature_reduction: str | None = "mean",
        horizon_reduction: str | None = "mean",
    ) -> dict[str, float]:
        """Evaluate the model on a test split using batched sliding windows.

        This method:
        1) Builds a torch DataLoader of (x, y) windows from ``test_data`` using the
            model configuration (lookback, horizon, output_mask).
        2) For each batch, computes predictions via ``self.predict``.
        3) Computes per-batch metrics via ``metrics_pipeline`` if provided,
            otherwise via ``complete_metrics``.
        4) Aggregates metrics across batches using a weighted average by the
            effective batch size.

        Args:
            test_data: Polars DataFrame containing only the feature columns used by
                the model.
            batch_size: Batch size used to create evaluation windows.
            drop_last: Whether to drop the last incomplete batch.
            metrics_pipeline: Optional callable replacing ``complete_metrics``. It must
                have signature ``(Y, Y_hat, feature_reduction, horizon_reduction)
                -> dict[str, Tensor]``.
            feature_reduction: Reduction over the feature dimension passed to the
                metrics pipeline.
            horizon_reduction: Reduction over the horizon dimension passed to the
                metrics pipeline.

        Returns:
            Dictionary mapping metric names to Python floats.
        """
        assert isinstance(test_data, pl.DataFrame)
        assert test_data.height > 0
        assert test_data.width == self._feature_size

        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert isinstance(drop_last, bool)
        assert feature_reduction in ("sum", "mean", None)
        assert horizon_reduction in ("sum", "mean", None)

        if metrics_pipeline is None:
            metrics_pipeline = complete_metrics

        float_dtypes: set[pl.DataType] = {pl.Float32, pl.Float64}
        for col in test_data.columns:
            dtype: pl.DataType = test_data.schema[col]
            assert dtype in float_dtypes, (
                f"Column '{col}' must be float dtype, got {dtype}."
            )

        loader: TorchDataLoader = generate_torch_dataloader(
            dataframe=test_data,
            output_mask=self._output_mask,
            batch=batch_size,
            lookback=self._temporal_lookback,
            horizon=self._temporal_horizon,
            slide=1,
            drop_last=drop_last,
            shuffle=False,
        )

        sums: dict[str, float] = {}
        total_n: int = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                b: int = int(x_batch.shape[0])
                assert b > 0

                y_hat_batch: Tensor = self.predict(x=x_batch)
                assert y_hat_batch.shape == y_batch.shape

                batch_metrics: dict[str, Tensor] = metrics_pipeline(
                    Y=y_batch,
                    Y_hat=y_hat_batch,
                    feature_reduction=feature_reduction,
                    horizon_reduction=horizon_reduction,
                )

                for name, value in batch_metrics.items():
                    value_scalar: float = float(value.reshape(-1).mean().item())
                    if name not in sums:
                        sums[name] = 0.0
                    sums[name] += value_scalar * float(b)

                total_n += b

        assert total_n > 0
        metrics: dict[str, float] = {k: v / float(total_n) for k, v in sums.items()}

        return dict(sorted(metrics.items()))

    @abstractmethod
    def save(self, store_path: Path) -> None:
        """Save the model configuration."""
        raise NotImplementedError

    @abstractmethod
    def load(self, store_path: Path) -> None:
        """Load a saved model configuration."""
        raise NotImplementedError

    @property
    def config(self) -> dict[str, Any]:
        return self._get_config()

    @abstractmethod
    def _get_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @config.setter
    def config(self, config: dict[str, Any]) -> None:
        return self._set_config(config=config)

    @abstractmethod
    def _set_config(self, config: dict[str, Any]) -> None:
        raise NotImplementedError
