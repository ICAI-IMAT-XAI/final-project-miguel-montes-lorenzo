from collections.abc import Callable

from torch import Tensor

from src.models.metrics.metrics import (
    huber,
    log_cosh,
    mae,
    mape,
    mse,
    pinball,
    rmse,
    smape,
    wape,
)


def basic_metrics(
    Y: Tensor,  # (B, H, F)
    Y_hat: Tensor,  # (B, H, F)
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> dict[str, Tensor]:
    """Compute a basic set of point-forecast error metrics for batched forecasts.

    This function evaluates several standard regression/forecasting metrics
    (MSE, RMSE, MAE, MAPE, sMAPE) on the same set of targets and predictions.
    All metrics share the same reduction semantics across feature and horizon
    dimensions, after averaging across the batch dimension.

    Tensor conventions:
      - Input shape: (B, H, F)
        * B: batch size
        * H: horizon
        * F: feature size

    Args:
        Y: Target tensor of shape (B, H, F).
        Y_hat: Prediction tensor of shape (B, H, F).
        feature_reduction: Reduction over the feature dimension ("sum", "mean",
            or None).
        horizon_reduction: Reduction over the horizon dimension ("sum", "mean",
            or None).

    Returns:
        Dictionary mapping metric names ("mse", "rmse", "mae", "mape", "smape")
        to their corresponding evaluation tensors. Reduced dimensions are kept
        with size 1 when a reduction is applied (``keepdim=True``).
    """
    assert isinstance(Y, Tensor)
    assert isinstance(Y_hat, Tensor)
    assert Y.shape == Y_hat.shape
    assert Y.ndim == 3

    metrics: dict[str, Tensor] = {}

    for loss_fn in (mse, rmse, mae, mape, smape):
        fn_name: str = loss_fn.__name__
        fn_eval: Tensor = loss_fn(
            Y=Y,
            Y_hat=Y_hat,
            feature_reduction=feature_reduction,
            horizon_reduction=horizon_reduction,
        )
        metrics[fn_name] = fn_eval

    return metrics


def complete_metrics(
    Y: Tensor,  # (B, H, F)
    Y_hat: Tensor,  # (B, H, F)
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> dict[str, Tensor]:
    """Compute an extended set of point-forecast error metrics for batched forecasts.

    This function extends ``basic_metrics`` by including additional losses that
    are commonly useful in time-series forecasting, especially under the
    presence of outliers or when robustness is desired.

    The evaluated metrics are:
      - MSE
      - RMSE
      - MAE
      - MAPE
      - sMAPE
      - Huber
      - log-cosh
      - Pinball (quantile loss, q=0.5)
      - WAPE

    All metrics share the same reduction semantics across feature and horizon
    dimensions, after averaging across the batch dimension.

    Tensor conventions:
      - Input shape: (B, H, F)
        * B: batch size
        * H: horizon
        * F: feature size

    Args:
        Y: Target tensor of shape (B, H, F).
        Y_hat: Prediction tensor of shape (B, H, F).
        feature_reduction: Reduction over the feature dimension ("sum", "mean",
            or None).
        horizon_reduction: Reduction over the horizon dimension ("sum", "mean",
            or None).

    Returns:
        Dictionary mapping metric names to their corresponding evaluation
        tensors. Reduced dimensions are kept with size 1 when a reduction is
        applied (``keepdim=True``).
    """
    assert isinstance(Y, Tensor)
    assert isinstance(Y_hat, Tensor)
    assert Y.shape == Y_hat.shape
    assert Y.ndim == 3

    metrics: dict[str, Tensor] = {}

    metric_fns: tuple[Callable[..., Tensor], ...] = (
        mse,
        rmse,
        mae,
        mape,
        smape,
        huber,
        log_cosh,
        pinball,
        wape,
    )

    for loss_fn in metric_fns:
        fn_name: str = loss_fn.__name__
        fn_eval: Tensor = loss_fn(
            Y=Y,
            Y_hat=Y_hat,
            feature_reduction=feature_reduction,
            horizon_reduction=horizon_reduction,
        )
        metrics[fn_name] = fn_eval

    return metrics
