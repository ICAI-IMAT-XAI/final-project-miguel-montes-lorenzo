from collections.abc import Callable

import torch
from torch import Tensor


def generic_loss(
    Y: Tensor,  # (B, H, F)
    Y_hat: Tensor,  # (B, H, F)
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> Tensor:
    """Compute a generic elementwise loss over batched target/prediction tensors.

    The function applies ``loss_fn`` elementwise to ``(Y_hat, Y)``, averages
    across the batch dimension, and then applies optional reductions across
    horizon and/or feature dimensions.

    Tensor conventions:
      - Input shape: (B, H, F)
        * B: batch size
        * H: horizon
        * F: feature size

    Args:
        Y: Target tensor of shape (B, H, F).
        Y_hat: Prediction tensor of shape (B, H, F).
        loss_fn: Callable mapping (y_hat, y) to an elementwise loss tensor of
            the same shape.
        feature_reduction: Reduction over the feature dimension ("sum", "mean",
            or None).
        horizon_reduction: Reduction over the horizon dimension ("sum", "mean",
            or None).

    Returns:
        A tensor with shape determined by the applied reductions. Reduced
        dimensions are retained with size 1 (``keepdim=True``).
    """
    assert isinstance(Y, Tensor)
    assert isinstance(Y_hat, Tensor)
    assert Y.shape == Y_hat.shape
    assert Y.ndim == 3  # (B, H, F)

    assert feature_reduction in ("sum", "mean", None)
    assert horizon_reduction in ("sum", "mean", None)

    # Elementwise loss: (B, H, F)
    loss: Tensor = loss_fn(Y_hat, Y)

    # Average across batch dimension
    out: Tensor = loss.mean(dim=0)  # (H, F)

    # Reduce feature dimension (dim=1)
    if feature_reduction == "mean":
        out = out.mean(dim=1, keepdim=True)
    elif feature_reduction == "sum":
        out = out.sum(dim=1, keepdim=True)

    # Reduce horizon dimension (dim=0)
    if horizon_reduction == "mean":
        out = out.mean(dim=0, keepdim=True)
    elif horizon_reduction == "sum":
        out = out.sum(dim=0, keepdim=True)

    return out


def mse(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> Tensor:
    """Mean squared error."""

    def _mse_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        return (y_hat - y) ** 2

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_mse_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def rmse(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> Tensor:
    """Root mean squared error."""
    return torch.sqrt(
        input=mse(
            Y=Y,
            Y_hat=Y_hat,
            feature_reduction=feature_reduction,
            horizon_reduction=horizon_reduction,
        )
    )


def mae(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> Tensor:
    """Mean absolute error."""

    def _mae_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        return torch.abs(input=y_hat - y)

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_mae_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def mape(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
    eps: float = 1e-12,
) -> Tensor:
    """Mean absolute percentage error."""
    assert eps > 0.0

    def _mape_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        denom: Tensor = torch.clamp(input=torch.abs(input=y), min=eps)
        return torch.abs(input=(y_hat - y) / denom)

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_mape_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def smape(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
    eps: float = 1e-12,
) -> Tensor:
    """Symmetric mean absolute percentage error."""
    assert eps > 0.0

    def _smape_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        denom: Tensor = torch.clamp(
            input=torch.abs(input=y) + torch.abs(input=y_hat), min=eps
        )
        return 2.0 * torch.abs(input=y_hat - y) / denom

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_smape_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def huber(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
    delta: float = 1.0,
) -> Tensor:
    """Huber loss (robust L2 near zero, L1 in tails).

    Args:
        Y: Target tensor (B, H, F).
        Y_hat: Prediction tensor (B, H, F).
        feature_reduction: Reduction over features.
        horizon_reduction: Reduction over horizon.
        delta: Threshold where loss transitions from quadratic to linear.

    Returns:
        Reduced Huber loss tensor.
    """
    assert delta > 0.0

    def _huber_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        err: Tensor = y_hat - y
        abs_err: Tensor = torch.abs(input=err)
        quadratic: Tensor = 0.5 * err**2
        linear: Tensor = delta * (abs_err - 0.5 * delta)
        return torch.where(condition=abs_err <= delta, input=quadratic, other=linear)

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_huber_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def log_cosh(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
) -> Tensor:
    """Log-cosh loss (smooth, robust-ish alternative to MSE).

    Args:
        Y: Target tensor (B, H, F).
        Y_hat: Prediction tensor (B, H, F).
        feature_reduction: Reduction over features.
        horizon_reduction: Reduction over horizon.

    Returns:
        Reduced log-cosh loss tensor.
    """

    def _log_cosh_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        err: Tensor = y_hat - y
        return torch.log(input=torch.cosh(input=err))

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_log_cosh_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def pinball(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
    q: float = 0.5,
) -> Tensor:
    """Quantile (pinball) loss.

    Useful for quantile forecasting (e.g., q=0.1, 0.5, 0.9).

    Args:
        Y: Target tensor (B, H, F).
        Y_hat: Prediction tensor (B, H, F).
        feature_reduction: Reduction over features.
        horizon_reduction: Reduction over horizon.
        q: Quantile in (0, 1).

    Returns:
        Reduced pinball loss tensor.
    """
    assert 0.0 < q < 1.0

    def _pinball_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        e: Tensor = y - y_hat
        # max(q*e, (q-1)*e)
        return torch.maximum(input=q * e, other=(q - 1.0) * e)

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_pinball_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )


def wape(
    Y: Tensor,
    Y_hat: Tensor,
    feature_reduction: str | None = "mean",
    horizon_reduction: str | None = "mean",
    eps: float = 1e-12,
) -> Tensor:
    """Weighted absolute percentage error (WAPE) in a generic_loss-compatible way.

    Nota: WAPE se define típicamente como sum(|e|)/sum(|y|) sobre un conjunto.
    Aquí lo implemento como un 'percentage-like' elementwise:
      |e| / clamp(mean(|y| over batch), eps)
    para mantener el mismo esquema de reducciones.

    Args:
        Y: Target tensor (B, H, F).
        Y_hat: Prediction tensor (B, H, F).
        feature_reduction: Reduction over features.
        horizon_reduction: Reduction over horizon.
        eps: Small positive constant to avoid division by zero.

    Returns:
        Reduced WAPE-like tensor.
    """
    assert eps > 0.0

    def _wape_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        abs_err: Tensor = torch.abs(input=y_hat - y)  # (B,H,F)
        denom: Tensor = torch.clamp(
            input=torch.abs(input=y).mean(dim=0, keepdim=True), min=eps
        )  # (1,H,F)
        return abs_err / denom

    return generic_loss(
        Y=Y,
        Y_hat=Y_hat,
        loss_fn=_wape_loss,
        feature_reduction=feature_reduction,
        horizon_reduction=horizon_reduction,
    )
