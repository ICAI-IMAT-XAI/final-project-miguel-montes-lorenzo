"""Common model outputs and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.dirichlet_losses import build_loss_fn


@dataclass(frozen=True)
class PortfolioPrediction:
    """Portfolio model prediction container.

    Attributes:
        weights: Long-only allocation weights with shape ``(B, N)``.
        variance: Predicted portfolio variance with shape ``(B, 1)``.
    weight_uncertainty: Optional uncertainty over weights with shape ``(B, N)``.
    variance_uncertainty: Optional uncertainty over variance with shape ``(B, 1)``.
        alpha: Optional Dirichlet concentration parameters with shape ``(B, N)``.
    """

    weights: torch.Tensor
    variance: torch.Tensor
    weight_uncertainty: torch.Tensor | None = None
    variance_uncertainty: torch.Tensor | None = None
    alpha: torch.Tensor | None = None


class PortfolioModule(nn.Module):
    """Base class for trainable portfolio modules."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Store a serializable model configuration.

        Args:
            config: Merged model configuration.
        """
        super().__init__()
        self.config: dict[str, Any] = config

    def predict(
        self,
        batch: dict[str, Any],
        mc_samples: int = 1,
    ) -> PortfolioPrediction:
        """Predict a portfolio allocation for a batch.

        Args:
            batch: Collated dataset batch.
            mc_samples: Number of stochastic samples for probabilistic models.

        Returns:
            Portfolio prediction.
        """
        del mc_samples
        return self.forward(batch=batch)


def get_activation(name: str) -> nn.Module:
    """Create an activation module by name.

    Args:
        name: Activation identifier.

    Returns:
        PyTorch activation module.
    """
    normalized: str = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    raise KeyError(f"Unsupported activation: {name}.")


def positive_variance(raw_variance: torch.Tensor) -> torch.Tensor:
    """Map unconstrained network outputs to positive variance values.

    Args:
        raw_variance: Unconstrained variance logits.

    Returns:
        Positive variance tensor.
    """
    return torch.nn.functional.softplus(input=raw_variance) + 1.0e-8


def dirichlet_alpha(raw_alpha: torch.Tensor) -> torch.Tensor:
    """Map unconstrained network outputs to Dirichlet concentrations."""
    return F.softplus(input=raw_alpha) + 1.0e-4


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    """Compute the mean weights implied by Dirichlet concentrations."""
    return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(min=1.0e-8)


def dirichlet_std(alpha: torch.Tensor) -> torch.Tensor:
    """Compute marginal standard deviation for Dirichlet weights."""
    alpha_0: torch.Tensor = alpha.sum(dim=-1, keepdim=True).clamp_min(min=1.0e-8)
    variance: torch.Tensor = alpha * (alpha_0 - alpha)
    variance = variance / (alpha_0.pow(exponent=2) * (alpha_0 + 1.0))
    return variance.clamp_min(min=0.0).sqrt()


def dirichlet_prediction(
    raw_alpha: torch.Tensor,
    variance: torch.Tensor,
) -> PortfolioPrediction:
    """Build a portfolio prediction from raw Dirichlet logits."""
    alpha: torch.Tensor = dirichlet_alpha(raw_alpha=raw_alpha)
    return PortfolioPrediction(
        weights=dirichlet_mean(alpha=alpha),
        variance=variance,
        weight_uncertainty=dirichlet_std(alpha=alpha),
        alpha=alpha,
    )


def _normalize_target_weights(target: torch.Tensor) -> torch.Tensor:
    """Make target weights strictly positive and normalized."""
    target = target.clamp_min(min=1.0e-8)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(min=1.0e-8)


def center_cross_sectional_returns(returns: torch.Tensor) -> torch.Tensor:
    """Center each cross-section of currency returns around zero."""
    return returns - returns.mean(dim=-1, keepdim=True)


def estimate_covariance(
    returns_window: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Estimate per-batch covariance matrices from return windows."""
    batch_size, window_size, num_assets = returns_window.shape
    if window_size < 2:
        eye = torch.eye(num_assets, device=returns_window.device, dtype=returns_window.dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1) * (1.0 + ridge)
    centered = returns_window - returns_window.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered
    covariance = covariance / float(max(window_size - 1, 1))
    eye = torch.eye(num_assets, device=returns_window.device, dtype=returns_window.dtype)
    return covariance + float(ridge) * eye.unsqueeze(0)


def project_to_simplex_torch(values: torch.Tensor) -> torch.Tensor:
    """Project batched vectors onto the probability simplex."""
    if values.ndim == 1:
        values = values.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    sorted_values, _ = torch.sort(values, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_values, dim=-1)
    steps = torch.arange(
        1,
        values.shape[-1] + 1,
        device=values.device,
        dtype=values.dtype,
    ).unsqueeze(0)
    mask = sorted_values * steps > (cumsum - 1.0)
    rho = mask.sum(dim=-1).clamp_min(1) - 1
    theta = (cumsum.gather(1, rho.unsqueeze(1)) - 1.0) / (rho.unsqueeze(1).to(values.dtype) + 1.0)
    projected = torch.clamp(values - theta, min=0.0)
    total = projected.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(projected, fill_value=1.0 / projected.shape[-1])
    projected = torch.where(total > 0.0, projected / total.clamp_min(1.0e-8), fallback)
    if squeeze:
        return projected.squeeze(0)
    return projected


def solve_mean_variance_weights(
    mu: torch.Tensor,
    covariance: torch.Tensor,
    risk_aversion: float,
    allow_short: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a batch of mean-variance allocation problems."""
    raw_weights: list[torch.Tensor] = []
    for mean_vector, cov_matrix in zip(mu, covariance, strict=False):
        scaled_cov = float(risk_aversion) * cov_matrix
        try:
            weights = torch.linalg.solve(scaled_cov, mean_vector)
        except RuntimeError:
            weights = torch.linalg.pinv(scaled_cov) @ mean_vector
        raw_weights.append(weights)
    stacked = torch.stack(raw_weights, dim=0)
    if allow_short:
        denominator = stacked.sum(dim=-1, keepdim=True)
        fallback = torch.full_like(stacked, fill_value=1.0 / stacked.shape[-1])
        weights = torch.where(
            denominator.abs() > 1.0e-8,
            stacked / denominator,
            fallback,
        )
    else:
        weights = project_to_simplex_torch(stacked)
    variance = torch.einsum("bi,bij,bj->b", weights, covariance, weights).unsqueeze(-1)
    return weights, variance.clamp_min(0.0)


def build_mean_variance_targets(
    next_log_returns: torch.Tensor,
    returns_window: torch.Tensor,
    risk_aversion: float,
    ridge: float,
    allow_short: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct teacher weights from centered next returns and historical covariance."""
    mu = center_cross_sectional_returns(next_log_returns)
    covariance = estimate_covariance(returns_window=returns_window, ridge=ridge)
    return solve_mean_variance_weights(
        mu=mu,
        covariance=covariance,
        risk_aversion=risk_aversion,
        allow_short=allow_short,
    )


def build_mean_variance_weight_features(
    returns_window: torch.Tensor,
    risk_aversion: float,
    ridge: float,
    allow_short: bool,
) -> torch.Tensor:
    """Convert return windows into sequences of rolling mean-variance weights."""
    batch_size, lookback, num_assets = returns_window.shape
    equal_weights = torch.full(
        size=(batch_size, num_assets),
        fill_value=1.0 / float(num_assets),
        device=returns_window.device,
        dtype=returns_window.dtype,
    )
    features: list[torch.Tensor] = []
    for time_idx in range(lookback):
        if time_idx == 0:
            features.append(equal_weights)
            continue
        mu = center_cross_sectional_returns(returns_window[:, time_idx - 1, :])
        covariance = estimate_covariance(
            returns_window=returns_window[:, :time_idx, :],
            ridge=ridge,
        )
        weights, _ = solve_mean_variance_weights(
            mu=mu,
            covariance=covariance,
            risk_aversion=risk_aversion,
            allow_short=allow_short,
        )
        features.append(weights)
    return torch.stack(features, dim=1).reshape(batch_size, lookback, num_assets)


def portfolio_loss(
    prediction: PortfolioPrediction,
    batch: dict[str, Any],
    weights_loss_coef: float,
    variance_loss_coef: float,
    loss_name: str = "mse",
    loss_regularizations: dict[str, Any] | None = None,
    extra_kl: torch.Tensor | None = None,
    n_data: int | None = None,
    risk_aversion: float = 1.0,
    ridge: float = 1.0e-4,
    allow_short: bool = False,
) -> torch.Tensor:
    """Compute the supervised portfolio allocation loss.

    Args:
        prediction: Model prediction.
        batch: Training batch with target tensors.
        weights_loss_coef: Coefficient for weight MSE.
        variance_loss_coef: Coefficient for variance MSE on log scale.

    Returns:
        Scalar training loss.
    """
    target_weights, target_variance = build_mean_variance_targets(
        next_log_returns=batch["next_log_returns"],
        returns_window=batch["portfolio_raw_returns"],
        risk_aversion=risk_aversion,
        ridge=ridge,
        allow_short=allow_short,
    )
    alpha_prediction: torch.Tensor = (
        prediction.alpha
        if prediction.alpha is not None
        else prediction.weights.clamp_min(1.0e-6)
    )
    loss_fn = build_loss_fn(
        {
            "loss": loss_name,
            "loss_regularizations": loss_regularizations,
        }
    )
    sigma = estimate_covariance(
        returns_window=batch["portfolio_raw_returns"],
        ridge=ridge,
    )
    weights_loss: torch.Tensor = loss_fn(
        alpha_pred=alpha_prediction,
        label=_normalize_target_weights(target_weights),
        lookback=int(batch["portfolio_raw_returns"].shape[1]),
        sigma=sigma,
        market_ret=batch["next_log_returns"].mean(dim=-1),
    )
    predicted_log_variance: torch.Tensor = torch.log(input=prediction.variance)
    target_log_variance: torch.Tensor = torch.log(input=target_variance + 1.0e-8)
    variance_loss: torch.Tensor = torch.mean(
        input=(predicted_log_variance - target_log_variance).pow(exponent=2)
    )
    loss: torch.Tensor = weights_loss_coef * weights_loss + variance_loss_coef * variance_loss
    regularizations: dict[str, Any] = loss_regularizations or {}
    kl_weight: float = float(regularizations.get("weight") or 0.0)
    if extra_kl is not None and kl_weight:
        denominator: float = float(max(n_data or int(target_weights.shape[0]), 1))
        loss = loss + kl_weight * extra_kl / denominator
    return loss
