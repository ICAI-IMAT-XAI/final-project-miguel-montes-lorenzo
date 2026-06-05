"""Model performance scores used by perturbation-based XAI checks."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import PortfolioPrediction


def model_log_returns(
    prediction: PortfolioPrediction,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Compute one-period log returns from predicted allocation weights."""
    return (prediction.weights * batch["next_log_returns"]).sum(dim=-1)


def equal_weight_log_returns(batch: dict[str, Any]) -> torch.Tensor:
    """Compute one-period log returns for the equal-weight standard portfolio."""
    return batch["next_log_returns"].mean(dim=-1)


def annualized_growth(log_returns: torch.Tensor, periods_per_year: int = 252) -> float:
    """Return annualized gross growth from one-period log returns."""
    if log_returns.numel() == 0:
        return 1.0
    mean_log_return = log_returns.detach().float().mean()
    return float(torch.exp(mean_log_return * float(periods_per_year)).cpu().item())


def annualized_return(log_returns: torch.Tensor, periods_per_year: int = 252) -> float:
    """Return annualized arithmetic return from one-period log returns."""
    return annualized_growth(
        log_returns=log_returns,
        periods_per_year=periods_per_year,
    ) - 1.0


def performance_ratio(
    model_returns: torch.Tensor,
    standard_returns: torch.Tensor,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compare model performance with the equal-weight standard portfolio."""
    model_growth = annualized_growth(
        log_returns=model_returns,
        periods_per_year=periods_per_year,
    )
    standard_growth = annualized_growth(
        log_returns=standard_returns,
        periods_per_year=periods_per_year,
    )
    denominator = standard_growth if abs(standard_growth) > 1.0e-12 else 1.0e-12
    return {
        "performance_ratio": model_growth / denominator,
        "model_annualized_return": model_growth - 1.0,
        "standard_annualized_return": standard_growth - 1.0,
    }
