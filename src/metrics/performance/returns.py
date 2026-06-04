"""Return-based performance metrics."""

from __future__ import annotations

import numpy as np


def cumulative_return(portfolio_values: np.ndarray) -> float:
    """Compute cumulative return from a portfolio value path.

    Args:
        portfolio_values: Portfolio value series.

    Returns:
        Cumulative return over the whole period.
    """
    if portfolio_values.size < 2:
        return 0.0
    return float(portfolio_values[-1] / portfolio_values[0] - 1.0)


def annualized_return(log_returns: np.ndarray, trading_days_per_year: int) -> float:
    """Compute annualized return from daily log returns.

    Args:
        log_returns: Daily portfolio log returns.
        trading_days_per_year: Number of trading days per year.

    Returns:
        Annualized arithmetic return implied by mean log return.
    """
    if log_returns.size == 0:
        return 0.0
    return float(np.exp(np.mean(a=log_returns) * trading_days_per_year) - 1.0)
