"""Sharpe ratio metric."""

from __future__ import annotations

import numpy as np


def sharpe_ratio(log_returns: np.ndarray, trading_days_per_year: int) -> float:
    """Compute annualized Sharpe ratio with zero risk-free rate.

    Args:
        log_returns: Daily portfolio log returns.
        trading_days_per_year: Number of trading days per year.

    Returns:
        Annualized Sharpe ratio.
    """
    if log_returns.size < 2:
        return 0.0
    volatility: float = float(np.std(a=log_returns, ddof=1))
    if volatility <= 1.0e-12 or not np.isfinite(volatility):
        return 0.0
    return float(np.mean(a=log_returns) / volatility * np.sqrt(trading_days_per_year))
