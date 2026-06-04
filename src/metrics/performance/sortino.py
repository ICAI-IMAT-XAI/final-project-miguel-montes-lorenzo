"""Sortino ratio metric."""

from __future__ import annotations

import numpy as np


def sortino_ratio(log_returns: np.ndarray, trading_days_per_year: int) -> float:
    """Compute annualized Sortino ratio with zero target return.

    Args:
        log_returns: Daily portfolio log returns.
        trading_days_per_year: Number of trading days per year.

    Returns:
        Annualized Sortino ratio.
    """
    downside: np.ndarray = log_returns[log_returns < 0.0]
    if downside.size < 2:
        return 0.0
    downside_volatility: float = float(np.std(a=downside, ddof=1))
    if downside_volatility <= 1.0e-12:
        return 0.0
    return float(
        np.mean(a=log_returns) / downside_volatility * np.sqrt(trading_days_per_year)
    )
