"""Drawdown metrics."""

from __future__ import annotations

import numpy as np


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Compute maximum drawdown from a portfolio value path.

    Args:
        portfolio_values: Portfolio value series.

    Returns:
        Minimum drawdown as a negative number.
    """
    if portfolio_values.size == 0:
        return 0.0
    running_max: np.ndarray = np.maximum.accumulate(portfolio_values)
    drawdowns: np.ndarray = portfolio_values / np.maximum(running_max, 1.0e-12) - 1.0
    return float(np.min(drawdowns))
