"""Portfolio turnover metrics."""

from __future__ import annotations

import numpy as np


def mean_turnover(weights: np.ndarray) -> float:
    """Compute mean one-way portfolio turnover.

    Args:
        weights: Weight matrix with shape ``(T, N)``.

    Returns:
        Average one-way turnover.
    """
    if weights.shape[0] < 2:
        return 0.0
    absolute_changes: np.ndarray = np.abs(weights[1:] - weights[:-1])
    return float(np.mean(a=0.5 * np.sum(a=absolute_changes, axis=1)))
