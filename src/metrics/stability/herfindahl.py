"""Portfolio concentration metrics."""

from __future__ import annotations

import numpy as np


def mean_herfindahl(weights: np.ndarray) -> float:
    """Compute the average Herfindahl concentration index.

    Args:
        weights: Weight matrix with shape ``(T, N)``.

    Returns:
        Mean Herfindahl index.
    """
    if weights.size == 0:
        return 0.0
    return float(np.mean(a=np.sum(a=weights**2, axis=1)))
