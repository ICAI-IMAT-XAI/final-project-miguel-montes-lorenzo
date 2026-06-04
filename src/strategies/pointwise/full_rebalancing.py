"""Full rebalancing baseline."""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy


class FullRebalancingStrategy(BaseStrategy):
    """Rebalance exactly to the model target every period."""

    requires_uncertainty = False

    def rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        uncertainties: np.ndarray | None,
        returns: np.ndarray,
        t: int,
        cfg: dict,
    ) -> np.ndarray:
        del current_weights, uncertainties, returns, t, cfg
        return target_weights.copy()
