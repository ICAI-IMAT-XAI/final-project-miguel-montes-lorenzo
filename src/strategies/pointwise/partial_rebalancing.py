"""Partial rebalancing strategy."""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy


class PartialRebalancingStrategy(BaseStrategy):
    """Move only a fraction of the way toward the target each period."""

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
        del uncertainties, returns, t
        alpha = float(cfg.get("alpha", 0.2))
        new_weights = current_weights + alpha * (target_weights - current_weights)
        total = float(new_weights.sum())
        if total <= 1.0e-8:
            return target_weights.copy()
        return new_weights / total
