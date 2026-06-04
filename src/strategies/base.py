"""Abstract base class for all rebalancing strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseStrategy(ABC):
    """Base class for rebalancing strategies."""

    requires_uncertainty: bool = False

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        uncertainties: np.ndarray | None,
        returns: np.ndarray,
        t: int,
        cfg: dict,
    ) -> np.ndarray:
        """Compute weights to hold for the current period."""
