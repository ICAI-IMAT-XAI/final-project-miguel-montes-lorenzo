"""Bayesian HTGNN implemented with MC dropout."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import PortfolioPrediction
from src.models.pointwise.HTGNN.model import HTGNNModel


class BHTGNNModel(HTGNNModel):
    """MC-dropout approximation to a Bayesian HTGNN."""

    def predict(
        self,
        batch: dict[str, Any],
        mc_samples: int = 50,
    ) -> PortfolioPrediction:
        """Estimate predictive mean and uncertainty using MC dropout.

        Args:
            batch: Collated dataset batch.
            mc_samples: Number of stochastic dropout passes.

        Returns:
            Mean prediction and predictive standard deviations.
        """
        was_training: bool = self.training
        self.train(mode=True)
        weight_samples: list[torch.Tensor] = []
        variance_samples: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(mc_samples):
                sample: PortfolioPrediction = self.forward(batch=batch)
                weight_samples.append(sample.weights)
                variance_samples.append(sample.variance)
        if not was_training:
            self.eval()
        stacked_weights: torch.Tensor = torch.stack(tensors=weight_samples, dim=0)
        stacked_variances: torch.Tensor = torch.stack(tensors=variance_samples, dim=0)
        return PortfolioPrediction(
            weights=stacked_weights.mean(dim=0),
            variance=stacked_variances.mean(dim=0),
            weight_uncertainty=stacked_weights.std(dim=0, unbiased=False),
            variance_uncertainty=stacked_variances.std(dim=0, unbiased=False),
        )
