"""Bayesian pointwise neural network with Dirichlet allocation head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.base import (
    PortfolioModule,
    PortfolioPrediction,
    build_mean_variance_weight_features,
    dirichlet_prediction,
    positive_variance,
)
from src.models.bayesian import BayesianMLP


class BNNModel(PortfolioModule):
    """Bayes-by-backprop BNN predicting Dirichlet portfolio concentrations."""

    def __init__(self, config: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Create the Bayesian pointwise network."""
        super().__init__(config=config)
        lookback: int = int(metadata["lookback"])
        portfolio_dim: int = int(metadata["portfolio_input_dim"])
        input_dim: int = lookback * portfolio_dim
        hidden_dims: list[int] = [
            int(value) for value in config.get("hidden_dims", [256, 256, 128])
        ]
        output_dim: int = len(metadata["currencies"]) + 1
        self.net: BayesianMLP = BayesianMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            prior_sigma=float(config.get("prior_sigma", 1.0)),
            dropout=float(config.get("dropout", 0.0)),
        )
        self.num_currencies: int = len(metadata["currencies"])
        self.last_kl: torch.Tensor | None = None

    def forward(self, batch: dict[str, Any]) -> PortfolioPrediction:
        """Run one stochastic Bayesian forward pass."""
        input_format: str = str(self.config.get("input_format", "returns")).lower()
        if input_format == "weights":
            features = build_mean_variance_weight_features(
                returns_window=batch["portfolio_raw_returns"],
                risk_aversion=float(self.config.get("risk_aversion", 1.0)),
                ridge=float(self.config.get("ridge", 1.0e-4)),
                allow_short=bool(self.config.get("allow_short", False)),
            ).flatten(start_dim=1)
        else:
            features = batch["portfolio_raw_returns"].flatten(start_dim=1)
        output, kl = self.net(features)
        self.last_kl = kl
        raw_alpha: torch.Tensor = output[:, : self.num_currencies]
        raw_variance: torch.Tensor = output[:, self.num_currencies :]
        return dirichlet_prediction(
            raw_alpha=raw_alpha,
            variance=positive_variance(raw_variance=raw_variance),
        )

    def predict(
        self,
        batch: dict[str, Any],
        mc_samples: int = 50,
    ) -> PortfolioPrediction:
        """Estimate predictive mean and uncertainty using posterior samples."""
        was_training: bool = self.training
        self.eval()
        weight_samples: list[torch.Tensor] = []
        variance_samples: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(mc_samples):
                sample: PortfolioPrediction = self.forward(batch=batch)
                weight_samples.append(sample.weights)
                variance_samples.append(sample.variance)
        if was_training:
            self.train()
        stacked_weights: torch.Tensor = torch.stack(tensors=weight_samples, dim=0)
        stacked_variances: torch.Tensor = torch.stack(tensors=variance_samples, dim=0)
        return PortfolioPrediction(
            weights=stacked_weights.mean(dim=0),
            variance=stacked_variances.mean(dim=0),
            weight_uncertainty=stacked_weights.std(dim=0, unbiased=False),
            variance_uncertainty=stacked_variances.std(dim=0, unbiased=False),
        )
