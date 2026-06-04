"""Pointwise neural network for allocation-target prediction."""

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


class ReturnPredictor(nn.Module):
    """Feed-forward predictor aligned with the working Dirichlet NN baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
        output_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        activation_name = activation.lower()
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            if activation_name == "relu":
                layers.append(nn.ReLU())
            elif activation_name == "tanh":
                layers.append(nn.Tanh())
            elif activation_name == "gelu":
                layers.append(nn.GELU())
            else:
                raise KeyError(f"Unsupported activation: {activation}.")
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NNModel(PortfolioModule):
    """Predict next-day Markowitz weights from previous optimal portfolios."""

    def __init__(self, config: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Create the deterministic pointwise MLP.

        Args:
            config: Model configuration.
            metadata: Processed dataset metadata.
        """
        super().__init__(config=config)
        lookback: int = int(metadata["lookback"])
        portfolio_dim: int = int(metadata["portfolio_input_dim"])
        input_dim: int = lookback * portfolio_dim
        hidden_dims: list[int] = [
            int(value) for value in config.get("hidden_dims", [128])
        ]
        dropout: float = float(config.get("dropout", 0.0))
        activation: str = str(config.get("activation", "gelu"))
        output_dim = len(metadata["currencies"]) + 1
        self.net = ReturnPredictor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            output_dim=output_dim,
        )
        self.num_currencies = len(metadata["currencies"])

    def forward(self, batch: dict[str, Any]) -> PortfolioPrediction:
        """Run one deterministic forward pass.

        Args:
            batch: Batch containing ``portfolio_features``.

        Returns:
            Predicted allocation weights and variance.
        """
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
        output: torch.Tensor = self.net(features)
        raw_alpha = output[:, : self.num_currencies]
        raw_variance = output[:, self.num_currencies :]
        return dirichlet_prediction(
            raw_alpha=raw_alpha,
            variance=positive_variance(raw_variance=raw_variance),
        )
