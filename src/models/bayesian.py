"""Bayesian neural-network building blocks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class BayesianLinear(nn.Module):
    """Bayes-by-backprop linear layer with diagonal Gaussian posterior."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float) -> None:
        super().__init__()
        self.prior_sigma: float = float(prior_sigma)
        self.weight_mu: nn.Parameter = nn.Parameter(
            data=torch.empty(size=(out_features, in_features))
        )
        self.weight_rho: nn.Parameter = nn.Parameter(
            data=torch.empty(size=(out_features, in_features))
        )
        self.bias_mu: nn.Parameter = nn.Parameter(data=torch.empty(size=(out_features,)))
        self.bias_rho: nn.Parameter = nn.Parameter(data=torch.empty(size=(out_features,)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize posterior parameters."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound: float = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3.0)

    @property
    def weight_sigma(self) -> torch.Tensor:
        """Positive posterior weight standard deviation."""
        return F.softplus(input=self.weight_rho)

    @property
    def bias_sigma(self) -> torch.Tensor:
        """Positive posterior bias standard deviation."""
        return F.softplus(input=self.bias_rho)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample weights and return output plus KL divergence."""
        weight: torch.Tensor = self.weight_mu + self.weight_sigma * torch.randn_like(
            input=self.weight_mu
        )
        bias: torch.Tensor = self.bias_mu + self.bias_sigma * torch.randn_like(
            input=self.bias_mu
        )
        output: torch.Tensor = F.linear(input=x, weight=weight, bias=bias)
        kl: torch.Tensor = self._kl(mu=self.weight_mu, sigma=self.weight_sigma)
        kl = kl + self._kl(mu=self.bias_mu, sigma=self.bias_sigma)
        return output, kl

    def _kl(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """KL divergence KL(q || p) against zero-mean Gaussian prior."""
        prior_sigma: torch.Tensor = torch.as_tensor(
            data=self.prior_sigma,
            dtype=mu.dtype,
            device=mu.device,
        )
        return (
            torch.log(input=prior_sigma)
            - torch.log(input=sigma.clamp_min(min=1.0e-8))
            + (sigma.pow(exponent=2) + mu.pow(exponent=2))
            / (2.0 * prior_sigma.pow(exponent=2))
            - 0.5
        ).sum()


class BayesianMLP(nn.Module):
    """Bayesian MLP returning output and accumulated KL."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        prior_sigma: float,
        dropout: float,
    ) -> None:
        super().__init__()
        dims: list[int] = [input_dim, *hidden_dims]
        self.hidden_layers: nn.ModuleList = nn.ModuleList(
            [
                BayesianLinear(
                    in_features=dims[index],
                    out_features=dims[index + 1],
                    prior_sigma=prior_sigma,
                )
                for index in range(len(dims) - 1)
            ]
        )
        self.output_layer: BayesianLinear = BayesianLinear(
            in_features=dims[-1],
            out_features=output_dim,
            prior_sigma=prior_sigma,
        )
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one stochastic forward pass."""
        total_kl: torch.Tensor = torch.zeros((), dtype=x.dtype, device=x.device)
        for layer in self.hidden_layers:
            x, kl = layer(x)
            total_kl = total_kl + kl
            x = F.relu(input=x)
            x = self.dropout(x)
        output, kl = self.output_layer(x)
        return output, total_kl + kl
