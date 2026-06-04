"""MLP building blocks for pointwise and probabilistic models."""

from __future__ import annotations

from torch import nn

from src.models.base import get_activation


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    activation: str,
) -> nn.Sequential:
    """Build a feed-forward MLP trunk.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        dropout: Dropout probability after each hidden activation.
        activation: Activation name.

    Returns:
        Sequential PyTorch module.
    """
    layers: list[nn.Module] = []
    previous_dim: int = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_features=previous_dim, out_features=hidden_dim))
        layers.append(get_activation(name=activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        previous_dim = hidden_dim
    return nn.Sequential(*layers)
