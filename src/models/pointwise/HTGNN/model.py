"""Heterogeneous temporal graph neural network for FOREX allocation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.base import (
    PortfolioModule,
    PortfolioPrediction,
    dirichlet_prediction,
    get_activation,
    positive_variance,
)


def classify_relation(source: str, target: str) -> str:
    """Assign a semantic relation type to a directed edge.

    Args:
        source: Source node name.
        target: Target node name.

    Returns:
        Relation type used by the message-passing layer.
    """
    if source.startswith("generic_latent_"):
        if target == "portfolio_signal":
            return "generic_to_portfolio"
        if target.startswith("generic_latent_"):
            return "generic_cross"
        return "generic_to_market"
    if target.startswith("generic_latent_"):
        if source == "portfolio_signal":
            return "portfolio_to_generic"
        return "market_to_generic"
    if source == "portfolio_signal":
        return "portfolio_to_market"
    if target == "portfolio_signal":
        if "fx" in source:
            return "fx_to_portfolio"
        if "bond" in source or "treasury" in source:
            return "rates_to_portfolio"
        if "commodity" in source:
            return "commodity_to_portfolio"
        if "equity" in source:
            return "equity_to_portfolio"
        return "macro_to_portfolio"
    return "market_cross"


class HeterogeneousMessagePassingLayer(nn.Module):
    """Typed message-passing layer over fixed heterogeneous nodes."""

    def __init__(
        self,
        hidden_dim: int,
        relation_names: list[str],
        dropout: float,
        activation: str,
        gate: str,
    ) -> None:
        """Create relation-specific transforms and node updater.

        Args:
            hidden_dim: Hidden state dimension for every node.
            relation_names: Edge relation types present in the graph.
            dropout: Dropout probability after message update.
            activation: Activation name used inside the update block.
            gate: Message update gate type. ``none`` preserves the ungated update;
                ``vectorial`` learns one gate value per hidden dimension.
        """
        super().__init__()
        if gate not in {"none", "vectorial"}:
            raise ValueError("HTGNN gate must be one of: 'none', 'vectorial'.")
        self.gate_type: str = gate
        self.relation_transforms: nn.ModuleDict = nn.ModuleDict(
            {
                relation: nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                for relation in relation_names
            }
        )
        self.update: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim),
            get_activation(name=activation),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )
        self.gate_layer: nn.Linear | None
        if self.gate_type == "vectorial":
            self.gate_layer = nn.Linear(
                in_features=2 * hidden_dim,
                out_features=hidden_dim,
            )
            nn.init.zeros_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, 2.0)
        else:
            self.gate_layer = None
        self.norm: nn.LayerNorm = nn.LayerNorm(normalized_shape=hidden_dim)

    def forward(
        self,
        states: dict[str, torch.Tensor],
        edges: list[tuple[str, str, str]],
    ) -> dict[str, torch.Tensor]:
        """Apply one round of typed message passing.

        Args:
            states: Node hidden states with shape ``(B, H)``.
            edges: Directed edges encoded as ``(source, target, relation)``.

        Returns:
            Updated node hidden states.
        """
        aggregations: dict[str, torch.Tensor] = {
            node_name: torch.zeros_like(input=state)
            for node_name, state in states.items()
        }
        degrees: dict[str, int] = {node_name: 0 for node_name in states}
        for source, target, relation in edges:
            message: torch.Tensor = self.relation_transforms[relation](states[source])
            aggregations[target] = aggregations[target] + message
            degrees[target] += 1
        updated_states: dict[str, torch.Tensor] = {}
        for node_name, state in states.items():
            degree: int = max(degrees[node_name], 1)
            aggregation: torch.Tensor = aggregations[node_name] / degree
            update_input: torch.Tensor = torch.cat(tensors=(state, aggregation), dim=-1)
            delta: torch.Tensor = self.update(update_input)
            if self.gate_layer is not None:
                delta = torch.sigmoid(self.gate_layer(update_input)) * delta
            updated_states[node_name] = self.norm(state + delta)
        return updated_states


class HTGNNModel(PortfolioModule):
    """Temporal heterogeneous graph network for currency allocation."""

    def __init__(self, config: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Create the HTGNN model from dataset metadata.

        Args:
            config: Model configuration.
            metadata: Processed dataset metadata.
        """
        super().__init__(config=config)
        self.observed_node_names: list[str] = list(metadata["node_names"])
        self.generic_node_count: int = int(config.get("generic_node_count", 0))
        self.generic_node_feature_dim: int = int(
            config.get("generic_node_feature_dim", 32)
        )
        self.generic_node_names: list[str] = [
            f"generic_latent_{idx + 1}"
            for idx in range(self.generic_node_count)
        ]
        self.node_names: list[str] = [
            *self.observed_node_names,
            *self.generic_node_names,
        ]
        self.hidden_dim: int = int(config.get("hidden_dim", 96))
        self.gate_type: str = str(config.get("gate", "none"))
        self.node_encoders: nn.ModuleDict = nn.ModuleDict(
            {
                node_name: nn.GRU(
                    input_size=int(metadata["node_input_dims"][node_name]),
                    hidden_size=self.hidden_dim,
                    num_layers=int(config.get("gru_layers", 1)),
                    batch_first=True,
                )
                for node_name in self.observed_node_names
            }
        )
        if self.generic_node_count > 0:
            self.generic_node_features: nn.Parameter | None = nn.Parameter(
                torch.randn(
                    self.generic_node_count,
                    self.generic_node_feature_dim,
                )
            )
            self.generic_encoder: nn.Sequential | None = nn.Sequential(
                nn.Linear(
                    in_features=self.generic_node_feature_dim,
                    out_features=self.hidden_dim,
                ),
                get_activation(name=str(config.get("activation", "gelu"))),
                nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            )
        else:
            self.generic_node_features = None
            self.generic_encoder = None
        self.edges: list[tuple[str, str, str]] = self._build_edges(
            node_names=self.node_names
        )
        relation_names: list[str] = sorted({edge[2] for edge in self.edges})
        self.message_layers: nn.ModuleList = nn.ModuleList(
            [
                HeterogeneousMessagePassingLayer(
                    hidden_dim=self.hidden_dim,
                    relation_names=relation_names,
                    dropout=float(config.get("dropout", 0.0)),
                    activation=str(config.get("activation", "gelu")),
                    gate=self.gate_type,
                )
                for _ in range(int(config.get("message_passing_layers", 2)))
            ]
        )
        self.readout: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            get_activation(name=str(config.get("activation", "gelu"))),
            nn.Dropout(p=float(config.get("dropout", 0.0))),
        )
        self.alpha_head: nn.Linear = nn.Linear(
            in_features=self.hidden_dim,
            out_features=len(metadata["currencies"]),
        )
        self.variance_head: nn.Linear = nn.Linear(
            in_features=self.hidden_dim,
            out_features=1,
        )

    def _build_edges(self, node_names: list[str]) -> list[tuple[str, str, str]]:
        """Build a fixed directed heterogeneous graph.

        Args:
            node_names: Available node names.

        Returns:
            Directed typed edges.
        """
        edges: list[tuple[str, str, str]] = []
        for source in node_names:
            for target in node_names:
                if source == target:
                    continue
                if (
                    source.startswith("generic_latent_")
                    or target.startswith("generic_latent_")
                    or source == "portfolio_signal"
                    or target == "portfolio_signal"
                ):
                    relation: str = classify_relation(source=source, target=target)
                    edges.append((source, target, relation))
        return edges

    def encode_nodes(
        self,
        nodes: dict[str, torch.Tensor],
        generic_node_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode each node's temporal feature window with a GRU.

        Args:
            nodes: Node input tensors with shape ``(B, T, F_node)``.
            generic_node_features: Optional latent generic-node features.

        Returns:
            Node hidden states with shape ``(B, H)``.
        """
        states: dict[str, torch.Tensor] = {}
        for node_name in self.observed_node_names:
            _, hidden = self.node_encoders[node_name](nodes[node_name])
            states[node_name] = hidden[-1]
        if (
            self.generic_node_count > 0
            and self.generic_node_features is not None
            and self.generic_encoder is not None
        ):
            batch_size = next(iter(states.values())).shape[0]
            generic_features = (
                generic_node_features
                if generic_node_features is not None
                else self.generic_node_features
            )
            generic_states = self.generic_encoder(generic_features)
            generic_states = generic_states.unsqueeze(0).expand(batch_size, -1, -1)
            for idx, node_name in enumerate(self.generic_node_names):
                states[node_name] = generic_states[:, idx, :]
        return states

    def forward(self, batch: dict[str, Any]) -> PortfolioPrediction:
        """Predict allocation from heterogeneous temporal node inputs.

        Args:
            batch: Batch containing a ``nodes`` dictionary.

        Returns:
            Predicted allocation weights and variance.
        """
        states: dict[str, torch.Tensor] = self.encode_nodes(
            nodes=batch["nodes"],
            generic_node_features=batch.get("generic_node_features"),
        )
        return self.propagate_states(
            states=states,
            ablated_nodes=batch.get("ablated_nodes", ()),
        )

    def propagate_states(
        self,
        states: dict[str, torch.Tensor],
        ablated_nodes: tuple[str, ...] = (),
    ) -> PortfolioPrediction:
        """Predict allocation from already encoded node states."""
        states = dict(states)
        for node_name in ablated_nodes:
            if node_name in states:
                states[node_name] = torch.zeros_like(states[node_name])
        for layer in self.message_layers:
            states = layer(states=states, edges=self.edges)
        portfolio_state: torch.Tensor = self.readout(states["portfolio_signal"])
        variance: torch.Tensor = positive_variance(
            raw_variance=self.variance_head(portfolio_state)
        )
        return dirichlet_prediction(
            raw_alpha=self.alpha_head(portfolio_state),
            variance=variance,
        )
