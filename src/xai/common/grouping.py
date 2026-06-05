"""Grouped feature definitions for XAI methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.dataset import ForexPortfolioDataset


@dataclass(frozen=True)
class FeatureGroup:
    """A semantically grouped model input block."""

    name: str
    node_name: str
    label: str
    observed: bool = True


def node_input_labels(
    dataset: ForexPortfolioDataset,
    node_name: str,
    feature_count: int,
) -> list[str]:
    """Return readable feature labels for a node."""
    metadata = dataset.metadata
    if node_name == "portfolio_signal":
        currencies = list(metadata.get("currencies", []))
        if len(currencies) == feature_count:
            return [str(value) for value in currencies]
        return [f"portfolio_feature_{idx + 1}" for idx in range(feature_count)]
    symbols = list(metadata.get("node_symbols", {}).get(node_name, []))
    if len(symbols) == feature_count:
        return [str(value) for value in symbols]
    return [f"input_{idx + 1}" for idx in range(feature_count)]


def observed_node_groups(dataset: ForexPortfolioDataset) -> list[FeatureGroup]:
    """Use each observed HTGNN node as one grouped explanation unit."""
    groups: list[FeatureGroup] = []
    for node_name in dataset.metadata.get("node_names", []):
        if node_name in dataset.node_inputs:
            groups.append(
                FeatureGroup(
                    name=node_name,
                    node_name=node_name,
                    label=node_name.replace("_", " "),
                    observed=True,
                )
            )
    return groups


def group_baselines(
    background_batches: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Compute mean baseline tensors per observed node."""
    totals: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for batch in background_batches:
        for node_name, value in batch["nodes"].items():
            node_mean = value.detach().mean(dim=0, keepdim=True)
            if node_name not in totals:
                totals[node_name] = torch.zeros_like(node_mean)
                counts[node_name] = 0
            totals[node_name] = totals[node_name] + node_mean * int(value.shape[0])
            counts[node_name] += int(value.shape[0])
    return {
        node_name: total / float(max(counts[node_name], 1))
        for node_name, total in totals.items()
    }


def summarize_nodes_for_surrogate(
    batch: dict[str, Any],
    groups: list[FeatureGroup],
) -> np.ndarray:
    """Convert grouped node tensors into interpretable tabular features."""
    columns: list[np.ndarray] = []
    for group in groups:
        node = batch["nodes"][group.node_name].detach().cpu()
        flat = node.reshape(node.shape[0], -1).numpy()
        last = node[:, -1, :].reshape(node.shape[0], -1).numpy()
        first = node[:, 0, :].reshape(node.shape[0], -1).numpy()
        columns.extend(
            [
                flat.mean(axis=1),
                flat.std(axis=1),
                np.abs(last).mean(axis=1),
                (last - first).mean(axis=1),
            ]
        )
    if not columns:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(columns, axis=1).astype(np.float64)


def surrogate_feature_names(groups: list[FeatureGroup]) -> list[str]:
    """Return tabular surrogate feature names."""
    names: list[str] = []
    for group in groups:
        names.extend(
            [
                f"{group.name}__mean",
                f"{group.name}__std",
                f"{group.name}__last_abs_mean",
                f"{group.name}__trend_mean",
            ]
        )
    return names


def surrogate_group_slices(groups: list[FeatureGroup]) -> dict[str, slice]:
    """Map each group to its contiguous surrogate feature slice."""
    return {
        group.name: slice(idx * 4, idx * 4 + 4)
        for idx, group in enumerate(groups)
    }

