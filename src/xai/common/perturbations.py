"""Perturbation helpers for local and evaluation methods."""

from __future__ import annotations

from typing import Any

import torch

from src.xai.common.grouping import FeatureGroup


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Clone tensors in a collated batch while preserving metadata."""
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "nodes":
            output[key] = {
                node_name: node_value.clone()
                for node_name, node_value in value.items()
            }
        elif torch.is_tensor(value):
            output[key] = value.clone()
        else:
            output[key] = value
    return output


def replace_node_with_baseline(
    batch: dict[str, Any],
    group: FeatureGroup,
    baselines: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Return a copy with one node replaced by its background mean."""
    perturbed = clone_batch(batch=batch)
    if group.node_name in perturbed["nodes"] and group.node_name in baselines:
        baseline = baselines[group.node_name].to(
            device=perturbed["nodes"][group.node_name].device,
            dtype=perturbed["nodes"][group.node_name].dtype,
        )
        perturbed["nodes"][group.node_name] = baseline.expand_as(
            perturbed["nodes"][group.node_name]
        ).clone()
    return perturbed


def temporal_window_occlusion(
    batch: dict[str, Any],
    baselines: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Replace a time window for every observed node with background values."""
    perturbed = clone_batch(batch=batch)
    for node_name, node_value in perturbed["nodes"].items():
        if node_name not in baselines:
            continue
        baseline = baselines[node_name].to(device=node_value.device, dtype=node_value.dtype)
        perturbed["nodes"][node_name][:, start:end, :] = baseline[:, start:end, :]
    return perturbed


def deletion_batch(
    batch: dict[str, Any],
    ordered_groups: list[FeatureGroup],
    baselines: dict[str, torch.Tensor],
    keep_top_n: int | None,
) -> dict[str, Any]:
    """Create deletion/insertion perturbations from an ordered group list."""
    perturbed = clone_batch(batch=batch)
    if keep_top_n is None:
        groups_to_replace = ordered_groups
    else:
        groups_to_replace = ordered_groups[keep_top_n:]
    for group in groups_to_replace:
        if group.node_name not in perturbed["nodes"] or group.node_name not in baselines:
            continue
        baseline = baselines[group.node_name].to(
            device=perturbed["nodes"][group.node_name].device,
            dtype=perturbed["nodes"][group.node_name].dtype,
        )
        perturbed["nodes"][group.node_name] = baseline.expand_as(
            perturbed["nodes"][group.node_name]
        ).clone()
    return perturbed

