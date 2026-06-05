"""Scalar target selection for explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.models.base import PortfolioPrediction


@dataclass(frozen=True)
class TargetSpec:
    """Resolved scalar target requested by the CLI."""

    raw: str
    kind: str
    index: int | None = None
    label: str | None = None


def resolve_target(target: str, metadata: dict[str, Any]) -> TargetSpec:
    """Resolve a CLI target value into a concrete scalar output."""
    currencies = [str(value) for value in metadata.get("currencies", [])]
    if target == "variance":
        return TargetSpec(raw=target, kind="variance", label="variance")
    if target == "allocation_change_norm":
        return TargetSpec(
            raw=target,
            kind="allocation_change_norm",
            label="allocation_change_norm",
        )
    if target.startswith("weights:"):
        selector = target.split(":", 1)[1]
        if selector.isdigit():
            index = int(selector)
        elif selector in currencies:
            index = currencies.index(selector)
        else:
            raise ValueError(
                f"Unknown weight target {selector!r}; available currencies: "
                f"{currencies}."
            )
        if index < 0 or index >= len(currencies):
            raise ValueError(
                f"Weight target index {index} is outside 0..{len(currencies) - 1}."
            )
        label = currencies[index] if currencies else f"weight_{index}"
        return TargetSpec(raw=target, kind="weight", index=index, label=label)
    raise ValueError(
        "--target must be variance, allocation_change_norm, or weights:<asset>."
    )


def default_target_if_needed(
    requested: str,
    prediction: PortfolioPrediction,
) -> str:
    """Fallback away from variance if a model output does not expose it."""
    if requested != "variance":
        return requested
    if getattr(prediction, "variance", None) is None:
        return "allocation_change_norm"
    return requested


def previous_allocation_from_batch(
    batch: dict[str, Any],
    num_assets: int,
) -> torch.Tensor:
    """Infer the previous allocation from inputs, falling back to equal weights."""
    features = batch.get("portfolio_features")
    if torch.is_tensor(features) and features.ndim == 3 and features.shape[-1] >= num_assets:
        previous = features[:, -1, :num_assets]
        previous = previous.clamp_min(0.0)
        total = previous.sum(dim=-1, keepdim=True)
        equal = torch.full_like(previous, fill_value=1.0 / float(num_assets))
        return torch.where(total > 1.0e-8, previous / total.clamp_min(1.0e-8), equal)
    weights = torch.full(
        size=(int(batch["next_log_returns"].shape[0]), num_assets),
        fill_value=1.0 / float(num_assets),
        device=batch["next_log_returns"].device,
        dtype=batch["next_log_returns"].dtype,
    )
    return weights


def scalar_target(
    prediction: PortfolioPrediction,
    batch: dict[str, Any],
    spec: TargetSpec,
) -> torch.Tensor:
    """Return the requested scalar target per sample."""
    if spec.kind == "variance":
        return prediction.variance.reshape(prediction.weights.shape[0], -1)[:, 0]
    if spec.kind == "weight":
        if spec.index is None:
            raise ValueError("Weight target requires a resolved index.")
        return prediction.weights[:, spec.index]
    if spec.kind == "allocation_change_norm":
        previous = previous_allocation_from_batch(
            batch=batch,
            num_assets=int(prediction.weights.shape[-1]),
        )
        return torch.linalg.vector_norm(prediction.weights - previous, ord=2, dim=-1)
    raise KeyError(f"Unsupported target kind: {spec.kind}.")

