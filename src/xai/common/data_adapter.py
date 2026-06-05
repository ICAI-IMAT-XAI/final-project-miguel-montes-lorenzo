"""Dataset adapters for XAI experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ForexPortfolioDataset, move_batch_to_device


@dataclass(frozen=True)
class XAIDataBundle:
    """Dataset and sampled subsets used by the pipeline."""

    dataset: ForexPortfolioDataset
    sampled_dataset: Subset[dict[str, Any]]
    background_dataset: Subset[dict[str, Any]]
    local_dataset: Subset[dict[str, Any]]
    sample_indices: list[int]
    background_indices: list[int]
    local_indices: list[int]


def _bounded_indices(length: int, limit: int) -> list[int]:
    """Return the first ``limit`` indices from a dataset length."""
    return list(range(min(max(int(limit), 0), int(length))))


def load_xai_data(
    processed_dir: str,
    split: str,
    max_samples: int,
    background_samples: int,
    n_local: int,
) -> XAIDataBundle:
    """Load the requested split and deterministic analysis subsets."""
    dataset = ForexPortfolioDataset(processed_dir=processed_dir, split=split)
    if len(dataset) == 0:
        raise RuntimeError(f"The requested split {split!r} is empty.")
    sample_indices = _bounded_indices(len(dataset), max_samples)
    background_indices = _bounded_indices(len(dataset), background_samples)
    local_indices = sample_indices[: min(int(n_local), len(sample_indices))]
    return XAIDataBundle(
        dataset=dataset,
        sampled_dataset=Subset(dataset, sample_indices),
        background_dataset=Subset(dataset, background_indices),
        local_dataset=Subset(dataset, local_indices),
        sample_indices=sample_indices,
        background_indices=background_indices,
        local_indices=local_indices,
    )


def make_loader(dataset: Any, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """Create a simple DataLoader for XAI sweeps."""
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a batch using the repository's dataset helper."""
    return move_batch_to_device(batch=batch, device=device)


def batch_dates(batch: dict[str, Any]) -> list[str]:
    """Return dates from a collated batch as strings."""
    values = batch.get("date", [])
    return [str(value) for value in values]

