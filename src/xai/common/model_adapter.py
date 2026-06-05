"""Model loading and prediction adapters for HTGNN/BHTGNN XAI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.eval import resolve_checkpoint
from src.models import load_model_from_checkpoint
from src.models.base import PortfolioModule, PortfolioPrediction


@dataclass(frozen=True)
class LoadedXAIModel:
    """Checkpoint-backed model bundle used by the XAI pipeline."""

    model: PortfolioModule
    config: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint_path: Path
    model_name: str


def choose_device(device_name: str) -> torch.device:
    """Resolve a CLI device value."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def load_xai_model(
    checkpoint: str | Path,
    requested_model: str,
    device: torch.device,
) -> LoadedXAIModel:
    """Load an HTGNN/BHTGNN checkpoint and validate the requested model type."""
    checkpoint_path = resolve_checkpoint(checkpoint=checkpoint)
    model, config, metadata = load_model_from_checkpoint(
        path=checkpoint_path,
        map_location=device,
    )
    model_name = str(config.get("name", model.__class__.__name__.replace("Model", "")))
    if requested_model != "auto" and requested_model != model_name:
        raise ValueError(
            f"Requested --model {requested_model!r}, but checkpoint contains "
            f"{model_name!r}."
        )
    if model_name not in {"HTGNN", "BHTGNN"}:
        raise TypeError(
            "The XAI pipeline currently targets HTGNN/BHTGNN checkpoints; "
            f"got {model_name!r}."
        )
    model.to(device=device)
    model.eval()
    return LoadedXAIModel(
        model=model,
        config=config,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )


def is_bayesian_model(model: PortfolioModule) -> bool:
    """Return whether a model should be sampled with MC dropout."""
    return model.__class__.__name__ == "BHTGNNModel"


def prediction_for_batch(
    model: PortfolioModule,
    batch: dict[str, Any],
    mc_samples: int = 16,
    gradient_mode: bool = False,
) -> PortfolioPrediction:
    """Predict with deterministic handling for HTGNN and MC dropout for BHTGNN."""
    if gradient_mode or not is_bayesian_model(model):
        return model.forward(batch=batch)
    return model.predict(batch=batch, mc_samples=mc_samples)


def graph_node_names(model: PortfolioModule) -> list[str]:
    """Return graph node names exposed by HTGNN-like models."""
    node_names = getattr(model, "node_names", None)
    if node_names is None:
        raise TypeError("Loaded model does not expose HTGNN node_names.")
    return list(node_names)

