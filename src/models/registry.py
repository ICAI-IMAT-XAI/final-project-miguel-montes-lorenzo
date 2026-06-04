"""Model registry and checkpoint loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.base import PortfolioModule
from src.models.pointwise.HTGNN import HTGNNModel
from src.models.pointwise.NN import NNModel
from src.models.probabilistic.BHTGNN import BHTGNNModel
from src.models.probabilistic.BNN import BNNModel

MODEL_REGISTRY: dict[str, type[PortfolioModule]] = {
    "NN": NNModel,
    "HTGNN": HTGNNModel,
    "BNN": BNNModel,
    "BHTGNN": BHTGNNModel,
}


def build_model(config: dict[str, Any], metadata: dict[str, Any]) -> PortfolioModule:
    """Instantiate a portfolio model from its config.

    Args:
        config: Model configuration containing ``name``.
        metadata: Processed dataset metadata.

    Returns:
        Instantiated portfolio module.
    """
    name: str = str(config["name"])
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](config=config, metadata=metadata)


def save_model_checkpoint(
    model: PortfolioModule,
    path: str | Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Save a model checkpoint with config and metadata.

    Args:
        model: Trained model.
        path: Output checkpoint path.
        config: Model configuration.
        metadata: Dataset metadata.
    """
    checkpoint_path: Path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": config["name"],
        "config": config,
        "metadata": metadata,
        "state_dict": model.state_dict(),
    }
    torch.save(obj=payload, f=checkpoint_path)


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[PortfolioModule, dict[str, Any], dict[str, Any]]:
    """Load a model, its config, and dataset metadata from a checkpoint.

    Args:
        path: Checkpoint path.
        map_location: Torch map location.

    Returns:
        Tuple containing model, config, and metadata.
    """
    checkpoint: dict[str, Any] = torch.load(f=path, map_location=map_location)
    config: dict[str, Any] = checkpoint["config"]
    metadata: dict[str, Any] = checkpoint["metadata"]
    model: PortfolioModule = build_model(config=config, metadata=metadata)
    model.load_state_dict(state_dict=checkpoint["state_dict"])
    return model, config, metadata
