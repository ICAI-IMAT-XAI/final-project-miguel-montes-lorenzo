"""Training entry point for FOREX allocation models."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ForexPortfolioDataset, move_batch_to_device
from src.models import build_model
from src.models.base import PortfolioModule, portfolio_loss
from src.models.registry import save_model_checkpoint
from src.utils import load_yaml, merge_dicts, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train a FOREX allocation model."
    )
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="Path to global training YAML config.",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/eval.yaml",
        help="Path to evaluation YAML config.",
    )
    return parser.parse_args()


def config_bool(value: Any, default: bool = False) -> bool:
    """Convert a configuration value into a boolean.

    Args:
        value: Raw configuration value.
        default: Fallback value used when the input is missing or invalid.

    Returns:
        Boolean interpretation of the configuration value.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "on"}:
            return True
        if normalized in {"false", "no", "n", "0", "off"}:
            return False
    return default


def choose_device(device_name: str) -> torch.device:
    """Resolve the requested training device.

    Args:
        device_name: ``auto``, ``cpu``, ``cuda``, or another torch device string.

    Returns:
        Resolved PyTorch device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_train_date_ranges(
    train_config: dict[str, Any],
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Resolve train and validation date windows from configuration.

    Args:
        train_config: Effective training configuration.

    Returns:
        Optional ``(start, end)`` tuples for train and validation datasets.
    """
    dates_config: dict[str, Any] = dict(train_config.get("dates") or {})
    train_range: Any = dates_config.get("train")
    validation_range: Any = dates_config.get("validation")

    if train_range is None and validation_range is None:
        start = dates_config.get("start")
        end = dates_config.get("end")
        if start is not None and end is not None:
            return (str(start), str(end)), None
        return None, None

    def _normalize_range(name: str, value: Any) -> tuple[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"train.yaml dates.{name} must be a two-item list like "
                f"[\"YYYY-MM-DD\", \"YYYY-MM-DD\"]."
            )
        start_value, end_value = value
        return str(start_value), str(end_value)

    return _normalize_range("train", train_range), _normalize_range(
        "validation",
        validation_range,
    )


def run_epoch(
    model: PortfolioModule,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    weights_loss_coef: float,
    variance_loss_coef: float,
    loss_name: str,
    loss_regularizations: dict[str, Any],
    n_data: int,
    n_mc_samples: int,
    risk_aversion: float,
    ridge: float,
    allow_short: bool,
) -> float:
    """Run one train or validation epoch.

    Args:
        model: Portfolio model.
        loader: Data loader for a split.
        device: Device used for tensors.
        optimizer: Optimizer for training, or ``None`` for validation.
        weights_loss_coef: Weight-target loss coefficient.
        variance_loss_coef: Variance-target loss coefficient.
        loss_name: Loss function name.
        loss_regularizations: Extra loss regularization parameters.
        n_data: Number of training samples.
        n_mc_samples: Number of Monte Carlo samples.

    Returns:
        Average epoch loss.
    """
    is_training: bool = optimizer is not None
    model.train(mode=is_training)

    total_loss: float = 0.0
    total_samples: int = 0

    for batch in loader:
        moved_batch: dict[str, Any] = move_batch_to_device(
            batch=batch,
            device=device,
        )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        loss: torch.Tensor = torch.zeros((), device=device)

        for _ in range(max(n_mc_samples, 1)):
            prediction = model.forward(batch=moved_batch)
            loss = loss + portfolio_loss(
                prediction=prediction,
                batch=moved_batch,
                weights_loss_coef=weights_loss_coef,
                variance_loss_coef=variance_loss_coef,
                loss_name=loss_name,
                loss_regularizations=loss_regularizations,
                extra_kl=getattr(model, "last_kl", None),
                n_data=n_data,
                risk_aversion=risk_aversion,
                ridge=ridge,
                allow_short=allow_short,
            )

        loss = loss / max(n_mc_samples, 1)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

        batch_size: int = int(moved_batch["next_log_returns"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def copy_run_configs(
    run_dir: Path,
    model_config_path: str | Path | None,
    train_config_path: str | Path | None,
) -> None:
    """Copy the raw config files used for a run into its artifact directory."""
    run_dir.mkdir(parents=True, exist_ok=True)

    for config_path in (train_config_path, model_config_path):
        if config_path is None:
            continue
        source: Path = Path(config_path)
        if not source.exists():
            continue
        shutil.copy2(src=source, dst=run_dir / source.name)


def train_model(
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    model_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
) -> Path:
    """Train a model and save the configured checkpoint state.

    Args:
        model_config: Model-specific configuration.
        train_config: Global training configuration.
        model_config_path: Path to the model YAML used for this run.
        train_config_path: Path to the train YAML used for this run.

    Returns:
        Path to the saved checkpoint.
    """
    effective_train_config: dict[str, Any] = merge_dicts(
        train_config,
        model_config.get("train", {}),
    )
    checkpoint_policy: str = str(effective_train_config.get("checkpoint", "best"))
    if checkpoint_policy not in {"best", "last"}:
        raise ValueError("train.checkpoint must be one of: 'best', 'last'.")
    model_build_config: dict[str, Any] = merge_dicts(
        {
            key: value
            for key, value in model_config.items()
            if key not in {"train", "eval"}
        },
        model_config.get("train", {}),
    )

    seed_everything(seed=int(effective_train_config.get("seed", 7)))

    processed_dir: str = str(
        effective_train_config.get("processed_dir", "data/processed")
    )
    train_dates, validation_dates = resolve_train_date_ranges(
        train_config=effective_train_config
    )

    if train_dates is not None:
        train_dataset = ForexPortfolioDataset(
            processed_dir=processed_dir,
            split="all",
            date_start=train_dates[0],
            date_end=train_dates[1],
        )
    else:
        train_dataset = ForexPortfolioDataset(
            processed_dir=processed_dir,
            split="train",
        )
    if validation_dates is not None:
        val_dataset = ForexPortfolioDataset(
            processed_dir=processed_dir,
            split="all",
            date_start=validation_dates[0],
            date_end=validation_dates[1],
        )
    else:
        val_dataset = ForexPortfolioDataset(
            processed_dir=processed_dir,
            split="val",
        )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "The training split is empty. Run data transformation first."
        )

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem: str = f"{model_build_config['name']}-{timestamp}"
    runs_dir: Path = Path(effective_train_config.get("runs_dir", "runs"))
    run_dir: Path = runs_dir / run_stem
    copy_run_configs(
        run_dir=run_dir,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
    )

    enable_tensorboard: bool = config_bool(
        effective_train_config.get("tensorboard"),
        default=True,
    )
    writer = None
    if enable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled, but the 'tensorboard' package "
                "is not installed in the active environment. Install it with "
                "'pip install -r requirements.txt' and rerun training."
            ) from exc
        writer = SummaryWriter(log_dir=str(run_dir))

    device_name: str = str(effective_train_config.get("device", "auto"))
    device: torch.device = choose_device(device_name=device_name)

    model: PortfolioModule = build_model(
        config=model_build_config,
        metadata=train_dataset.metadata,
    ).to(device=device)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=float(
            effective_train_config.get(
                "lr",
                effective_train_config.get("learning_rate", 1.0e-3),
            )
        ),
        weight_decay=float(effective_train_config.get("weight_decay", 1.0e-4)),
    )

    batch_size: int = int(effective_train_config.get("batch_size", 64))
    num_workers: int = int(effective_train_config.get("num_workers", 0))

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    best_val_loss: float = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0

    patience: int = int(
        effective_train_config.get(
            "early_stopping_patience",
            effective_train_config.get("patience", 12),
        )
    )
    epochs: int = int(effective_train_config.get("epochs", 80))
    loss_name: str = str(effective_train_config.get("loss", "mse"))
    loss_regularizations: dict[str, Any] = dict(
        effective_train_config.get("loss_regularizations") or {}
    )

    if "weight" not in loss_regularizations:
        default_kl_weight: float = 1.0 if hasattr(model, "last_kl") else 0.0
        loss_regularizations["weight"] = float(
            effective_train_config.get("kl_weight", default_kl_weight)
        )
    elif loss_regularizations["weight"] is None and hasattr(model, "last_kl"):
        loss_regularizations["weight"] = 1.0

    progress = tqdm(iterable=range(epochs), desc=model_build_config["name"])

    for epoch_index, _ in enumerate(progress, start=1):
        train_loss: float = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            weights_loss_coef=float(
                effective_train_config.get("weights_loss_coef", 1.0)
            ),
            variance_loss_coef=float(
                effective_train_config.get("variance_loss_coef", 0.05)
            ),
            loss_name=loss_name,
            loss_regularizations=loss_regularizations,
            n_data=len(train_dataset),
            n_mc_samples=int(effective_train_config.get("n_mc_samples", 1)),
            risk_aversion=float(effective_train_config.get("risk_aversion", 1.0)),
            ridge=float(effective_train_config.get("ridge", 1.0e-4)),
            allow_short=bool(effective_train_config.get("allow_short", False)),
        )

        if len(val_dataset) > 0:
            with torch.no_grad():
                val_loss: float = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    weights_loss_coef=float(
                        effective_train_config.get("weights_loss_coef", 1.0)
                    ),
                    variance_loss_coef=float(
                        effective_train_config.get("variance_loss_coef", 0.05)
                    ),
                    loss_name=loss_name,
                    loss_regularizations=loss_regularizations,
                    n_data=len(train_dataset),
                    n_mc_samples=int(effective_train_config.get("n_mc_samples", 1)),
                    risk_aversion=float(
                        effective_train_config.get("risk_aversion", 1.0)
                    ),
                    ridge=float(effective_train_config.get("ridge", 1.0e-4)),
                    allow_short=bool(effective_train_config.get("allow_short", False)),
                )
        else:
            val_loss = train_loss

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch_index)
            writer.add_scalar("loss/validation", val_loss, epoch_index)
            writer.add_scalar(
                "loss/best_validation",
                min(best_val_loss, val_loss),
                epoch_index,
            )
            writer.add_scalar(
                "optimizer/learning_rate",
                float(optimizer.param_groups[0]["lr"]),
                epoch_index,
            )

        progress.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_loss=f"{val_loss:.5f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if checkpoint_policy == "best" and best_state is not None:
        model.load_state_dict(state_dict=best_state)

    checkpoint_dir: Path = Path(
        effective_train_config.get("checkpoint_dir", "checkpoints")
    )
    checkpoint_path: Path = checkpoint_dir / f"{run_stem}.pt"

    merged_config: dict[str, Any] = merge_dicts(
        model_build_config,
        {"train": effective_train_config},
        {"eval": model_config.get("eval", {})},
    )

    save_model_checkpoint(
        model=model.cpu(),
        path=checkpoint_path,
        config=merged_config,
        metadata=train_dataset.metadata,
    )

    if writer is not None:
        writer.add_text("artifacts/checkpoint", str(checkpoint_path))
        writer.add_text("artifacts/checkpoint_policy", checkpoint_policy)
        writer.flush()
        writer.close()

    return checkpoint_path


def should_run_eval_after_train(
    model_config: dict[str, Any],
    train_config: dict[str, Any],
) -> bool:
    """Check whether evaluation should run after training.

    Args:
        model_config: Model-specific configuration.
        train_config: Global training configuration.

    Returns:
        Whether post-training evaluation is enabled.
    """
    effective_train_config: dict[str, Any] = merge_dicts(
        train_config,
        model_config.get("train", {}),
    )
    return config_bool(
        value=effective_train_config.get("eval_after_train"),
        default=False,
    )


def run_eval_after_train(checkpoint_path: Path, eval_config_path: str | Path) -> None:
    """Evaluate a checkpoint with the configured evaluation YAML.

    Args:
        checkpoint_path: Path to the trained checkpoint.
        eval_config_path: Path to the evaluation YAML file.
    """
    from src.eval import evaluate_checkpoint

    eval_config: dict[str, Any] = load_yaml(path=eval_config_path)
    payload: dict[str, Any] = evaluate_checkpoint(
        checkpoint=checkpoint_path,
        eval_config=eval_config,
    )
    print(f"Evaluation completed: {payload}")


def main() -> None:
    """Train a model from command-line configuration files."""
    args: argparse.Namespace = parse_args()

    model_config: dict[str, Any] = load_yaml(path=args.config)
    train_config: dict[str, Any] = load_yaml(path=args.train_config)

    checkpoint_path: Path = train_model(
        model_config=model_config,
        train_config=train_config,
        model_config_path=args.config,
        train_config_path=args.train_config,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    if should_run_eval_after_train(
        model_config=model_config,
        train_config=train_config,
    ):
        run_eval_after_train(
            checkpoint_path=checkpoint_path,
            eval_config_path=args.eval_config,
        )


if __name__ == "__main__":
    main()
