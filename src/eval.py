"""Evaluation and plotting entry point for trained FOREX models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ForexPortfolioDataset, move_batch_to_device
from src.metrics import (
    ALL_BACKTEST_METRICS,
    compute_backtest_metrics,
    select_backtest_metrics,
)
from src.models import load_model_from_checkpoint
from src.models.base import PortfolioModule, PortfolioPrediction
from src.utils import ensure_dir, load_yaml, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evaluate a trained FOREX allocation model."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint, or 'latest' for the newest checkpoints/*.pt.",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/eval.yaml",
        help="Path to evaluation YAML config.",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint path, supporting the special value ``latest``.

    Args:
        checkpoint: Explicit checkpoint path or the string ``latest``.

    Returns:
        Resolved checkpoint path.
    """
    checkpoint_value: str = str(checkpoint)
    if checkpoint_value != "latest":
        return Path(checkpoint_value)

    checkpoint_dir: Path = Path("checkpoints")
    candidates: list[Path] = sorted(
        checkpoint_dir.glob(pattern="*.pt"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No checkpoints/*.pt files found; train a model before using "
            "--checkpoint latest."
        )
    return candidates[0]


def collect_predictions(
    model: PortfolioModule,
    dataset: ForexPortfolioDataset,
    device: torch.device,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Collect model predictions and realized next-day returns.

    Args:
        model: Trained portfolio model.
        dataset: Evaluation dataset.
        device: Evaluation device.
        mc_samples: Number of stochastic samples for probabilistic models.

    Returns:
        Tuple with weights, variances, next log returns, and dates.
    """
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,
    )
    model.eval()
    model.to(device=device)
    weights: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    next_log_returns: list[np.ndarray] = []
    dates: list[str] = []
    with torch.no_grad():
        for batch in loader:
            moved_batch: dict[str, Any] = move_batch_to_device(
                batch=batch,
                device=device,
            )
            prediction: PortfolioPrediction = model.predict(
                batch=moved_batch,
                mc_samples=mc_samples,
            )
            weights.append(prediction.weights.detach().cpu().numpy())
            variances.append(prediction.variance.detach().cpu().numpy())
            next_log_returns.append(
                moved_batch["next_log_returns"].detach().cpu().numpy()
            )
            dates.extend([str(value) for value in batch["date"]])
    return (
        np.concatenate(weights, axis=0),
        np.concatenate(variances, axis=0),
        np.concatenate(next_log_returns, axis=0),
        dates,
    )


def compute_value_path(log_returns: np.ndarray, initial_value: float) -> np.ndarray:
    """Convert daily log returns into a portfolio value path.

    Args:
        log_returns: Daily log-return series.
        initial_value: Initial portfolio value.

    Returns:
        Portfolio value path with initial value prepended.
    """
    cumulative: np.ndarray = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(a=log_returns))
    )
    return initial_value * np.exp(cumulative)


def run_backtest(
    weights: np.ndarray,
    next_log_returns: np.ndarray,
    initial_value: float,
) -> dict[str, np.ndarray | int]:
    """Run the allocation backtest and requested baselines.

    Args:
        weights: Model weights with shape ``(T, N)``.
        next_log_returns: Realized currency log returns with shape ``(T, N)``.
        initial_value: Initial value in USD.

    Returns:
        Dictionary with model and baseline value paths.
    """
    model_log_returns: np.ndarray = np.sum(a=weights * next_log_returns, axis=1)
    model_values: np.ndarray = compute_value_path(
        log_returns=model_log_returns,
        initial_value=initial_value,
    )
    usd_values: np.ndarray = np.full(
        shape=model_values.shape,
        fill_value=initial_value,
        dtype=np.float64,
    )
    best_currency_idx: int = int(np.argmax(a=np.sum(a=next_log_returns, axis=0)))
    best_currency_values: np.ndarray = compute_value_path(
        log_returns=next_log_returns[:, best_currency_idx],
        initial_value=initial_value,
    )
    return {
        "model_log_returns": model_log_returns,
        "model_values": model_values,
        "usd_values": usd_values,
        "best_currency_idx": best_currency_idx,
        "best_currency_values": best_currency_values,
    }


def plot_value_paths(
    dates: list[str],
    model_values: np.ndarray,
    usd_values: np.ndarray,
    best_currency_values: np.ndarray,
    best_currency: str,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Plot model portfolio value and the two requested baselines.

    Args:
        dates: Evaluation dates.
        model_values: Model portfolio value path.
        usd_values: Baseline value path for 100 percent USD.
        best_currency_values: Hindsight best-currency value path.
        best_currency: Best currency ticker.
        model_name: Name of evaluated model.
        output_path: Output PNG path.
    """
    date_index: pd.DatetimeIndex = pd.to_datetime(arg=[dates[0], *dates])
    plt.figure(figsize=(11, 6))
    plt.plot(date_index, model_values, label=model_name)
    plt.plot(date_index, usd_values, linestyle="--", label="100% USD")
    plt.plot(
        date_index,
        best_currency_values,
        linestyle="--",
        label=f"100% {best_currency} (best in hindsight)",
    )
    plt.title("Portfolio value in USD")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname=output_path, dpi=160)
    plt.close()


def plot_mean_currency_allocations(
    weights: np.ndarray,
    currencies: list[str],
    output_path: str | Path,
) -> None:
    """Plot mean allocation percentages with variance intervals."""
    mean_weights = weights.mean(axis=0) * 100.0
    variance_weights = weights.var(axis=0) * 100.0
    positions = np.arange(len(currencies))
    plt.figure(figsize=(11, 6))
    plt.bar(positions, mean_weights, color="#4C78A8", alpha=0.9)
    plt.errorbar(
        positions,
        mean_weights,
        yerr=variance_weights,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
    )
    plt.xticks(positions, currencies)
    plt.ylabel("Mean allocation (%)")
    plt.title("Mean currency allocation with variance intervals")
    plt.tight_layout()
    plt.savefig(fname=output_path, format="svg")
    plt.close()


def plot_lagged_allocation_correlation_heatmap(
    weights: np.ndarray,
    currencies: list[str],
    output_path: str | Path,
) -> None:
    """Plot correlation between today's and yesterday's allocations."""
    if weights.shape[0] < 2:
        correlation = np.eye(len(currencies), dtype=np.float64)
    else:
        today = weights[1:]
        previous_day = weights[:-1]
        correlation = np.empty((weights.shape[1], weights.shape[1]), dtype=np.float64)
        for row_idx in range(weights.shape[1]):
            for col_idx in range(weights.shape[1]):
                x = today[:, row_idx]
                y = previous_day[:, col_idx]
                if np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
                    correlation[row_idx, col_idx] = 0.0
                else:
                    correlation[row_idx, col_idx] = float(np.corrcoef(x, y)[0, 1])
        correlation = np.nan_to_num(correlation, nan=0.0)
    plt.figure(figsize=(8.5, 7.0))
    color_norm = SymLogNorm(
        linthresh=0.05,
        linscale=0.25,
        vmin=-1.0,
        vmax=1.0,
        base=10,
    )
    image = plt.imshow(correlation, cmap="coolwarm", norm=color_norm)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Correlation")
    plt.xticks(np.arange(len(currencies)), currencies, rotation=45, ha="right")
    plt.yticks(np.arange(len(currencies)), currencies)
    for row_idx in range(correlation.shape[0]):
        for col_idx in range(correlation.shape[1]):
            value = correlation[row_idx, col_idx]
            text_color = "white" if abs(value) >= 0.5 else "black"
            plt.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    plt.xlabel("Previous-day allocation")
    plt.ylabel("Current-day allocation")
    plt.title("Lag-1 allocation correlation heatmap")
    plt.tight_layout()
    plt.savefig(fname=output_path, format="svg")
    plt.close()


def evaluate_checkpoint(
    checkpoint: str | Path,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one checkpoint and save plots, weights, and metrics.

    Args:
        checkpoint: Model checkpoint path.
        eval_config: Evaluation configuration.

    Returns:
        Dictionary with metrics and output paths.
    """
    checkpoint_path: Path = resolve_checkpoint(checkpoint=checkpoint)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, metadata = load_model_from_checkpoint(
        path=checkpoint_path,
        map_location=device,
    )
    dates_config: dict[str, Any] = eval_config.get("dates", {}) or {}
    split: str = str(eval_config.get("split", "test"))
    dataset: ForexPortfolioDataset = ForexPortfolioDataset(
        processed_dir=str(eval_config.get("processed_dir", "data/processed")),
        split=split,
        date_start=dates_config.get("start"),
        date_end=dates_config.get("end"),
    )
    model_eval_config: dict[str, Any] = model_config.get("eval", {})
    mc_samples: int = int(
        model_eval_config.get(
            "mc_samples",
            model_eval_config.get(
                "n_samples",
                model_config.get("mc_samples", 1),
            ),
        )
    )
    weights, variances, next_log_returns, dates = collect_predictions(
        model=model,
        dataset=dataset,
        device=device,
        mc_samples=mc_samples,
    )
    backtest: dict[str, np.ndarray | int] = run_backtest(
        weights=weights,
        next_log_returns=next_log_returns,
        initial_value=float(eval_config.get("initial_value", 1.0)),
    )
    currencies: list[str] = list(metadata["currencies"])
    best_currency_idx: int = int(backtest["best_currency_idx"])
    metrics: dict[str, float] = compute_backtest_metrics(
        portfolio_values=np.asarray(backtest["model_values"]),
        log_returns=np.asarray(backtest["model_log_returns"]),
        weights=weights,
        trading_days_per_year=int(eval_config.get("trading_days_per_year", 252)),
    )
    metrics = select_backtest_metrics(
        metrics=metrics,
        requested_metrics=eval_config.get("metrics"),
    )
    evaluation_dir: Path = ensure_dir(
        path=eval_config.get("evaluation_dir", "evaluation")
    )
    model_name: str = str(model_config["name"])
    checkpoint_stem: str = checkpoint_path.stem
    eval_timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_eval_dir: Path = ensure_dir(
        path=evaluation_dir / f"{eval_timestamp}-{checkpoint_stem}"
    )
    plot_path: Path = checkpoint_eval_dir / "portfolio_value.png"
    metrics_path: Path = checkpoint_eval_dir / "metrics.json"
    predictions_path: Path = checkpoint_eval_dir / "predictions.csv"
    mean_allocations_svg_path: Path = checkpoint_eval_dir / "mean_currency_allocations.svg"
    lagged_correlation_svg_path: Path = (
        checkpoint_eval_dir / "allocation_lag1_correlation_heatmap.svg"
    )
    plot_value_paths(
        dates=dates,
        model_values=np.asarray(backtest["model_values"]),
        usd_values=np.asarray(backtest["usd_values"]),
        best_currency_values=np.asarray(backtest["best_currency_values"]),
        best_currency=currencies[best_currency_idx],
        model_name=model_name,
        output_path=plot_path,
    )
    plot_mean_currency_allocations(
        weights=weights,
        currencies=currencies,
        output_path=mean_allocations_svg_path,
    )
    plot_lagged_allocation_correlation_heatmap(
        weights=weights,
        currencies=currencies,
        output_path=lagged_correlation_svg_path,
    )
    prediction_frame: pd.DataFrame = pd.DataFrame(data=weights, columns=currencies)
    prediction_frame.insert(loc=0, column="date", value=dates)
    prediction_frame["predicted_variance"] = variances.reshape(-1)
    prediction_frame.to_csv(path_or_buf=predictions_path, index=False)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "split": split if not dates_config else None,
        "dates": {
            "requested_start": dates_config.get("start"),
            "requested_end": dates_config.get("end"),
            "actual_start": dates[0],
            "actual_end": dates[-1],
        },
        "best_hindsight_currency": currencies[best_currency_idx],
        "metrics": metrics,
        "requested_metrics": list(eval_config.get("metrics") or ALL_BACKTEST_METRICS),
        "plot_path": str(plot_path),
        "mean_currency_allocations_svg_path": str(mean_allocations_svg_path),
        "allocation_lag1_correlation_heatmap_svg_path": str(
            lagged_correlation_svg_path
        ),
        "predictions_path": str(predictions_path),
    }
    write_json(data=payload, path=metrics_path)
    return payload


def main() -> None:
    """Evaluate a trained model from command-line arguments."""
    args: argparse.Namespace = parse_args()
    eval_config: dict[str, Any] = load_yaml(path=args.eval_config)
    payload: dict[str, Any] = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        eval_config=eval_config,
    )
    print(payload)


if __name__ == "__main__":
    main()
