"""Visualize processed inputs and mean-variance target allocations."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch

from src.models.base import build_mean_variance_targets
from src.utils import ensure_dir, read_json


def compute_target_allocations(
    processed_dir: str | Path = "data/processed",
) -> tuple[np.ndarray, list[str]]:
    """Rebuild MV teacher allocations from processed arrays."""
    processed_path = Path(processed_dir)
    metadata = read_json(processed_path / "metadata.json")
    portfolio_raw_returns = np.load(processed_path / "portfolio_raw_returns.npy")
    next_log_returns = np.load(processed_path / "next_log_returns.npy")

    target_weights, _ = build_mean_variance_targets(
        next_log_returns=torch.as_tensor(next_log_returns, dtype=torch.float32),
        returns_window=torch.as_tensor(portfolio_raw_returns, dtype=torch.float32),
        risk_aversion=1.0,
        ridge=1.0e-4,
        allow_short=False,
    )
    return target_weights.cpu().numpy(), list(metadata["currencies"])


def plot_target_allocation_histogram(
    target_weights: np.ndarray,
    currencies: list[str],
    output_path: str | Path,
) -> None:
    """Plot average target allocation with variance and std intervals."""
    mean_alloc = target_weights.mean(axis=0) * 100.0
    variance_alloc = target_weights.var(axis=0) * 100.0
    x = np.arange(len(currencies))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x,
        mean_alloc,
        color="#4C78A8",
        edgecolor="#1e293b",
        linewidth=0.8,
        width=0.72,
        label="Mean target allocation",
    )
    ax.errorbar(
        x,
        mean_alloc,
        yerr=variance_alloc,
        fmt="none",
        ecolor="#f59e0b",
        elinewidth=1.8,
        capsize=4,
        label="Variance interval",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(currencies, fontsize=10)
    ax.set_ylabel("Allocation (%)")
    ax.set_title("Mean-Variance Target Allocations", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def compute_lag1_correlation_matrix(weights: np.ndarray) -> np.ndarray:
    """Compute Corr(weight_t[i], weight_{t-1}[j]) for all currency pairs."""
    current = weights[1:]
    previous = weights[:-1]
    num_assets = weights.shape[1]
    matrix = np.zeros((num_assets, num_assets), dtype=np.float64)
    for row_idx in range(num_assets):
        for col_idx in range(num_assets):
            x = current[:, row_idx]
            y = previous[:, col_idx]
            if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
                matrix[row_idx, col_idx] = 0.0
            else:
                matrix[row_idx, col_idx] = float(np.corrcoef(x, y)[0, 1])
    return matrix


def plot_lag1_heatmap(
    weights: np.ndarray,
    currencies: list[str],
    output_path: str | Path,
) -> None:
    """Plot a lag-1 target allocation correlation heatmap with annotations."""
    matrix = compute_lag1_correlation_matrix(weights=weights)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    color_norm = SymLogNorm(
        linthresh=0.05,
        linscale=0.25,
        vmin=-1.0,
        vmax=1.0,
        base=10,
    )
    image = ax.imshow(matrix, cmap="coolwarm", norm=color_norm)
    ax.set_xticks(np.arange(len(currencies)))
    ax.set_yticks(np.arange(len(currencies)))
    ax.set_xticklabels(currencies)
    ax.set_yticklabels(currencies)
    ax.set_xlabel("Allocation at t-1")
    ax.set_ylabel("Allocation at t")
    ax.set_title(
        "Lag-1 Correlation of MV Target Allocations",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if abs(value) > 0.45 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create processed-data visualizations under visualizations/inputs."""
    output_dir = ensure_dir(Path("visualizations") / "inputs")
    target_weights, currencies = compute_target_allocations()
    plot_target_allocation_histogram(
        target_weights=target_weights,
        currencies=currencies,
        output_path=output_dir / "target_allocation_histogram.svg",
    )
    plot_lag1_heatmap(
        weights=target_weights,
        currencies=currencies,
        output_path=output_dir / "target_allocation_lag1_correlation_heatmap.svg",
    )
    print(
        {
            "output_dir": str(output_dir),
            "currencies": currencies,
            "num_samples": int(target_weights.shape[0]),
        }
    )


if __name__ == "__main__":
    main()
