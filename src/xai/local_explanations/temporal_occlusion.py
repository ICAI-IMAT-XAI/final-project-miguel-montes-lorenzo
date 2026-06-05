"""Temporal window occlusion local explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.xai.common.data_adapter import make_loader, move_to_device
from src.xai.common.grouping import FeatureGroup, group_baselines, observed_node_groups
from src.xai.common.io import manifest_entry
from src.xai.common.model_adapter import prediction_for_batch
from src.xai.common.performance import (
    equal_weight_log_returns,
    model_log_returns,
    performance_ratio,
)
from src.xai.common.perturbations import clone_batch, temporal_window_occlusion
from src.xai.common.plotting import save_bar_plot
from src.xai.common.targets import TargetSpec


def _window_bounds(lookback: int, windows: int) -> list[tuple[int, int]]:
    """Split a lookback length into deterministic windows."""
    windows = max(1, min(int(windows), int(lookback)))
    bounds = []
    for idx in range(windows):
        start = round(idx * lookback / windows)
        end = round((idx + 1) * lookback / windows)
        bounds.append((int(start), int(max(end, start + 1))))
    return bounds


def run_temporal_occlusion(
    model: torch.nn.Module,
    local_dataset: Any,
    background_dataset: Any,
    target_spec: TargetSpec,
    device: torch.device,
    output_dir: str | Path,
    windows: int = 5,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, str]]]:
    """Measure performance change after occluding temporal windows."""
    del target_spec
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_dataset = local_dataset.dataset if hasattr(local_dataset, "dataset") else local_dataset
    if hasattr(base_dataset, "dataset"):
        base_dataset = base_dataset.dataset
    groups = observed_node_groups(dataset=base_dataset)
    background_batches = [
        move_to_device(batch=batch, device=device)
        for batch in make_loader(background_dataset, batch_size=64, shuffle=False)
    ]
    baselines = group_baselines(background_batches=background_batches)
    rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    loader = make_loader(local_dataset, batch_size=max(len(local_dataset), 1), shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            moved = move_to_device(batch=batch, device=device)
            prediction = prediction_for_batch(model=model, batch=moved)
            standard_returns = equal_weight_log_returns(batch=moved)
            base_returns = model_log_returns(
                prediction=prediction,
                batch=moved,
            )
            base_metrics = performance_ratio(
                model_returns=base_returns,
                standard_returns=standard_returns,
            )
            lookback = int(next(iter(moved["nodes"].values())).shape[1])
            dates = [str(value) for value in batch["date"]]
            for start, end in _window_bounds(lookback=lookback, windows=windows):
                perturbed = temporal_window_occlusion(
                    batch=moved,
                    baselines=baselines,
                    start=start,
                    end=end,
                )
                perturbed_prediction = prediction_for_batch(model=model, batch=perturbed)
                perturbed_returns = model_log_returns(
                    prediction=perturbed_prediction,
                    batch=perturbed,
                )
                perturbed_metrics = performance_ratio(
                    model_returns=perturbed_returns,
                    standard_returns=standard_returns,
                )
                performance_drop = (
                    base_metrics["performance_ratio"]
                    - perturbed_metrics["performance_ratio"]
                )
                rows.append(
                    {
                        "batch_index": batch_idx,
                        "date_start": dates[0],
                        "date_end": dates[-1],
                        "window_start": start,
                        "window_end": end,
                        "base_performance_ratio": base_metrics["performance_ratio"],
                        "performance_ratio": perturbed_metrics["performance_ratio"],
                        "performance_ratio_drop": performance_drop,
                        "abs_performance_ratio_drop": abs(performance_drop),
                        "model_annualized_return": perturbed_metrics[
                            "model_annualized_return"
                        ],
                        "standard_annualized_return": perturbed_metrics[
                            "standard_annualized_return"
                        ],
                    }
                )
                for group in groups:
                    node_perturbed = _node_temporal_window_occlusion(
                        batch=moved,
                        group=group,
                        baselines=baselines,
                        start=start,
                        end=end,
                    )
                    node_prediction = prediction_for_batch(
                        model=model,
                        batch=node_perturbed,
                    )
                    node_returns = model_log_returns(
                        prediction=node_prediction,
                        batch=node_perturbed,
                    )
                    node_metrics = performance_ratio(
                        model_returns=node_returns,
                        standard_returns=standard_returns,
                    )
                    node_performance_drop = (
                        base_metrics["performance_ratio"]
                        - node_metrics["performance_ratio"]
                    )
                    node_rows.append(
                        {
                            "batch_index": batch_idx,
                            "date_start": dates[0],
                            "date_end": dates[-1],
                            "group": group.name,
                            "window_start": start,
                            "window_end": end,
                            "base_performance_ratio": base_metrics[
                                "performance_ratio"
                            ],
                            "performance_ratio": node_metrics["performance_ratio"],
                            "performance_ratio_drop": node_performance_drop,
                            "abs_performance_ratio_drop": abs(node_performance_drop),
                            "model_annualized_return": node_metrics[
                                "model_annualized_return"
                            ],
                            "standard_annualized_return": node_metrics[
                                "standard_annualized_return"
                            ],
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["window_start", "window_end"], as_index=False)
        .agg(
            mean_performance_ratio_drop=("performance_ratio_drop", "mean"),
            mean_abs_performance_ratio_drop=("abs_performance_ratio_drop", "mean"),
            mean_performance_ratio=("performance_ratio", "mean"),
        )
        .sort_values("window_start")
    )
    scores = {
        f"{int(row['window_start'])}:{int(row['window_end'])}": float(row["mean_performance_ratio_drop"])
        for _, row in summary.iterrows()
    }
    node_frame = pd.DataFrame(node_rows)
    node_summary = (
        node_frame.groupby(["group", "window_start", "window_end"], as_index=False)
        .agg(
            mean_performance_ratio_drop=("performance_ratio_drop", "mean"),
            mean_abs_performance_ratio_drop=("abs_performance_ratio_drop", "mean"),
            mean_performance_ratio=("performance_ratio", "mean"),
        )
        .sort_values(["group", "window_start"])
    )
    node_scores = (
        node_summary.groupby("group")["mean_performance_ratio_drop"].mean().to_dict()
    )
    sample_path = output / "temporal_occlusion_local_values.csv"
    summary_path = output / "temporal_occlusion_window_summary.csv"
    node_sample_path = output / "node_temporal_occlusion_local_values.csv"
    node_summary_path = output / "node_temporal_occlusion_window_summary.csv"
    figure_path = output / "temporal_occlusion_windows.svg"
    node_figure_path = output / "node_temporal_occlusion_windows.svg"
    frame.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    node_frame.to_csv(node_sample_path, index=False)
    node_summary.to_csv(node_summary_path, index=False)
    save_bar_plot(
        values=scores,
        output_path=figure_path,
        title="Temporal occlusion window effect",
        ylabel="Mean performance ratio drop",
        width=6.0,
    )
    _save_node_window_histograms(
        summary=node_summary,
        output_path=node_figure_path,
        title="Node-wise temporal occlusion window effect",
    )
    artifacts = [
        manifest_entry(sample_path, "table", "temporal_occlusion"),
        manifest_entry(summary_path, "table", "temporal_occlusion"),
        manifest_entry(figure_path, "figure", "temporal_occlusion"),
        manifest_entry(node_sample_path, "table", "node_temporal_occlusion"),
        manifest_entry(node_summary_path, "table", "node_temporal_occlusion"),
        manifest_entry(node_figure_path, "figure", "node_temporal_occlusion"),
    ]
    return scores, {str(key): float(value) for key, value in node_scores.items()}, artifacts


def _node_temporal_window_occlusion(
    batch: dict[str, Any],
    group: FeatureGroup,
    baselines: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Occlude one temporal window for one observed node."""
    perturbed = clone_batch(batch=batch)
    if group.node_name not in perturbed["nodes"] or group.node_name not in baselines:
        return perturbed
    node = perturbed["nodes"][group.node_name]
    baseline = baselines[group.node_name].to(device=node.device, dtype=node.dtype)
    perturbed["nodes"][group.node_name][:, start:end, :] = baseline[:, start:end, :]
    return perturbed


def _save_node_window_histograms(
    summary: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save vertical mini-histograms, one per node, for window occlusion."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    node_names = summary["group"].drop_duplicates().tolist()
    if not node_names:
        return output
    cols = 3
    rows = int(np.ceil(len(node_names) / cols))
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(4.2 * cols, 3.2 * rows),
        squeeze=False,
    )
    for axis_idx, node_name in enumerate(node_names):
        ax = axes[axis_idx // cols][axis_idx % cols]
        node_frame = summary[summary["group"] == node_name].sort_values("window_start")
        labels = [
            f"{int(row.window_start)}-{int(row.window_end)}"
            for row in node_frame.itertuples(index=False)
        ]
        values = node_frame["mean_performance_ratio_drop"].to_numpy(dtype=float)
        ax.bar(np.arange(len(labels)), values, color="#4C78A8", edgecolor="#1e293b")
        ax.axhline(0.0, color="#0f172a", linewidth=0.7)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(str(node_name).replace("_", " "), fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    for axis_idx in range(len(node_names), rows * cols):
        axes[axis_idx // cols][axis_idx % cols].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output
