"""Integrated Gradients local explanations for HTGNN/BHTGNN inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.analysis.common import plot_focus_graph, plot_focus_histogram
from src.xai.common.data_adapter import make_loader, move_to_device
from src.xai.common.grouping import (
    group_baselines,
    node_input_labels,
    observed_node_groups,
)
from src.xai.common.io import manifest_entry
from src.xai.common.model_adapter import prediction_for_batch
from src.xai.common.targets import TargetSpec


def _background_baselines(
    background_dataset: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect background mean baselines for observed nodes."""
    batches = [
        move_to_device(batch=batch, device=device)
        for batch in make_loader(background_dataset, batch_size=64, shuffle=False)
    ]
    return group_baselines(background_batches=batches)


def run_integrated_gradients(
    model: torch.nn.Module,
    local_dataset: Any,
    background_dataset: Any,
    full_dataset: Any,
    target_spec: TargetSpec,
    device: torch.device,
    output_dir: str | Path,
    steps: int = 24,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, str]]]:
    """Compute local Integrated Gradients for selected instances."""
    del target_spec
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_dataset = full_dataset.dataset if hasattr(full_dataset, "dataset") else full_dataset
    groups = observed_node_groups(dataset=base_dataset)
    graph_node_names = list(getattr(model, "node_names", [group.name for group in groups]))
    baselines = _background_baselines(background_dataset=background_dataset, device=device)
    loader = make_loader(local_dataset, batch_size=max(len(local_dataset), 1), shuffle=False)
    rows: list[dict[str, Any]] = []
    feature_totals: dict[str, np.ndarray] = {}
    node_totals = {group.name: 0.0 for group in groups}
    node_counts = {group.name: 0 for group in groups}

    model.eval()
    for batch in loader:
        moved = move_to_device(batch=batch, device=device)
        base_nodes = {
            name: value.detach()
            for name, value in moved["nodes"].items()
        }
        baseline_nodes = {
            name: baselines[name].to(device=value.device, dtype=value.dtype).expand_as(value)
            for name, value in base_nodes.items()
            if name in baselines
        }
        gradient_sums = {
            name: torch.zeros_like(value)
            for name, value in base_nodes.items()
            if name in baseline_nodes
        }
        for step_idx in range(1, max(int(steps), 1) + 1):
            with torch.no_grad():
                direction = prediction_for_batch(
                    model=model,
                    batch={**moved, "nodes": base_nodes},
                    gradient_mode=True,
                ).weights.detach()
            alpha = float(step_idx) / float(max(int(steps), 1))
            scaled_nodes = {}
            for name, value in base_nodes.items():
                if name in baseline_nodes:
                    scaled = baseline_nodes[name] + alpha * (value - baseline_nodes[name])
                    scaled_nodes[name] = scaled.detach().requires_grad_(True)
                else:
                    scaled_nodes[name] = value.detach()
            prediction = prediction_for_batch(
                model=model,
                batch={**moved, "nodes": scaled_nodes},
                gradient_mode=True,
            )
            score = (prediction.weights * direction).sum(dim=-1).sum()
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=[scaled_nodes[name] for name in gradient_sums],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            for name, gradient in zip(gradient_sums, gradients, strict=True):
                gradient_sums[name] = gradient_sums[name] + gradient

        dates = [str(value) for value in batch["date"]]
        for group in groups:
            if group.node_name not in gradient_sums:
                continue
            avg_gradient = gradient_sums[group.node_name] / float(max(int(steps), 1))
            attribution = (base_nodes[group.node_name] - baseline_nodes[group.node_name]) * avg_gradient
            sample_scores = attribution.detach().abs().sum(dim=(1, 2)).cpu().numpy()
            signed_scores = attribution.detach().sum(dim=(1, 2)).cpu().numpy()
            feature_scores = attribution.detach().abs().sum(dim=1).cpu().numpy()
            if group.name not in feature_totals:
                feature_totals[group.name] = np.zeros(feature_scores.shape[1], dtype=np.float64)
            feature_totals[group.name] += feature_scores.sum(axis=0)
            for sample_idx, (date, abs_value, signed_value) in enumerate(
                zip(dates, sample_scores, signed_scores, strict=True)
            ):
                rows.append(
                    {
                        "local_index": sample_idx,
                        "date": date,
                        "group": group.name,
                        "signed_ig": float(signed_value),
                        "abs_ig": float(abs_value),
                    }
                )
                node_totals[group.name] += float(abs_value)
                node_counts[group.name] += 1

    frame = pd.DataFrame(rows)
    summary_frame = (
        frame.groupby("group", as_index=False)
        .agg(mean_signed_ig=("signed_ig", "mean"), mean_abs_ig=("abs_ig", "mean"))
        .sort_values("mean_abs_ig", ascending=False)
    )
    pivot = frame.pivot_table(
        index=["local_index", "date"],
        columns="group",
        values="abs_ig",
        aggfunc="sum",
        fill_value=0.0,
    )
    for node_name in graph_node_names:
        if node_name not in pivot.columns:
            pivot[node_name] = 0.0
    pivot = pivot[graph_node_names]
    row_totals = pivot.sum(axis=1).replace(0.0, np.nan)
    focus_distribution = pivot.div(row_totals, axis=0).fillna(0.0)
    htgnn_summary = {
        node_name: {
            "mean": float(focus_distribution[node_name].mean()),
            "variance": float(focus_distribution[node_name].var(ddof=0)),
            "std": float(focus_distribution[node_name].std(ddof=0)),
        }
        for node_name in graph_node_names
    }
    node_scores = {
        node_name: htgnn_summary[node_name]["mean"]
        for node_name in graph_node_names
    }
    feature_attention: dict[str, dict[str, float]] = {}
    for group in groups:
        if group.name not in feature_totals:
            continue
        totals = feature_totals[group.name]
        total = float(totals.sum())
        shares = totals / total if total > 1.0e-12 else totals
        labels = node_input_labels(
            dataset=base_dataset,
            node_name=group.node_name,
            feature_count=len(shares),
        )
        feature_attention[group.name] = {
            label: float(value)
            for label, value in zip(labels, shares, strict=True)
        }

    sample_path = output / "integrated_gradients_local_values.csv"
    summary_path = output / "integrated_gradients_node_summary.csv"
    bar_path = output / "integrated_gradients_node_importance.svg"
    graph_path = output / "integrated_gradients_graph.svg"
    frame.to_csv(sample_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    plot_focus_histogram(
        summary=htgnn_summary,
        output_path=bar_path,
        title="Integrated Gradients Node Focus for mean",
    )
    plot_focus_graph(
        summary=htgnn_summary,
        output_path=graph_path,
        title="Integrated Gradients Node Focus for mean",
    )
    artifacts = [
        manifest_entry(sample_path, "table", "integrated_gradients"),
        manifest_entry(summary_path, "table", "integrated_gradients"),
        manifest_entry(bar_path, "figure", "integrated_gradients"),
        manifest_entry(graph_path, "figure", "integrated_gradients"),
    ]
    return node_scores, feature_attention, artifacts
