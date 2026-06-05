"""Subgraph/node occlusion local explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.xai.common.data_adapter import make_loader, move_to_device
from src.xai.common.graph_plotting import plot_graph_scores
from src.xai.common.grouping import FeatureGroup, group_baselines, observed_node_groups
from src.xai.common.io import manifest_entry
from src.xai.common.model_adapter import prediction_for_batch
from src.xai.common.perturbations import replace_node_with_baseline
from src.xai.common.plotting import save_bar_plot
from src.xai.common.targets import TargetSpec, scalar_target


def run_subgraph_occlusion(
    model: torch.nn.Module,
    local_dataset: Any,
    background_dataset: Any,
    full_dataset: Any,
    target_spec: TargetSpec,
    device: torch.device,
    output_dir: str | Path,
) -> tuple[dict[str, float], list[FeatureGroup], list[dict[str, str]]]:
    """Occlude one observed graph node/subgraph at a time."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_dataset = full_dataset.dataset if hasattr(full_dataset, "dataset") else full_dataset
    groups = observed_node_groups(dataset=base_dataset)
    background_batches = [
        move_to_device(batch=batch, device=device)
        for batch in make_loader(background_dataset, batch_size=64, shuffle=False)
    ]
    baselines = group_baselines(background_batches=background_batches)
    rows: list[dict[str, Any]] = []
    loader = make_loader(local_dataset, batch_size=max(len(local_dataset), 1), shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_to_device(batch=batch, device=device)
            prediction = prediction_for_batch(model=model, batch=moved)
            base_values = scalar_target(prediction=prediction, batch=moved, spec=target_spec)
            dates = [str(value) for value in batch["date"]]
            for group in groups:
                perturbed = replace_node_with_baseline(
                    batch=moved,
                    group=group,
                    baselines=baselines,
                )
                perturbed_prediction = prediction_for_batch(model=model, batch=perturbed)
                perturbed_values = scalar_target(
                    prediction=perturbed_prediction,
                    batch=perturbed,
                    spec=target_spec,
                )
                deltas = (base_values - perturbed_values).detach().cpu().numpy()
                for local_idx, (date, delta) in enumerate(zip(dates, deltas, strict=True)):
                    rows.append(
                        {
                            "local_index": local_idx,
                            "date": date,
                            "group": group.name,
                            "target_drop": float(delta),
                            "abs_target_drop": float(abs(delta)),
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("group", as_index=False)
        .agg(mean_target_drop=("target_drop", "mean"), mean_abs_target_drop=("abs_target_drop", "mean"))
        .sort_values("mean_abs_target_drop", ascending=False)
    )
    scores = {
        str(row["group"]): float(row["mean_target_drop"])
        for _, row in summary.iterrows()
    }
    ordered_groups = [
        group for group_name in summary["group"].tolist() for group in groups if group.name == group_name
    ]
    sample_path = output / "subgraph_occlusion_local_values.csv"
    summary_path = output / "subgraph_occlusion_node_summary.csv"
    bar_path = output / "subgraph_occlusion_node_importance.svg"
    graph_path = output / "subgraph_occlusion_graph.svg"
    frame.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_bar_plot(
        values=scores,
        output_path=bar_path,
        title=f"Subgraph occlusion effect for {target_spec.label}",
        ylabel="Mean target drop",
    )
    plot_graph_scores(
        scores=scores,
        output_path=graph_path,
        title=f"Subgraph occlusion graph effect for {target_spec.label}",
    )
    artifacts = [
        manifest_entry(sample_path, "table", "subgraph_occlusion"),
        manifest_entry(summary_path, "table", "subgraph_occlusion"),
        manifest_entry(bar_path, "figure", "subgraph_occlusion"),
        manifest_entry(graph_path, "figure", "subgraph_occlusion"),
    ]
    return scores, ordered_groups, artifacts

