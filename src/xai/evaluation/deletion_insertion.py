"""Deletion/insertion sanity check for grouped explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.xai.common.data_adapter import make_loader, move_to_device
from src.xai.common.grouping import FeatureGroup, group_baselines, observed_node_groups
from src.xai.common.io import manifest_entry, write_json
from src.xai.common.model_adapter import prediction_for_batch
from src.xai.common.performance import (
    annualized_growth,
    equal_weight_log_returns,
    model_log_returns,
    performance_ratio,
)
from src.xai.common.perturbations import clone_batch
from src.xai.common.plotting import save_line_plot
from src.xai.common.targets import TargetSpec


def _ordered_groups_from_scores(
    full_dataset: Any,
    node_scores: dict[str, float],
) -> list[FeatureGroup]:
    """Resolve node score ordering into FeatureGroup objects."""
    base_dataset = full_dataset.dataset if hasattr(full_dataset, "dataset") else full_dataset
    groups = observed_node_groups(dataset=base_dataset)
    by_name = {group.name: group for group in groups}
    ordered_names = sorted(
        [name for name in node_scores if name in by_name],
        key=lambda name: float(node_scores[name]),
        reverse=True,
    )
    missing = [group for group in groups if group.name not in ordered_names]
    ordered = [by_name[name] for name in ordered_names] + missing
    return ordered if ordered else groups


def _replace_groups(
    batch: dict[str, Any],
    groups: list[FeatureGroup],
    baselines: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Return a copy with selected groups replaced by background baselines."""
    perturbed = clone_batch(batch=batch)
    for group in groups:
        if group.node_name not in perturbed["nodes"] or group.node_name not in baselines:
            continue
        value = perturbed["nodes"][group.node_name]
        baseline = baselines[group.node_name].to(device=value.device, dtype=value.dtype)
        perturbed["nodes"][group.node_name] = baseline.expand_as(value).clone()
    return perturbed


def run_deletion_insertion(
    model: torch.nn.Module,
    dataset: Any,
    background_dataset: Any,
    full_dataset: Any,
    target_spec: TargetSpec,
    node_scores: dict[str, float],
    top_k: int,
    device: torch.device,
    output_dir: str | Path,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    """Run deletion/insertion by keeping or removing top-ranked groups."""
    del target_spec
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ordered_groups = _ordered_groups_from_scores(
        full_dataset=full_dataset,
        node_scores=node_scores,
    )
    max_group_count = min(max(int(top_k), 1), len(ordered_groups))
    top_groups = ordered_groups[:max_group_count]
    point_count = min(6, max_group_count + 1)
    steps_to_evaluate = sorted(
        set(np.linspace(0, max_group_count, point_count).astype(int).tolist())
    )
    background_batches = [
        move_to_device(batch=batch, device=device)
        for batch in make_loader(background_dataset, batch_size=64, shuffle=False)
    ]
    baselines = group_baselines(background_batches=background_batches)
    curve_rows: list[dict[str, float | int]] = []
    detail_rows: list[dict[str, float | int | str]] = []
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    returns_by_curve: dict[str, list[torch.Tensor]] = {
        "original": [],
        **{f"deletion_{step}": [] for step in steps_to_evaluate},
        **{f"insertion_{step}": [] for step in steps_to_evaluate},
    }
    standard_returns: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            moved = move_to_device(batch=batch, device=device)
            standard_returns.append(equal_weight_log_returns(batch=moved).detach().cpu())
            original_prediction = prediction_for_batch(model=model, batch=moved)
            original_returns = model_log_returns(
                prediction=original_prediction,
                batch=moved,
            )
            returns_by_curve["original"].append(original_returns.detach().cpu())
            for step in steps_to_evaluate:
                deletion = _replace_groups(
                    batch=moved,
                    groups=top_groups[:step],
                    baselines=baselines,
                )
                insertion = _replace_groups(
                    batch=moved,
                    groups=top_groups[step:],
                    baselines=baselines,
                )
                deletion_prediction = prediction_for_batch(model=model, batch=deletion)
                insertion_prediction = prediction_for_batch(model=model, batch=insertion)
                deletion_returns = model_log_returns(
                    prediction=deletion_prediction,
                    batch=deletion,
                )
                insertion_returns = model_log_returns(
                    prediction=insertion_prediction,
                    batch=insertion,
                )
                returns_by_curve[f"deletion_{step}"].append(
                    deletion_returns.detach().cpu()
                )
                returns_by_curve[f"insertion_{step}"].append(
                    insertion_returns.detach().cpu()
                )
                batch_standard = standard_returns[-1]
                deletion_metrics = performance_ratio(
                    model_returns=deletion_returns.detach().cpu(),
                    standard_returns=batch_standard,
                )
                insertion_metrics = performance_ratio(
                    model_returns=insertion_returns.detach().cpu(),
                    standard_returns=batch_standard,
                )
                detail_rows.append(
                    {
                        "batch_index": batch_idx,
                        "step": step,
                        "deleted_groups": step,
                        "inserted_groups": step,
                        "insertion_order": "most_to_least_important",
                        "deletion_performance_ratio": deletion_metrics[
                            "performance_ratio"
                        ],
                        "insertion_performance_ratio": insertion_metrics[
                            "performance_ratio"
                        ],
                        "deletion_model_annualized_return": deletion_metrics[
                            "model_annualized_return"
                        ],
                        "insertion_model_annualized_return": insertion_metrics[
                            "model_annualized_return"
                        ],
                        "standard_annualized_return": deletion_metrics[
                            "standard_annualized_return"
                        ],
                    }
                )

    standard = torch.cat(standard_returns, dim=0)
    original_returns = torch.cat(returns_by_curve["original"], dim=0)
    original_growth = annualized_growth(log_returns=original_returns)
    original_metrics = performance_ratio(
        model_returns=original_returns,
        standard_returns=standard,
    )
    for step in steps_to_evaluate:
        deletion_returns = torch.cat(returns_by_curve[f"deletion_{step}"], dim=0)
        insertion_returns = torch.cat(returns_by_curve[f"insertion_{step}"], dim=0)
        deletion_metrics = performance_ratio(
            model_returns=deletion_returns,
            standard_returns=standard,
        )
        insertion_metrics = performance_ratio(
            model_returns=insertion_returns,
            standard_returns=standard,
        )
        deletion_growth = annualized_growth(log_returns=deletion_returns)
        insertion_growth = annualized_growth(log_returns=insertion_returns)
        curve_rows.append(
            {
                "step": step,
                "deletion": deletion_metrics["performance_ratio"],
                "insertion": insertion_metrics["performance_ratio"],
                "deletion_vs_full_model": deletion_growth / original_growth,
                "insertion_vs_full_model": insertion_growth / original_growth,
                "deletion_model_annualized_return": deletion_metrics[
                    "model_annualized_return"
                ],
                "insertion_model_annualized_return": insertion_metrics[
                    "model_annualized_return"
                ],
                "standard_annualized_return": deletion_metrics[
                    "standard_annualized_return"
                ],
            }
        )
    curve = pd.DataFrame(curve_rows)
    full_model_curve = curve[
        ["step", "deletion_vs_full_model", "insertion_vs_full_model"]
    ].rename(
        columns={
            "deletion_vs_full_model": "deletion",
            "insertion_vs_full_model": "insertion",
        }
    )
    details = pd.DataFrame(detail_rows)
    deletion_auc = float(np.trapezoid(curve["deletion"], curve["step"]))
    insertion_auc = float(np.trapezoid(curve["insertion"], curve["step"]))
    deletion_full_auc = float(
        np.trapezoid(full_model_curve["deletion"], full_model_curve["step"])
    )
    insertion_full_auc = float(
        np.trapezoid(full_model_curve["insertion"], full_model_curve["step"])
    )
    metrics = {
        "original_performance_ratio": original_metrics["performance_ratio"],
        "original_model_annualized_return": original_metrics[
            "model_annualized_return"
        ],
        "standard_annualized_return": original_metrics["standard_annualized_return"],
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "auc_gap_insertion_minus_deletion": insertion_auc - deletion_auc,
        "deletion_vs_full_model_auc": deletion_full_auc,
        "insertion_vs_full_model_auc": insertion_full_auc,
        "vs_full_model_auc_gap_insertion_minus_deletion": (
            insertion_full_auc - deletion_full_auc
        ),
        "top_k": int(max_group_count),
        "curve_points": int(len(steps_to_evaluate)),
        "evaluated_steps": [int(step) for step in steps_to_evaluate],
        "deletion_order": [group.name for group in top_groups],
        "insertion_order": [group.name for group in top_groups],
    }
    curve_path = output / "deletion_insertion_curve.csv"
    full_model_curve_path = output / "deletion_insertion_vs_full_model_curve.csv"
    details_path = output / "deletion_insertion_batch_details.csv"
    metrics_path = output / "deletion_insertion_metrics.json"
    figure_path = output / "deletion_insertion_curve.svg"
    full_model_figure_path = output / "deletion_insertion_vs_full_model_curve.svg"
    curve.to_csv(curve_path, index=False)
    full_model_curve.to_csv(full_model_curve_path, index=False)
    details.to_csv(details_path, index=False)
    write_json(metrics, metrics_path)
    point_labels = {
        "deletion": {
            0: "full",
            **{
                idx + 1: f"- {group.name}"
                for idx, group in enumerate(top_groups)
            },
        },
        "insertion": {
            0: "full - 5",
            **{
                idx + 1: f"+ {group.name}"
                for idx, group in enumerate(top_groups)
            },
        },
    }
    save_line_plot(
        frame=curve[["step", "deletion", "insertion"]],
        output_path=figure_path,
        title="Deletion/insertion sanity check",
        xlabel="Number of top groups perturbed/restored",
        ylabel="Annualized return ratio vs equal-weight",
        point_labels=point_labels,
    )
    save_line_plot(
        frame=full_model_curve,
        output_path=full_model_figure_path,
        title="Deletion/insertion sanity check vs full model",
        xlabel="Number of top groups perturbed/restored",
        ylabel="Annualized return ratio vs full model",
        point_labels=point_labels,
    )
    artifacts = [
        manifest_entry(curve_path, "table", "deletion_insertion"),
        manifest_entry(
            full_model_curve_path,
            "table",
            "deletion_insertion_vs_full_model",
        ),
        manifest_entry(details_path, "table", "deletion_insertion"),
        manifest_entry(metrics_path, "metrics", "deletion_insertion"),
        manifest_entry(figure_path, "figure", "deletion_insertion"),
        manifest_entry(
            full_model_figure_path,
            "figure",
            "deletion_insertion_vs_full_model",
        ),
    ]
    return metrics, artifacts
