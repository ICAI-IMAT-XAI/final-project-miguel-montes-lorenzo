"""Grouped SHAP-style global explanations with an interpretable surrogate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.xai.common.data_adapter import make_loader, move_to_device
from src.xai.common.grouping import (
    FeatureGroup,
    observed_node_groups,
    summarize_nodes_for_surrogate,
    surrogate_feature_names,
    surrogate_group_slices,
)
from src.xai.common.io import manifest_entry, write_json
from src.xai.common.model_adapter import prediction_for_batch
from src.xai.common.plotting import save_bar_plot
from src.xai.common.targets import TargetSpec, scalar_target


def _collect_surrogate_table(
    model: torch.nn.Module,
    dataset: Any,
    groups: list[FeatureGroup],
    target_spec: TargetSpec,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect grouped tabular features and black-box targets."""
    loader = make_loader(dataset=dataset, batch_size=batch_size, shuffle=False)
    feature_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dates: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_to_device(batch=batch, device=device)
            prediction = prediction_for_batch(model=model, batch=moved)
            values = scalar_target(
                prediction=prediction,
                batch=moved,
                spec=target_spec,
            )
            feature_blocks.append(summarize_nodes_for_surrogate(batch=moved, groups=groups))
            targets.append(values.detach().cpu().numpy())
            dates.extend([str(value) for value in batch["date"]])
    return np.concatenate(feature_blocks, axis=0), np.concatenate(targets, axis=0), dates


def _collect_mean_surrogate_table(
    model: torch.nn.Module,
    dataset: Any,
    groups: list[FeatureGroup],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect grouped tabular features and vector mean-allocation targets."""
    loader = make_loader(dataset=dataset, batch_size=batch_size, shuffle=False)
    feature_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dates: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_to_device(batch=batch, device=device)
            prediction = prediction_for_batch(model=model, batch=moved)
            feature_blocks.append(summarize_nodes_for_surrogate(batch=moved, groups=groups))
            targets.append(prediction.weights.detach().cpu().numpy())
            dates.extend([str(value) for value in batch["date"]])
    return np.concatenate(feature_blocks, axis=0), np.concatenate(targets, axis=0), dates


def _linear_group_contributions(
    pipeline: Pipeline,
    x_values: np.ndarray,
    groups: list[FeatureGroup],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute independent-baseline linear SHAP values per group."""
    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["ridge"]
    z_values = scaler.transform(x_values)
    baseline = z_values.mean(axis=0, keepdims=True)
    centered = z_values - baseline
    coefficients = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
    contributions = centered * coefficients.reshape(1, -1)
    group_slices = surrogate_group_slices(groups=groups)

    rows: list[dict[str, float | str]] = []
    sample_rows: list[dict[str, float | str | int]] = []
    for group in groups:
        group_values = contributions[:, group_slices[group.name]].sum(axis=1)
        rows.append(
            {
                "group": group.name,
                "mean_signed_shap": float(group_values.mean()),
                "mean_abs_shap": float(np.abs(group_values).mean()),
            }
        )
        for sample_idx, value in enumerate(group_values):
            sample_rows.append(
                {
                    "sample_index": sample_idx,
                    "group": group.name,
                    "shap_value": float(value),
                    "abs_shap_value": float(abs(value)),
                }
            )
    summary = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    per_sample = pd.DataFrame(sample_rows)
    return summary, per_sample


def _fit_ridge_surrogate(x_values: np.ndarray, y_values: np.ndarray) -> Pipeline:
    """Fit the standardized Ridge surrogate used by grouped SHAP."""
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 17))),
        ]
    )
    pipeline.fit(x_values, y_values)
    return pipeline


def run_grouped_shap(
    model: torch.nn.Module,
    dataset: Any,
    full_dataset: Any,
    target_spec: TargetSpec,
    device: torch.device,
    output_dir: str | Path,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict[str, str]], dict[str, float]]:
    """Run grouped SHAP over a linear surrogate of the checkpoint outputs."""
    del full_dataset
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = observed_node_groups(dataset=dataset.dataset if hasattr(dataset, "dataset") else dataset)
    x_values, y_values, dates = _collect_surrogate_table(
        model=model,
        dataset=dataset,
        groups=groups,
        target_spec=target_spec,
        device=device,
        batch_size=batch_size,
    )
    if x_values.shape[0] < 2:
        raise RuntimeError("Grouped SHAP needs at least two samples.")

    pipeline = _fit_ridge_surrogate(x_values=x_values, y_values=y_values)
    predictions = pipeline.predict(x_values)
    metrics = {
        "surrogate_r2": float(r2_score(y_values, predictions)),
        "surrogate_mse": float(mean_squared_error(y_values, predictions)),
        "n_samples": int(x_values.shape[0]),
        "n_surrogate_features": int(x_values.shape[1]),
    }
    summary, per_sample = _linear_group_contributions(
        pipeline=pipeline,
        x_values=x_values,
        groups=groups,
    )
    per_sample["date"] = per_sample["sample_index"].map(lambda idx: dates[int(idx)])

    summary_path = output / "grouped_shap_node_importance.csv"
    sample_path = output / "grouped_shap_sample_values.csv"
    metrics_path = output / "grouped_shap_metrics.json"
    coefficients_path = output / "grouped_shap_surrogate_coefficients.csv"
    figure_path = output / "grouped_shap_node_importance.svg"

    summary.to_csv(summary_path, index=False)
    per_sample.to_csv(sample_path, index=False)
    coefficient_frame = pd.DataFrame(
        {
            "feature": surrogate_feature_names(groups=groups),
            "coefficient": pipeline.named_steps["ridge"].coef_,
        }
    )
    coefficient_frame.to_csv(coefficients_path, index=False)
    write_json(metrics, metrics_path)
    save_bar_plot(
        values={
            str(row["group"]): float(row["mean_signed_shap"])
            for _, row in summary.iterrows()
        },
        output_path=figure_path,
        title=f"Grouped SHAP surrogate attribution for {target_spec.label}",
        ylabel="Mean signed SHAP value",
        top_n=20,
    )
    node_scores = {
        str(row["group"]): float(row["mean_signed_shap"])
        for _, row in summary.iterrows()
    }
    artifacts = [
        manifest_entry(summary_path, "table", "grouped_shap"),
        manifest_entry(sample_path, "table", "grouped_shap"),
        manifest_entry(coefficients_path, "table", "grouped_shap"),
        manifest_entry(metrics_path, "metrics", "grouped_shap"),
        manifest_entry(figure_path, "figure", "grouped_shap"),
    ]
    return node_scores, artifacts, metrics


def run_grouped_shap_mean(
    model: torch.nn.Module,
    dataset: Any,
    full_dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict[str, str]], dict[str, float]]:
    """Run grouped SHAP for the model mean allocation vector."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    groups = observed_node_groups(dataset=base_dataset)
    x_values, y_values, dates = _collect_mean_surrogate_table(
        model=model,
        dataset=dataset,
        groups=groups,
        device=device,
        batch_size=batch_size,
    )
    if x_values.shape[0] < 2:
        raise RuntimeError("Grouped SHAP for mean needs at least two samples.")

    currencies = list(getattr(full_dataset, "metadata", {}).get("currencies", []))
    if not currencies:
        currencies = [f"asset_{idx}" for idx in range(y_values.shape[1])]

    group_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    signed_totals = {group.name: [] for group in groups}
    abs_totals = {group.name: [] for group in groups}

    for asset_idx, asset_name in enumerate(currencies):
        asset_target = y_values[:, asset_idx]
        pipeline = _fit_ridge_surrogate(x_values=x_values, y_values=asset_target)
        predictions = pipeline.predict(x_values)
        metric_rows.append(
            {
                "asset": asset_name,
                "surrogate_r2": float(r2_score(asset_target, predictions)),
                "surrogate_mse": float(mean_squared_error(asset_target, predictions)),
            }
        )
        summary, per_sample = _linear_group_contributions(
            pipeline=pipeline,
            x_values=x_values,
            groups=groups,
        )
        for _, row in summary.iterrows():
            group_name = str(row["group"])
            signed_value = float(row["mean_signed_shap"])
            abs_value = float(row["mean_abs_shap"])
            signed_totals[group_name].append(signed_value)
            abs_totals[group_name].append(abs_value)
            group_rows.append(
                {
                    "asset": asset_name,
                    "group": group_name,
                    "mean_signed_shap": signed_value,
                    "mean_abs_shap": abs_value,
                }
            )
        per_sample["asset"] = asset_name
        per_sample["date"] = per_sample["sample_index"].map(
            lambda idx: dates[int(idx)]
        )
        sample_rows.extend(per_sample.to_dict(orient="records"))
        for feature, coefficient in zip(
            surrogate_feature_names(groups=groups),
            pipeline.named_steps["ridge"].coef_,
            strict=True,
        ):
            coefficient_rows.append(
                {
                    "asset": asset_name,
                    "feature": feature,
                    "coefficient": float(coefficient),
                }
            )

    summary_frame = pd.DataFrame(
        [
            {
                "group": group.name,
                "mean_signed_shap": float(np.mean(signed_totals[group.name])),
                "mean_abs_shap": float(np.mean(abs_totals[group.name])),
            }
            for group in groups
        ]
    ).sort_values("mean_abs_shap", ascending=False)
    by_asset_frame = pd.DataFrame(group_rows)
    sample_frame = pd.DataFrame(sample_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics = {
        "mean_surrogate_r2": float(metrics_frame["surrogate_r2"].mean()),
        "mean_surrogate_mse": float(metrics_frame["surrogate_mse"].mean()),
        "n_samples": int(x_values.shape[0]),
        "n_surrogate_features": int(x_values.shape[1]),
        "n_assets": int(y_values.shape[1]),
    }

    summary_path = output / "grouped_shap_mean_node_importance.csv"
    by_asset_path = output / "grouped_shap_mean_by_asset.csv"
    sample_path = output / "grouped_shap_mean_sample_values.csv"
    coefficients_path = output / "grouped_shap_mean_surrogate_coefficients.csv"
    metrics_path = output / "grouped_shap_mean_metrics.json"
    figure_path = output / "grouped_shap_mean_node_importance.svg"

    summary_frame.to_csv(summary_path, index=False)
    by_asset_frame.to_csv(by_asset_path, index=False)
    sample_frame.to_csv(sample_path, index=False)
    pd.DataFrame(coefficient_rows).to_csv(coefficients_path, index=False)
    write_json(metrics, metrics_path)
    save_bar_plot(
        values={
            str(row["group"]): float(row["mean_abs_shap"])
            for _, row in summary_frame.iterrows()
        },
        output_path=figure_path,
        title="Grouped SHAP surrogate attribution for mean",
        ylabel="Mean absolute SHAP value across assets",
        top_n=20,
    )
    node_scores = {
        str(row["group"]): float(row["mean_abs_shap"])
        for _, row in summary_frame.iterrows()
    }
    artifacts = [
        manifest_entry(summary_path, "table", "grouped_shap_mean"),
        manifest_entry(by_asset_path, "table", "grouped_shap_mean"),
        manifest_entry(sample_path, "table", "grouped_shap_mean"),
        manifest_entry(coefficients_path, "table", "grouped_shap_mean"),
        manifest_entry(metrics_path, "metrics", "grouped_shap_mean"),
        manifest_entry(figure_path, "figure", "grouped_shap_mean"),
    ]
    return node_scores, artifacts, metrics


def _collect_portfolio_signal_table(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Collect portfolio-signal inputs and per-currency mean/variance outputs."""
    loader = make_loader(dataset=dataset, batch_size=batch_size, shuffle=False)
    input_blocks: list[np.ndarray] = []
    mean_targets: list[np.ndarray] = []
    variance_targets: list[np.ndarray] = []
    dates: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_to_device(batch=batch, device=device)
            prediction = prediction_for_batch(model=model, batch=moved)
            portfolio_signal = moved["nodes"]["portfolio_signal"]
            input_blocks.append(
                portfolio_signal.reshape(portfolio_signal.shape[0], -1)
                .detach()
                .cpu()
                .numpy()
            )
            mean_targets.append(prediction.weights.detach().cpu().numpy())
            if prediction.weight_uncertainty is not None:
                variance_targets.append(
                    prediction.weight_uncertainty.pow(2).detach().cpu().numpy()
                )
            else:
                variance_targets.append(
                    np.zeros_like(prediction.weights.detach().cpu().numpy())
                )
            dates.extend([str(value) for value in batch["date"]])
    return (
        np.concatenate(input_blocks, axis=0),
        np.concatenate(mean_targets, axis=0),
        np.concatenate(variance_targets, axis=0),
        dates,
        int(input_blocks[0].shape[1] // mean_targets[0].shape[1]),
    )


def _linear_feature_contributions(
    pipeline: Pipeline,
    x_values: np.ndarray,
) -> np.ndarray:
    """Compute linear SHAP-style feature contributions for one output."""
    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["ridge"]
    z_values = scaler.transform(x_values)
    centered = z_values - z_values.mean(axis=0, keepdims=True)
    coefficients = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
    return centered * coefficients.reshape(1, -1)


def _plot_portfolio_signal_shap_grid(
    summary: pd.DataFrame,
    currencies: list[str],
    constant_inputs: set[str],
    output_path: str | Path,
    title: str,
    ylabel: str,
    value_column: str,
) -> Path:
    """Plot 8 SHAP bar subplots, one output currency per subplot."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    max_value = float(summary[value_column].abs().max())
    y_limit = max(max_value * 1.18, 1.0e-8)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 7.4), sharey=True)
    axes_flat = axes.reshape(-1)
    for axis, output_currency in zip(axes_flat, currencies, strict=False):
        rows = summary[summary["output_currency"] == output_currency]
        values = rows.set_index("input_currency").loc[currencies, value_column]
        if value_column == "mean_signed_shap":
            colors = ["#dc2626" if value >= 0.0 else "#2563eb" for value in values]
            axis.axhline(0.0, color="#0f172a", linewidth=0.8)
            axis.set_ylim(-y_limit, y_limit)
        else:
            colors = ["#2563eb" for _ in values]
            axis.set_ylim(0.0, y_limit)
        axis.bar(np.arange(len(currencies)), values.to_numpy(), color=colors, alpha=0.88)
        for input_idx, input_currency in enumerate(currencies):
            if input_currency in constant_inputs:
                axis.axvspan(input_idx - 0.45, input_idx + 0.45, color="#e5e7eb", alpha=0.55)
        axis.set_title(output_currency, fontsize=11, fontweight="bold")
        tick_labels = [
            f"{currency}\nconst." if currency in constant_inputs else currency
            for currency in currencies
        ]
        axis.set_xticks(np.arange(len(currencies)), tick_labels, rotation=45, ha="right")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    for axis in axes_flat[len(currencies) :]:
        axis.axis("off")
    axes_flat[0].set_ylabel(ylabel)
    axes_flat[4].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if constant_inputs:
        fig.text(
            0.5,
            0.01,
            "Grey input columns are constant in portfolio_signal, so SHAP is exactly zero there.",
            ha="center",
            fontsize=9,
            color="#475569",
        )
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output


def _portfolio_signal_shap_for_targets(
    x_values: np.ndarray,
    y_values: np.ndarray,
    currencies: list[str],
    target_name: str,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fit one surrogate per output currency and compute feature SHAP summaries."""
    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for output_idx, output_currency in enumerate(currencies):
        target = y_values[:, output_idx]
        pipeline = _fit_ridge_surrogate(x_values=x_values, y_values=target)
        predictions = pipeline.predict(x_values)
        contributions = _linear_feature_contributions(
            pipeline=pipeline,
            x_values=x_values,
        )
        metric_rows.append(
            {
                "target": target_name,
                "output_currency": output_currency,
                "surrogate_r2": float(r2_score(target, predictions)),
                "surrogate_mse": float(mean_squared_error(target, predictions)),
            }
        )
        for input_idx, input_currency in enumerate(currencies):
            feature_indices = [
                time_idx * len(currencies) + input_idx for time_idx in range(lookback)
            ]
            values = contributions[:, feature_indices].sum(axis=1)
            summary_rows.append(
                {
                    "target": target_name,
                    "output_currency": output_currency,
                    "input_currency": input_currency,
                    "mean_signed_shap": float(values.mean()),
                    "mean_abs_shap": float(np.abs(values).mean()),
                }
            )
            for sample_idx, value in enumerate(values):
                sample_rows.append(
                    {
                        "target": target_name,
                        "sample_index": sample_idx,
                        "output_currency": output_currency,
                        "input_currency": input_currency,
                        "shap_value": float(value),
                        "abs_shap_value": float(abs(value)),
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics = {
        "target": target_name,
        "mean_surrogate_r2": float(metrics_frame["surrogate_r2"].mean()),
        "mean_surrogate_mse": float(metrics_frame["surrogate_mse"].mean()),
        "n_samples": int(x_values.shape[0]),
        "n_input_currencies": int(len(currencies)),
        "lookback": int(lookback),
        "n_surrogate_features": int(x_values.shape[1]),
        "n_output_currencies": int(y_values.shape[1]),
    }
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(sample_rows),
        metrics_frame,
        metrics,
    )


def run_portfolio_signal_shap(
    model: torch.nn.Module,
    dataset: Any,
    full_dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict[str, str]], dict[str, Any]]:
    """Run SHAP over portfolio-signal currency inputs for mean and variance."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    currencies = list(getattr(full_dataset, "metadata", {}).get("currencies", []))
    if not currencies:
        raise RuntimeError("Portfolio-signal SHAP needs currency metadata.")
    x_values, mean_values, variance_values, dates, lookback = _collect_portfolio_signal_table(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
    )
    if x_values.shape[1] != lookback * len(currencies):
        raise RuntimeError(
            "Portfolio-signal SHAP expects one temporal channel per currency; "
            f"got {x_values.shape[1]} inputs and {len(currencies)} currencies."
        )
    constant_inputs = {
        currency
        for input_idx, currency in enumerate(currencies)
        if float(
            x_values[
                :,
                [time_idx * len(currencies) + input_idx for time_idx in range(lookback)],
            ].std()
        )
        <= 1.0e-12
    }

    artifacts: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    node_score_values: list[float] = []

    for target_name, y_values, figure_title, figure_path, signed_figure_path in (
        (
            "mean",
            mean_values,
            "Portfolio-signal SHAP influence on predicted means",
            output / "portfolio_signal_shap_mean.svg",
            output / "portfolio_signal_shap_mean_signed.svg",
        ),
        (
            "variance",
            variance_values,
            "Portfolio-signal SHAP influence on predicted variances",
            output / "portfolio_signal_shap_variance.svg",
            output / "portfolio_signal_shap_variance_signed.svg",
        ),
    ):
        summary, samples, surrogate_metrics, target_metrics = (
            _portfolio_signal_shap_for_targets(
                x_values=x_values,
                y_values=y_values,
                currencies=currencies,
                target_name=target_name,
                lookback=lookback,
            )
        )
        target_metrics["constant_input_currencies"] = sorted(constant_inputs)
        samples["date"] = samples["sample_index"].map(lambda idx: dates[int(idx)])
        summary_path = output / f"portfolio_signal_shap_{target_name}_summary.csv"
        sample_path = output / f"portfolio_signal_shap_{target_name}_samples.csv"
        metrics_path = output / f"portfolio_signal_shap_{target_name}_metrics.json"
        surrogate_metrics_path = (
            output / f"portfolio_signal_shap_{target_name}_surrogate_metrics.csv"
        )
        summary.to_csv(summary_path, index=False)
        samples.to_csv(sample_path, index=False)
        surrogate_metrics.to_csv(surrogate_metrics_path, index=False)
        write_json(target_metrics, metrics_path)
        _plot_portfolio_signal_shap_grid(
            summary=summary,
            currencies=currencies,
            constant_inputs=constant_inputs,
            output_path=figure_path,
            title=figure_title,
            ylabel="Mean absolute SHAP value",
            value_column="mean_abs_shap",
        )
        _plot_portfolio_signal_shap_grid(
            summary=summary,
            currencies=currencies,
            constant_inputs=constant_inputs,
            output_path=signed_figure_path,
            title=f"{figure_title} (signed)",
            ylabel="Mean signed SHAP value",
            value_column="mean_signed_shap",
        )
        artifacts.extend(
            [
                manifest_entry(summary_path, "table", f"portfolio_signal_shap_{target_name}"),
                manifest_entry(sample_path, "table", f"portfolio_signal_shap_{target_name}"),
                manifest_entry(
                    surrogate_metrics_path,
                    "table",
                    f"portfolio_signal_shap_{target_name}",
                ),
                manifest_entry(metrics_path, "metrics", f"portfolio_signal_shap_{target_name}"),
                manifest_entry(figure_path, "figure", f"portfolio_signal_shap_{target_name}"),
                manifest_entry(
                    signed_figure_path,
                    "figure",
                    f"portfolio_signal_shap_{target_name}",
                ),
            ]
        )
        metrics[f"portfolio_signal_shap_{target_name}"] = target_metrics
        node_score_values.append(float(summary["mean_abs_shap"].mean()))

    node_scores = {"portfolio_signal": float(np.mean(node_score_values))}
    return node_scores, artifacts, metrics
