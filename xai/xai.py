"""Main entry point for the HTGNN/BHTGNN XAI pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from src.utils import seed_everything
from src.xai.common.data_adapter import load_xai_data
from src.xai.common.io import ensure_xai_dirs, manifest_entry, write_json, write_manifest
from src.xai.common.model_adapter import choose_device, load_xai_model
from src.xai.common.recycled_htgnn_plots import input_attention_pie_charts
from src.xai.common.targets import resolve_target
from src.xai.evaluation.deletion_insertion import run_deletion_insertion
from src.xai.global_explanations.grouped_shap import run_portfolio_signal_shap
from src.xai.local_explanations.integrated_gradients import run_integrated_gradients
from src.xai.local_explanations.temporal_occlusion import run_temporal_occlusion


ALL_METHODS = {
    "grouped_shap",
    "integrated_gradients",
    "temporal_occlusion",
    "subgraph_occlusion",
    "deletion_insertion",
}


def parse_args() -> argparse.Namespace:
    """Parse XAI pipeline CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete XAI pipeline for HTGNN/BHTGNN checkpoints."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or latest.")
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "HTGNN", "BHTGNN"],
        help="Expected model type, or auto to infer it from the checkpoint.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to explain.",
    )
    parser.add_argument(
        "--target",
        default="variance",
        help="Target scalar: variance, allocation_change_norm, or weights:<asset>.",
    )
    parser.add_argument(
        "--output-dir",
        default="xai",
        help="Directory where XAI artifacts are saved.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed dataset tensors.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, or auto.",
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--background-samples", type=int, default=32)
    parser.add_argument("--n-local", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods or all. Available: grouped_shap, "
            "integrated_gradients, temporal_occlusion, subgraph_occlusion, "
            "deletion_insertion."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Internal XAI batch size.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=24,
        help="Integrated Gradients interpolation steps.",
    )
    parser.add_argument(
        "--temporal-windows",
        type=int,
        default=5,
        help="Number of temporal windows used by occlusion.",
    )
    return parser.parse_args()


def resolve_methods(raw_methods: str) -> set[str]:
    """Resolve CLI method selection."""
    if raw_methods == "all":
        return set(ALL_METHODS)
    methods = {value.strip() for value in raw_methods.split(",") if value.strip()}
    unknown = methods - ALL_METHODS
    if unknown:
        raise ValueError(f"Unknown XAI methods: {sorted(unknown)}.")
    return methods


def _write_run_config(
    args: argparse.Namespace,
    loaded_model: Any,
    target_label: str,
    output_dir: Path,
    methods: set[str],
) -> Path:
    """Persist the effective XAI run configuration."""
    return write_json(
        {
            "checkpoint": str(loaded_model.checkpoint_path),
            "model": loaded_model.model_name,
            "split": args.split,
            "target": args.target,
            "resolved_target": target_label,
            "output_dir": str(output_dir),
            "processed_dir": args.processed_dir,
            "device": args.device,
            "max_samples": int(args.max_samples),
            "background_samples": int(args.background_samples),
            "n_local": int(args.n_local),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "methods": sorted(methods),
            "batch_size": int(args.batch_size),
            "ig_steps": int(args.ig_steps),
            "temporal_windows": int(args.temporal_windows),
        },
        output_dir / "xai_run_config.json",
    )


def _remove_obsolete_recycled_artifacts(local_dir: Path) -> None:
    """Remove legacy duplicated recycled plots from previous pipeline runs."""
    for filename in (
        "recycled_htgnn_graph.svg",
        "recycled_htgnn_histogram.svg",
        "recycled_htgnn_pie_charts.svg",
        "subgraph_occlusion_graph.svg",
        "subgraph_occlusion_node_importance.svg",
        "subgraph_occlusion_local_values.csv",
        "subgraph_occlusion_node_summary.csv",
    ):
        path = local_dir / filename
        if path.exists():
            path.unlink()


def _remove_obsolete_global_artifacts(global_dir: Path) -> None:
    """Remove legacy grouped-SHAP artifacts superseded by portfolio-signal SHAP."""
    for pattern in (
        "grouped_shap_node_importance.*",
        "grouped_shap_sample_values.csv",
        "grouped_shap_surrogate_coefficients.csv",
        "grouped_shap_metrics.json",
        "grouped_shap_mean_*",
    ):
        for path in global_dir.glob(pattern):
            path.unlink()


def main() -> None:
    """Run the complete XAI pipeline."""
    args = parse_args()
    methods = resolve_methods(args.methods)
    seed_everything(seed=int(args.seed))
    torch.set_num_threads(1)
    output_paths = ensure_xai_dirs(output_dir=args.output_dir)
    _remove_obsolete_recycled_artifacts(local_dir=output_paths["local"])
    _remove_obsolete_global_artifacts(global_dir=output_paths["global"])
    artifacts: list[dict[str, str]] = []

    device = choose_device(args.device)
    loaded = load_xai_model(
        checkpoint=args.checkpoint,
        requested_model=args.model,
        device=device,
    )
    data = load_xai_data(
        processed_dir=args.processed_dir,
        split=args.split,
        max_samples=args.max_samples,
        background_samples=args.background_samples,
        n_local=args.n_local,
    )
    target_spec = resolve_target(target=args.target, metadata=loaded.metadata)

    config_path = _write_run_config(
        args=args,
        loaded_model=loaded,
        target_label=str(target_spec.label),
        output_dir=output_paths["root"],
        methods=methods,
    )
    artifacts.append(manifest_entry(config_path, "config", "pipeline"))

    global_scores: dict[str, float] = {}
    ig_scores: dict[str, float] = {}
    ig_feature_attention: dict[str, dict[str, float]] = {}
    node_occlusion_scores: dict[str, float] = {}
    metrics: dict[str, Any] = {}

    if "grouped_shap" in methods:
        global_scores, method_artifacts, grouped_metrics = run_portfolio_signal_shap(
            model=loaded.model,
            dataset=data.sampled_dataset,
            full_dataset=data.dataset,
            device=device,
            output_dir=output_paths["global"],
            batch_size=args.batch_size,
        )
        artifacts.extend(method_artifacts)
        metrics["grouped_shap"] = grouped_metrics

    if "integrated_gradients" in methods:
        ig_scores, ig_feature_attention, method_artifacts = run_integrated_gradients(
            model=loaded.model,
            local_dataset=data.local_dataset,
            background_dataset=data.background_dataset,
            full_dataset=data.dataset,
            target_spec=target_spec,
            device=device,
            output_dir=output_paths["local"],
            steps=args.ig_steps,
        )
        artifacts.extend(method_artifacts)

    if "temporal_occlusion" in methods or "subgraph_occlusion" in methods:
        _, node_occlusion_scores, method_artifacts = run_temporal_occlusion(
            model=loaded.model,
            local_dataset=data.local_dataset,
            background_dataset=data.background_dataset,
            target_spec=target_spec,
            device=device,
            output_dir=output_paths["local"],
            windows=args.temporal_windows,
        )
        artifacts.extend(method_artifacts)

    if ig_feature_attention:
        ig_summary = {
            node_name: {
                "mean": float(score),
                "variance": 0.0,
                "std": 0.0,
            }
            for node_name, score in ig_scores.items()
        }
        pies_path = input_attention_pie_charts(
            summary=ig_summary,
            feature_attention=ig_feature_attention,
            output_path=output_paths["local"] / "input_attention_pie_charts.svg",
            title="Integrated Gradients Input Attribution by Node for mean",
        )
        artifacts.append(
            manifest_entry(pies_path, "figure", "input_attention_pie_charts")
        )

    if "deletion_insertion" in methods:
        ranking_scores = node_occlusion_scores or ig_scores or global_scores
        deletion_metrics, method_artifacts = run_deletion_insertion(
            model=loaded.model,
            dataset=data.sampled_dataset,
            background_dataset=data.background_dataset,
            full_dataset=data.dataset,
            target_spec=target_spec,
            node_scores=ranking_scores,
            top_k=args.top_k,
            device=device,
            output_dir=output_paths["evaluation"],
            batch_size=args.batch_size,
        )
        artifacts.extend(method_artifacts)
        metrics["deletion_insertion"] = deletion_metrics

    metrics_path = write_json(metrics, output_paths["root"] / "xai_metrics.json")
    artifacts.append(manifest_entry(metrics_path, "metrics", "pipeline"))
    manifest_path = write_manifest(entries=artifacts, output_dir=output_paths["root"])
    print(
        f"XAI pipeline complete for {loaded.model_name}. "
        f"Wrote {len(artifacts)} artifacts; manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
