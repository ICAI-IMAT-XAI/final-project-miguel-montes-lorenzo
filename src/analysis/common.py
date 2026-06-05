"""Shared helpers for analysis visualizations."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ForexPortfolioDataset, move_batch_to_device
from src.eval import resolve_checkpoint
from src.models import load_model_from_checkpoint
from src.models.base import PortfolioModule
from src.models.pointwise.HTGNN.model import HTGNNModel, classify_relation
from src.utils import ensure_dir

BLUE_RED_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_red_fast",
    ["#1d4ed8", "#93c5fd", "#fca5a5", "#dc2626"],
)


def parse_focus_args(description: str) -> argparse.Namespace:
    """Parse common CLI arguments for node-focus scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint, or 'latest'.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed tensors.",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Dataset split to analyze: train | val | test | all.",
    )
    parser.add_argument(
        "--date-start",
        default=None,
        help="Optional analysis start date (inclusive).",
    )
    parser.add_argument(
        "--date-end",
        default=None,
        help="Optional analysis end date (exclusive).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for attribution analysis.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50,
        help="MC samples for probabilistic prediction summaries.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Integrated-gradient interpolation steps; 1 is SmoothGrad x input.",
    )
    parser.add_argument(
        "--smooth-samples",
        type=int,
        default=4,
        help="SmoothGrad noise samples averaged for gradient focus estimation.",
    )
    parser.add_argument(
        "--smooth-noise-std",
        type=float,
        default=0.02,
        help="SmoothGrad noise standard deviation in normalized input units.",
    )
    parser.add_argument(
        "--gradient-mc-samples",
        type=int,
        default=4,
        help="Dropout samples for BHTGNN gradient attributions.",
    )
    return parser.parse_args()


def build_analysis_dir(checkpoint: str | Path) -> tuple[Path, Path]:
    """Return resolved checkpoint path and visualization output directory."""
    checkpoint_path = resolve_checkpoint(checkpoint=checkpoint)
    output_dir = ensure_dir(Path("visualizations") / checkpoint_path.stem)
    return checkpoint_path, output_dir


def load_graph_model_and_dataset(
    checkpoint: str | Path,
    processed_dir: str | Path,
    split: str,
    date_start: str | None,
    date_end: str | None,
) -> tuple[PortfolioModule, dict[str, Any], ForexPortfolioDataset, torch.device, Path]:
    """Load a checkpoint, dataset, and preferred inference device."""
    checkpoint_path = resolve_checkpoint(checkpoint=checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, _ = load_model_from_checkpoint(
        path=checkpoint_path,
        map_location=device,
    )
    dataset = ForexPortfolioDataset(
        processed_dir=processed_dir,
        split=split,
        date_start=date_start,
        date_end=date_end,
    )
    return model, model_config, dataset, device, checkpoint_path


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize per-sample node scores onto the simplex."""
    denominator = scores.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 1.0e-8,
        scores / denominator,
        torch.zeros_like(scores),
    )

def _is_bayesian_htgnn(model: PortfolioModule) -> bool:
    """Return whether the model should use stochastic dropout attribution."""
    return model.__class__.__name__ == "BHTGNNModel"


def _stochastic_pass_count(model: PortfolioModule, mc_samples: int) -> int:
    """Resolve attribution stochastic passes for deterministic/probabilistic models."""
    return max(int(mc_samples), 1) if _is_bayesian_htgnn(model=model) else 1


def _smooth_noise_like(tensor: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Create scale-aware SmoothGrad noise for one tensor."""
    if noise_std <= 0.0:
        return torch.zeros_like(tensor)
    scale = tensor.detach().std().clamp_min(min=1.0e-6)
    return torch.randn_like(tensor) * float(noise_std) * scale


def _mean_direction_from_states(
    model: HTGNNModel,
    states: dict[str, torch.Tensor],
    mc_samples: int,
) -> torch.Tensor:
    """Build a detached vector-output direction for gradient attribution."""
    weight_samples: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(_stochastic_pass_count(model=model, mc_samples=mc_samples)):
            weight_samples.append(model.propagate_states(states=states).weights)
    return torch.stack(tensors=weight_samples, dim=0).mean(dim=0).detach()


def _mean_direction_from_batch(
    model: PortfolioModule,
    batch: dict[str, Any],
    mc_samples: int,
) -> torch.Tensor:
    """Build a detached vector-output direction from raw batch inputs."""
    weight_samples: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(_stochastic_pass_count(model=model, mc_samples=mc_samples)):
            weight_samples.append(model.forward(batch=batch).weights)
    return torch.stack(tensors=weight_samples, dim=0).mean(dim=0).detach()


def compute_node_focus_distribution(
    model: HTGNNModel,
    dataset: ForexPortfolioDataset,
    device: torch.device,
    node_names: list[str],
    batch_size: int,
    mc_samples: int,
    gradient_steps: int,
    smooth_samples: int,
    smooth_noise_std: float,
) -> np.ndarray:
    """Estimate node-importance distributions with SmoothGrad/IG attribution."""
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device=device)
    rows: list[np.ndarray] = []

    steps = max(int(gradient_steps), 1)
    smooth_passes = max(int(smooth_samples), 1)
    stochastic_passes = _stochastic_pass_count(model=model, mc_samples=mc_samples)
    total_passes = smooth_passes * stochastic_passes
    was_training = model.training
    model.train(mode=_is_bayesian_htgnn(model=model))

    for batch in loader:
        moved_batch = move_batch_to_device(batch=batch, device=device)
        with torch.no_grad():
            encoded_states = model.encode_nodes(nodes=moved_batch["nodes"])
            base_states = {
                name: value.detach()
                for name, value in encoded_states.items()
            }
        direction = _mean_direction_from_states(
            model=model,
            states=base_states,
            mc_samples=mc_samples,
        )
        batch_scores = torch.zeros(
            size=(direction.shape[0], len(node_names)),
            device=device,
            dtype=direction.dtype,
        )

        for _ in range(total_passes):
            noisy_states = {
                name: state + _smooth_noise_like(
                    tensor=state,
                    noise_std=smooth_noise_std,
                )
                for name, state in base_states.items()
            }
            gradient_totals = {
                name: torch.zeros_like(noisy_states[name])
                for name in node_names
            }

            for step_idx in range(1, steps + 1):
                alpha = float(step_idx) / float(steps)
                scaled_states = {
                    name: (alpha * noisy_states[name])
                    .detach()
                    .requires_grad_(True)
                    for name in node_names
                }
                prediction = model.propagate_states(states=scaled_states)
                score = (prediction.weights * direction).sum(dim=-1).sum()
                gradients = torch.autograd.grad(
                    outputs=score,
                    inputs=[scaled_states[name] for name in node_names],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for node_name, gradient in zip(
                    node_names,
                    gradients,
                    strict=True,
                ):
                    gradient_totals[node_name] = (
                        gradient_totals[node_name] + gradient / float(steps)
                    )

            for node_idx, node_name in enumerate(node_names):
                attribution = noisy_states[node_name] * gradient_totals[node_name]
                batch_scores[:, node_idx] += attribution.abs().sum(dim=-1)

        batch_scores = batch_scores / float(total_passes)
        rows.append(_normalize_scores(batch_scores).detach().cpu().numpy())

    model.train(mode=was_training)
    if not was_training:
        model.eval()

    return np.concatenate(rows, axis=0)


def summarize_focus_scores(
    focus_distribution: np.ndarray,
    node_names: list[str],
) -> dict[str, dict[str, float]]:
    """Convert per-sample focus distributions into summary statistics."""
    means = focus_distribution.mean(axis=0)
    variances = focus_distribution.var(axis=0)
    stds = focus_distribution.std(axis=0)
    return {
        node_name: {
            "mean": float(means[idx]),
            "variance": float(variances[idx]),
            "std": float(stds[idx]),
        }
        for idx, node_name in enumerate(node_names)
    }


def compute_node_input_attention(
    model: HTGNNModel,
    dataset: ForexPortfolioDataset,
    device: torch.device,
    batch_size: int,
    mc_samples: int,
    gradient_steps: int,
    smooth_samples: int,
    smooth_noise_std: float,
) -> dict[str, dict[str, float]]:
    """Estimate within-node input shares with SmoothGrad/IG attribution."""
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device=device)
    target_nodes = [
        node_name
        for node_name in model.node_names
        if (
            not node_name.startswith("generic_latent_")
            and node_name in dataset.node_inputs
        )
    ]
    input_labels = {
        node_name: node_input_labels(
            dataset=dataset,
            node_name=node_name,
            feature_count=int(dataset.node_inputs[node_name].shape[-1]),
        )
        for node_name in target_nodes
    }
    totals: dict[str, np.ndarray] = {
        node_name: np.zeros(len(labels), dtype=np.float64)
        for node_name, labels in input_labels.items()
    }
    counts: dict[str, int] = {node_name: 0 for node_name in target_nodes}

    steps = max(int(gradient_steps), 1)
    smooth_passes = max(int(smooth_samples), 1)
    stochastic_passes = _stochastic_pass_count(model=model, mc_samples=mc_samples)
    total_passes = smooth_passes * stochastic_passes
    was_training = model.training
    model.train(mode=_is_bayesian_htgnn(model=model))

    for batch in loader:
        moved_batch = move_batch_to_device(batch=batch, device=device)
        base_nodes = {
            name: value.detach()
            for name, value in moved_batch["nodes"].items()
        }
        direction = _mean_direction_from_batch(
            model=model,
            batch={**moved_batch, "nodes": base_nodes},
            mc_samples=mc_samples,
        )
        batch_feature_scores = {
            node_name: torch.zeros(
                size=(
                    direction.shape[0],
                    int(base_nodes[node_name].shape[-1]),
                ),
                device=device,
                dtype=direction.dtype,
            )
            for node_name in target_nodes
        }

        for _ in range(total_passes):
            noisy_nodes = {
                name: node + _smooth_noise_like(
                    tensor=node,
                    noise_std=smooth_noise_std,
                )
                for name, node in base_nodes.items()
            }
            gradient_totals = {
                name: torch.zeros_like(noisy_nodes[name])
                for name in base_nodes
            }

            for step_idx in range(1, steps + 1):
                alpha = float(step_idx) / float(steps)
                scaled_nodes = {
                    name: (alpha * noisy_nodes[name])
                    .detach()
                    .requires_grad_(True)
                    for name in base_nodes
                }
                prediction = model.forward(batch={**moved_batch, "nodes": scaled_nodes})
                score = (prediction.weights * direction).sum(dim=-1).sum()
                gradients = torch.autograd.grad(
                    outputs=score,
                    inputs=[scaled_nodes[name] for name in base_nodes],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for node_name, gradient in zip(
                    base_nodes,
                    gradients,
                    strict=True,
                ):
                    gradient_totals[node_name] = (
                        gradient_totals[node_name] + gradient / float(steps)
                    )

            for node_name in target_nodes:
                attribution = noisy_nodes[node_name] * gradient_totals[node_name]
                batch_feature_scores[node_name] += attribution.abs().sum(dim=1)

        for node_name in target_nodes:
            normalized_scores = _normalize_scores(
                batch_feature_scores[node_name] / float(total_passes)
            )
            totals[node_name] += normalized_scores.sum(dim=0).detach().cpu().numpy()
            counts[node_name] += int(normalized_scores.shape[0])

    model.train(mode=was_training)
    if not was_training:
        model.eval()

    output: dict[str, dict[str, float]] = {}
    for node_name, node_totals in totals.items():
        denominator = float(max(counts[node_name], 1))
        scores = node_totals / denominator
        total_score = float(scores.sum())
        if total_score > 1.0e-8:
            scores = scores / total_score
        output[node_name] = {
            label: float(score)
            for label, score in zip(input_labels[node_name], scores, strict=True)
        }
    return output


def compute_focus_attributions(
    model: HTGNNModel,
    dataset: ForexPortfolioDataset,
    device: torch.device,
    node_names: list[str],
    batch_size: int,
    mc_samples: int,
    gradient_steps: int,
    smooth_samples: int,
    smooth_noise_std: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Compute node and input attributions in one SmoothGrad/IG sweep."""
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device=device)
    observed_nodes = [
        node_name
        for node_name in node_names
        if node_name in dataset.node_inputs
    ]
    generic_nodes = [
        node_name
        for node_name in node_names
        if node_name.startswith("generic_latent_")
    ]
    input_labels = {
        node_name: node_input_labels(
            dataset=dataset,
            node_name=node_name,
            feature_count=int(dataset.node_inputs[node_name].shape[-1]),
        )
        for node_name in observed_nodes
    }
    feature_totals: dict[str, np.ndarray] = {
        node_name: np.zeros(len(labels), dtype=np.float64)
        for node_name, labels in input_labels.items()
    }
    feature_counts: dict[str, int] = {node_name: 0 for node_name in observed_nodes}
    focus_rows: list[np.ndarray] = []

    steps = max(int(gradient_steps), 1)
    smooth_passes = max(int(smooth_samples), 1)
    stochastic_passes = _stochastic_pass_count(model=model, mc_samples=mc_samples)
    total_passes = smooth_passes * stochastic_passes
    was_training = model.training
    model.train(mode=_is_bayesian_htgnn(model=model))

    for batch in tqdm(iterable=loader, desc="Attributing focus", unit="batch"):
        moved_batch = move_batch_to_device(batch=batch, device=device)
        base_nodes = {
            name: value.detach()
            for name, value in moved_batch["nodes"].items()
        }
        with torch.no_grad():
            base_states = {
                name: value.detach()
                for name, value in model.encode_nodes(nodes=base_nodes).items()
            }
        direction = _mean_direction_from_states(
            model=model,
            states=base_states,
            mc_samples=mc_samples,
        )
        batch_node_scores = torch.zeros(
            size=(direction.shape[0], len(node_names)),
            device=device,
            dtype=direction.dtype,
        )
        batch_feature_scores = {
            node_name: torch.zeros(
                size=(direction.shape[0], int(base_nodes[node_name].shape[-1])),
                device=device,
                dtype=direction.dtype,
            )
            for node_name in observed_nodes
        }

        for _ in range(total_passes):
            noisy_nodes = {
                name: node + _smooth_noise_like(
                    tensor=node,
                    noise_std=smooth_noise_std,
                )
                for name, node in base_nodes.items()
            }
            noisy_generic_states = {
                name: base_states[name] + _smooth_noise_like(
                    tensor=base_states[name],
                    noise_std=smooth_noise_std,
                )
                for name in generic_nodes
            }
            node_gradient_totals = {
                name: torch.zeros_like(noisy_nodes[name])
                for name in observed_nodes
            }
            generic_gradient_totals = {
                name: torch.zeros_like(noisy_generic_states[name])
                for name in generic_nodes
            }

            for step_idx in range(1, steps + 1):
                alpha = float(step_idx) / float(steps)
                scaled_nodes = {
                    name: (alpha * noisy_nodes[name])
                    .detach()
                    .requires_grad_(True)
                    for name in observed_nodes
                }
                scaled_generic_states = {
                    name: (alpha * noisy_generic_states[name])
                    .detach()
                    .requires_grad_(True)
                    for name in generic_nodes
                }
                states = model.encode_nodes(nodes=scaled_nodes)
                for node_name, generic_state in scaled_generic_states.items():
                    states[node_name] = generic_state
                prediction = model.propagate_states(states=states)
                score = (prediction.weights * direction).sum(dim=-1).sum()
                gradient_inputs = [
                    *[scaled_nodes[name] for name in observed_nodes],
                    *[scaled_generic_states[name] for name in generic_nodes],
                ]
                gradients = torch.autograd.grad(
                    outputs=score,
                    inputs=gradient_inputs,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                observed_gradients = gradients[: len(observed_nodes)]
                generic_gradients = gradients[len(observed_nodes) :]
                for node_name, gradient in zip(
                    observed_nodes,
                    observed_gradients,
                    strict=True,
                ):
                    node_gradient_totals[node_name] = (
                        node_gradient_totals[node_name] + gradient / float(steps)
                    )
                for node_name, gradient in zip(
                    generic_nodes,
                    generic_gradients,
                    strict=True,
                ):
                    generic_gradient_totals[node_name] = (
                        generic_gradient_totals[node_name] + gradient / float(steps)
                    )

            for node_idx, node_name in enumerate(node_names):
                if node_name in observed_nodes:
                    attribution = (
                        noisy_nodes[node_name] * node_gradient_totals[node_name]
                    )
                    batch_node_scores[:, node_idx] += attribution.abs().sum(
                        dim=(1, 2)
                    )
                    batch_feature_scores[node_name] += attribution.abs().sum(dim=1)
                elif node_name in generic_nodes:
                    attribution = (
                        noisy_generic_states[node_name]
                        * generic_gradient_totals[node_name]
                    )
                    batch_node_scores[:, node_idx] += attribution.abs().sum(dim=-1)

        batch_node_scores = batch_node_scores / float(total_passes)
        focus_rows.append(_normalize_scores(batch_node_scores).detach().cpu().numpy())
        for node_name in observed_nodes:
            normalized_scores = _normalize_scores(
                batch_feature_scores[node_name] / float(total_passes)
            )
            feature_totals[node_name] += normalized_scores.sum(
                dim=0
            ).detach().cpu().numpy()
            feature_counts[node_name] += int(normalized_scores.shape[0])

    model.train(mode=was_training)
    if not was_training:
        model.eval()

    input_attention: dict[str, dict[str, float]] = {}
    for node_name, totals in feature_totals.items():
        denominator = float(max(feature_counts[node_name], 1))
        scores = totals / denominator
        total_score = float(scores.sum())
        if total_score > 1.0e-8:
            scores = scores / total_score
        input_attention[node_name] = {
            label: float(score)
            for label, score in zip(input_labels[node_name], scores, strict=True)
        }
    return np.concatenate(focus_rows, axis=0), input_attention


def node_input_labels(
    dataset: ForexPortfolioDataset,
    node_name: str,
    feature_count: int,
) -> list[str]:
    """Return readable labels for one node's input feature channels."""
    metadata = dataset.metadata
    if node_name == "portfolio_signal":
        currencies = list(metadata.get("currencies", []))
        if len(currencies) == feature_count:
            return currencies
        return [
            f"portfolio_feature_{idx + 1}"
            for idx in range(feature_count)
        ]

    symbols = list(metadata.get("node_symbols", {}).get(node_name, []))
    if len(symbols) == feature_count:
        return symbols
    return [
        f"input_{idx + 1}"
        for idx in range(feature_count)
    ]


def node_display_name(node_name: str) -> str:
    """Pretty label for node names."""
    return node_name.replace("_", "\n")


def node_position_map(node_names: list[str]) -> dict[str, tuple[float, float]]:
    """Create a stable semantic layout for the heterogeneous graph."""
    positions: dict[str, tuple[float, float]] = {}
    positions["portfolio_signal"] = (0.0, 0.0)
    category_slots = {
        "fx": (-3.5, 0.0),
        "bond": (2.8, 1.6),
        "commodity": (2.2, -1.8),
        "equity": (-2.2, -2.1),
        "macro": (-2.0, 2.2),
    }
    counters = {key: 0 for key in category_slots}
    for node_name in node_names:
        if node_name == "portfolio_signal":
            continue
        if "fx" in node_name:
            key = "fx"
        elif node_name.startswith("generic_latent_"):
            key = "macro"
        elif "bond" in node_name or "treasury" in node_name:
            key = "bond"
        elif "commodity" in node_name:
            key = "commodity"
        elif "equity" in node_name:
            key = "equity"
        else:
            key = "macro"
        base_x, base_y = category_slots[key]
        offset = counters[key]
        counters[key] += 1
        if node_name.startswith("generic_latent_"):
            positions[node_name] = (-0.6 + 0.6 * offset, 2.8)
        else:
            positions[node_name] = (
                base_x + 0.9 * (offset % 3),
                base_y - 0.9 * (offset // 3),
            )
    return enforce_minimum_node_distance(positions=positions, min_distance=1.55)


def enforce_minimum_node_distance(
    positions: dict[str, tuple[float, float]],
    min_distance: float,
    iterations: int = 80,
) -> dict[str, tuple[float, float]]:
    """Push nodes apart until they satisfy a minimum pairwise distance."""
    adjusted = {
        node_name: np.array(coordinates, dtype=np.float64)
        for node_name, coordinates in positions.items()
    }
    node_names = list(adjusted)
    for _ in range(iterations):
        moved = False
        for idx, source in enumerate(node_names):
            for target in node_names[idx + 1 :]:
                delta = adjusted[target] - adjusted[source]
                distance = float(np.linalg.norm(delta))
                if distance >= min_distance:
                    continue
                moved = True
                if distance <= 1.0e-8:
                    direction = np.array([1.0, 0.0], dtype=np.float64)
                else:
                    direction = delta / distance
                correction = 0.5 * (min_distance - distance) * direction
                adjusted[source] -= correction
                adjusted[target] += correction
        if not moved:
            break
    return {
        node_name: (float(coords[0]), float(coords[1]))
        for node_name, coords in adjusted.items()
    }


def build_graph(node_names: list[str]) -> nx.DiGraph:
    """Build the same directed graph topology used by HTGNN."""
    graph = nx.DiGraph()
    graph.add_nodes_from(node_names)
    for source in node_names:
        for target in node_names:
            if source == target:
                continue
            if (
                source.startswith("generic_latent_")
                or target.startswith("generic_latent_")
                or source == "portfolio_signal"
                or target == "portfolio_signal"
            ):
                graph.add_edge(
                    source,
                    target,
                    relation=classify_relation(source, target),
                )
    return graph


def _bbox_circle_overlap(
    bbox: Any,
    center_x: float,
    center_y: float,
    radius: float,
) -> bool:
    """Return whether a display-space box overlaps a display-space circle."""
    closest_x = min(max(center_x, bbox.x0), bbox.x1)
    closest_y = min(max(center_y, bbox.y0), bbox.y1)
    return (closest_x - center_x) ** 2 + (closest_y - center_y) ** 2 <= radius**2


def _candidate_label_offsets(radius_points: float) -> list[tuple[float, float]]:
    """Return display-space candidate label offsets around a node."""
    base_distance = max(radius_points + 8.0, 26.0)
    angles = [
        0.0,
        math.pi,
        math.pi / 2.0,
        -math.pi / 2.0,
        math.pi / 4.0,
        3.0 * math.pi / 4.0,
        -math.pi / 4.0,
        -3.0 * math.pi / 4.0,
        math.pi / 6.0,
        5.0 * math.pi / 6.0,
        -math.pi / 6.0,
        -5.0 * math.pi / 6.0,
    ]
    offsets: list[tuple[float, float]] = []
    for ring_scale in (1.0, 1.18, 1.4, 1.7, 2.1, 2.6, 3.2, 3.9, 4.8, 5.9):
        distance = base_distance * ring_scale
        offsets.extend(
            (distance * math.cos(angle), distance * math.sin(angle))
            for angle in angles
        )
    return offsets


def _place_node_label(
    ax: Any,
    renderer: Any,
    label: str,
    node_center_display: np.ndarray,
    node_radius_pixels: float,
    node_obstacles: list[tuple[float, float, float]],
    placed_label_boxes: list[Any],
) -> Any:
    """Place one node label without overlapping nodes or previous labels."""
    dpi_scale = ax.figure.dpi / 72.0
    candidates = _candidate_label_offsets(
        radius_points=node_radius_pixels / dpi_scale
    )
    best_text = None
    best_penalty = float("inf")

    for offset_x_points, offset_y_points in candidates:
        offset_display = np.array(
            [
                offset_x_points * dpi_scale,
                offset_y_points * dpi_scale,
            ],
            dtype=np.float64,
        )
        label_display = node_center_display + offset_display
        label_data = ax.transData.inverted().transform(label_display)
        horizontal_alignment = "center"
        if offset_x_points > 8.0:
            horizontal_alignment = "left"
        elif offset_x_points < -8.0:
            horizontal_alignment = "right"

        text = ax.text(
            float(label_data[0]),
            float(label_data[1]),
            label,
            ha=horizontal_alignment,
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#0f172a",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": (1.0, 1.0, 1.0, 0.82),
                "edgecolor": "#cbd5e1",
                "linewidth": 0.35,
            },
        )
        bbox = text.get_window_extent(renderer=renderer).expanded(1.04, 1.10)
        label_overlap_count = sum(
            1 for placed_box in placed_label_boxes if bbox.overlaps(placed_box)
        )
        node_overlap_count = sum(
            1
            for center_x, center_y, radius in node_obstacles
            if _bbox_circle_overlap(
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
            )
        )
        if label_overlap_count == 0 and node_overlap_count == 0:
            if best_text is not None:
                best_text.remove()
            placed_label_boxes.append(bbox)
            return text

        penalty = (
            1000.0 * label_overlap_count
            + 1000.0 * node_overlap_count
            + float(np.linalg.norm(offset_display))
        )
        if penalty < best_penalty:
            if best_text is not None:
                best_text.remove()
            best_text = text
            best_penalty = penalty
        else:
            text.remove()

    if best_text is not None:
        best_text.remove()
    raise RuntimeError(
        f"Could not place node label without overlap: {label!r}."
    )


def plot_focus_graph(
    summary: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot the heterogeneous graph with node colors from focus strength."""
    node_names = list(summary)
    graph = build_graph(node_names=node_names)
    positions = node_position_map(node_names=node_names)
    values = np.array([summary[node]["mean"] for node in node_names], dtype=float)
    node_sizes = 1400 + 9000 * values
    color_norm = plt.matplotlib.colors.Normalize(vmin=0.0, vmax=0.10, clip=True)
    value_by_node = {
        node_name: float(value)
        for node_name, value in zip(node_names, values, strict=True)
    }
    size_by_node = {
        node_name: float(node_size)
        for node_name, node_size in zip(node_names, node_sizes, strict=True)
    }
    generic_node_names = [
        node_name
        for node_name in node_names
        if node_name.startswith("generic_latent_")
    ]
    observed_node_names = [
        node_name
        for node_name in node_names
        if not node_name.startswith("generic_latent_")
    ]
    generic_edges = [
        (source, target)
        for source, target in graph.edges()
        if source.startswith("generic_latent_") or target.startswith("generic_latent_")
    ]
    portfolio_out_edges = [
        (source, target)
        for source, target in graph.edges()
        if source == "portfolio_signal" and (source, target) not in generic_edges
    ]
    portfolio_in_edges = [
        (source, target)
        for source, target in graph.edges()
        if target == "portfolio_signal" and (source, target) not in generic_edges
    ]
    fx_to_portfolio_edges = [
        (source, target)
        for source, target in portfolio_in_edges
        if graph.edges[source, target].get("relation") == "fx_to_portfolio"
    ]
    other_portfolio_in_edges = [
        edge for edge in portfolio_in_edges if edge not in fx_to_portfolio_edges
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=generic_edges,
        ax=ax,
        edge_color="#94a3b8",
        arrows=True,
        alpha=0.30,
        width=0.9,
        style="dashed",
        arrowstyle="-|>",
        arrowsize=9,
        connectionstyle="arc3,rad=0.18",
    )
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=portfolio_out_edges,
        ax=ax,
        edge_color="#94a3b8",
        arrows=True,
        alpha=0.30,
        width=1.0,
        arrowstyle="-|>",
        arrowsize=11,
        connectionstyle="arc3,rad=-0.12",
    )
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=other_portfolio_in_edges,
        ax=ax,
        edge_color="#64748b",
        arrows=True,
        alpha=0.58,
        width=1.55,
        arrowstyle="-|>",
        arrowsize=14,
        connectionstyle="arc3,rad=0.12",
    )
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=fx_to_portfolio_edges,
        ax=ax,
        edge_color="#0f766e",
        arrows=True,
        alpha=0.85,
        width=2.0,
        arrowstyle="-|>",
        arrowsize=16,
        connectionstyle="arc3,rad=0.12",
    )
    if observed_node_names:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=observed_node_names,
            ax=ax,
            node_color=[value_by_node[node] for node in observed_node_names],
            cmap=BLUE_RED_CMAP,
            vmin=color_norm.vmin,
            vmax=color_norm.vmax,
            node_size=[size_by_node[node] for node in observed_node_names],
            linewidths=1.0,
            edgecolors="#0f172a",
        )
    if generic_node_names:
        generic_nodes = nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=generic_node_names,
            ax=ax,
            node_color=[value_by_node[node] for node in generic_node_names],
            cmap=BLUE_RED_CMAP,
            vmin=color_norm.vmin,
            vmax=color_norm.vmax,
            node_size=[size_by_node[node] for node in generic_node_names],
            linewidths=2.1,
            edgecolors="#0f172a",
        )
        generic_nodes.set_linestyle("dashed")
    scalar_map = plt.matplotlib.cm.ScalarMappable(
        norm=color_norm,
        cmap=BLUE_RED_CMAP,
    )
    scalar_map.set_array(values)
    colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Mean focus share (red >= 10%)", rotation=90)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi_scale = fig.dpi / 72.0
    node_obstacles = []
    node_display_centers: dict[str, np.ndarray] = {}
    node_radius_pixels: dict[str, float] = {}
    for node_name, node_size in zip(node_names, node_sizes, strict=True):
        display_center = ax.transData.transform(positions[node_name])
        radius_pixels = math.sqrt(float(node_size) / math.pi) * dpi_scale + 3.0
        node_display_centers[node_name] = display_center
        node_radius_pixels[node_name] = radius_pixels
        node_obstacles.append(
            (
                float(display_center[0]),
                float(display_center[1]),
                radius_pixels,
            )
        )

    placed_label_boxes: list[Any] = [
        colorbar.ax.get_window_extent(renderer=renderer).expanded(1.08, 1.08)
    ]
    label_order = sorted(
        node_names,
        key=lambda name: summary[name]["mean"],
        reverse=True,
    )
    for node_name in label_order:
        importance_pct = 100.0 * summary[node_name]["mean"]
        label = f"{node_name}\n{importance_pct:.1f}%"
        _place_node_label(
            ax=ax,
            renderer=renderer,
            label=label,
            node_center_display=node_display_centers[node_name],
            node_radius_pixels=node_radius_pixels[node_name],
            node_obstacles=node_obstacles,
            placed_label_boxes=placed_label_boxes,
        )
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_focus_histogram(
    summary: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot mean node focus with a variance interval on each bar."""
    node_names = list(summary)
    mean_values = np.array([summary[name]["mean"] for name in node_names]) * 100.0
    variance_values = (
        np.array([summary[name]["variance"] for name in node_names]) * 100.0
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(node_names))
    ax.bar(
        x,
        mean_values,
        color="#4C78A8",
        edgecolor="#1e293b",
        linewidth=0.8,
        width=0.7,
    )
    ax.errorbar(
        x,
        mean_values,
        yerr=variance_values,
        fmt="none",
        ecolor="#f59e0b",
        elinewidth=1.8,
        capsize=4,
        label="Variance interval",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([node_display_name(name) for name in node_names], fontsize=9)
    ax.set_ylabel("Mean focus (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_node_input_attention_pies(
    summary: dict[str, dict[str, float]],
    input_attention: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot per-node input-channel attention decompositions as pie charts."""
    target_nodes = [
        node_name
        for node_name in summary
        if not node_name.startswith("generic_latent_")
    ]
    if not target_nodes:
        return

    num_cols = min(4, len(target_nodes))
    num_rows = int(math.ceil(len(target_nodes) / num_cols))
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(3.9 * num_cols, 4.05 * num_rows),
        squeeze=False,
    )
    color_map = plt.get_cmap("tab20")

    for axis_idx, target in enumerate(target_nodes):
        ax = axes[axis_idx // num_cols][axis_idx % num_cols]
        source_values = input_attention.get(target, {})
        items = sorted(
            (
                (source, value)
                for source, value in source_values.items()
                if value > 1.0e-10
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not items:
            ax.text(
                0.5,
                0.5,
                "No input\nattention",
                ha="center",
                va="center",
                fontsize=10,
                color="#64748b",
            )
            ax.axis("off")
            continue

        labels = [source.replace("_", " ") for source, _ in items]
        values = [value for _, value in items]
        colors = [color_map(idx % color_map.N) for idx in range(len(items))]
        wedges, _, _ = ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= 3.0 else "",
            pctdistance=0.72,
            textprops={"fontsize": 7},
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 0.8,
            },
        )
        legend_columns = 1
        if len(labels) > 16:
            legend_columns = 3
        elif len(labels) > 7:
            legend_columns = 2
        legend_labels = [
            f"{label}: {100.0 * value:.1f}%"
            for label, value in zip(labels, values, strict=True)
        ]
        ax.legend(
            wedges,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=legend_columns,
            frameon=False,
            fontsize=6,
            handlelength=1.0,
            columnspacing=0.8,
            labelspacing=0.35,
        )
        focus_pct = 100.0 * summary[target]["mean"]
        ax.set_title(
            f"{target.replace('_', ' ')}\nfocus {focus_pct:.1f}%",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )
        ax.axis("equal")

    for axis_idx in range(len(target_nodes), num_rows * num_cols):
        axes[axis_idx // num_cols][axis_idx % num_cols].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=4.2)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def attention_summary_payload(
    summary: dict[str, dict[str, float]],
    checkpoint_path: Path,
    input_attention: dict[str, dict[str, float]] | None = None,
    attribution_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable payload for console/debug output."""
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "attribution_method": "smooth_gradient_attribution",
        "nodes": summary,
    }
    if attribution_config is not None:
        payload["attribution_config"] = attribution_config
    if input_attention is not None:
        payload["node_input_attention"] = input_attention
    return payload


def attribution_config_payload(
    args: argparse.Namespace,
    mc_samples: int,
) -> dict[str, Any]:
    """Build a serializable attribution-configuration payload."""
    return {
        "gradient_steps": int(args.gradient_steps),
        "smooth_samples": int(args.smooth_samples),
        "smooth_noise_std": float(args.smooth_noise_std),
        "mc_samples": int(mc_samples),
    }
