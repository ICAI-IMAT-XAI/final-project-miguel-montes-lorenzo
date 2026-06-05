"""Matplotlib plotting helpers for SVG XAI artifacts."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator


def save_bar_plot(
    values: dict[str, float],
    output_path: str | Path,
    title: str,
    ylabel: str,
    top_n: int | None = None,
) -> Path:
    """Save a horizontal bar chart sorted by absolute value."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
    if top_n is not None:
        items = items[:top_n]
    if not items:
        items = [("none", 0.0)]
    labels = [item[0].replace("_", " ") for item in items][::-1]
    scores = np.array([item[1] for item in items], dtype=float)[::-1]
    colors = ["#dc2626" if score >= 0 else "#2563eb" for score in scores]
    height = max(4.0, 0.36 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(np.arange(len(labels)), scores, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#0f172a", linewidth=0.8)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
    return output


def save_line_plot(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    point_labels: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Save line plot for deletion/insertion curves."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted_columns: list[str] = []
    line_colors: dict[str, str] = {}
    for column in frame.columns:
        if column == "step":
            continue
        line = ax.plot(
            frame["step"],
            frame[column],
            marker="o",
            linewidth=2.0,
            label=column,
        )[0]
        plotted_columns.append(column)
        line_colors[column] = line.get_color()

    values = frame[plotted_columns].to_numpy(dtype=float) if plotted_columns else np.zeros((1, 1))
    max_value = float(np.nanmax(values))
    upper = max(max_value * 1.42, max_value + 0.10, 1.0e-6)
    ax.set_ylim(bottom=0.0, top=upper)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    if point_labels is not None:
        _place_point_labels(
            fig=fig,
            ax=ax,
            frame=frame,
            plotted_columns=plotted_columns,
            line_colors=line_colors,
            point_labels=point_labels,
        )
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output


def _place_point_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    frame: pd.DataFrame,
    plotted_columns: list[str],
    line_colors: dict[str, str],
    point_labels: dict[str, dict[int, str]],
) -> None:
    """Place point labels near markers without overlapping text or lines."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    line_samples = _line_display_samples(ax=ax, frame=frame, columns=plotted_columns)
    placed_boxes: list[Bbox] = []
    steps = [int(value) for value in frame["step"].tolist()]
    min_step = min(steps)
    max_step = max(steps)

    for column in plotted_columns:
        if column not in point_labels:
            continue
        for row in frame[["step", column]].itertuples(index=False):
            step = int(row.step)
            label = point_labels[column].get(step)
            if label is None:
                continue
            text = _place_single_point_label(
                ax=ax,
                renderer=renderer,
                label=label,
                x_value=float(row.step),
                y_value=float(getattr(row, column)),
                column=column,
                color=line_colors[column],
                min_step=min_step,
                max_step=max_step,
                placed_boxes=placed_boxes,
                line_samples=line_samples,
            )
            placed_boxes.append(
                text.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
            )


def _line_display_samples(
    ax: plt.Axes,
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Sample plotted line segments in display coordinates for collision checks."""
    samples: list[np.ndarray] = []
    x_values = frame["step"].to_numpy(dtype=float)
    for column in columns:
        y_values = frame[column].to_numpy(dtype=float)
        for idx in range(len(x_values) - 1):
            x_segment = np.linspace(x_values[idx], x_values[idx + 1], num=24)
            y_segment = np.linspace(y_values[idx], y_values[idx + 1], num=24)
            samples.append(np.column_stack([x_segment, y_segment]))
    if not samples:
        return np.empty((0, 2), dtype=float)
    return ax.transData.transform(np.vstack(samples))


def _candidate_offsets(
    column: str,
    step: int,
    min_step: int,
    max_step: int,
) -> list[tuple[int, int]]:
    """Return candidate label offsets in points, biased away from plot edges."""
    vertical_sign = 1 if column == "deletion" else -1
    if step == max_step:
        vertical_sign *= -1
    right_first = step != max_step
    horizontal_offsets = [10, 32, -32, 54, -54, 0, 74, -74]
    if not right_first:
        horizontal_offsets = [-10, -32, 32, -54, 54, 0, -74, 74]
    if step == min_step:
        horizontal_offsets = [abs(value) for value in horizontal_offsets]
    preferred_vertical_offsets = [
        16 * vertical_sign,
        28 * vertical_sign,
        40 * vertical_sign,
        54 * vertical_sign,
        70 * vertical_sign,
        88 * vertical_sign,
    ]
    fallback_vertical_offsets = [
        -24 * vertical_sign,
        -40 * vertical_sign,
        -58 * vertical_sign,
        -76 * vertical_sign,
    ]
    return [
        (horizontal_offset, vertical_offset)
        for vertical_offset in [*preferred_vertical_offsets, *fallback_vertical_offsets]
        for horizontal_offset in horizontal_offsets
    ]


def _alignment_from_offset(offset_x: int, offset_y: int) -> tuple[str, str]:
    """Resolve text alignment from display offset."""
    if offset_x < 0:
        horizontal_alignment = "right"
    elif offset_x > 0:
        horizontal_alignment = "left"
    else:
        horizontal_alignment = "center"
    vertical_alignment = "bottom" if offset_y >= 0 else "top"
    return horizontal_alignment, vertical_alignment


def _place_single_point_label(
    ax: plt.Axes,
    renderer: object,
    label: str,
    x_value: float,
    y_value: float,
    column: str,
    color: str,
    min_step: int,
    max_step: int,
    placed_boxes: list[Bbox],
    line_samples: np.ndarray,
) -> object:
    """Place one label using the first candidate with no detected collision."""
    ax.figure.canvas.draw()
    axes_box = ax.get_window_extent(renderer=renderer)
    for offset_x, offset_y in _candidate_offsets(
        column=column,
        step=int(x_value),
        min_step=min_step,
        max_step=max_step,
    ):
        horizontal_alignment, vertical_alignment = _alignment_from_offset(
            offset_x=offset_x,
            offset_y=offset_y,
        )
        text = ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7,
            color=color,
            rotation=7,
            ha=horizontal_alignment,
            va=vertical_alignment,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
        bbox = text.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
        inside_axes = _bbox_inside(container=axes_box, bbox=bbox)
        label_overlap = any(bbox.overlaps(placed_box) for placed_box in placed_boxes)
        line_overlap = _bbox_hits_line_samples(bbox=bbox, line_samples=line_samples)
        if inside_axes and not label_overlap and not line_overlap:
            return text
        text.remove()
    raise RuntimeError(f"Could not place point label: {label!r}.")


def _bbox_inside(container: Bbox, bbox: Bbox) -> bool:
    """Return whether one display-space box is fully inside another."""
    return (
        bbox.x0 >= container.x0
        and bbox.x1 <= container.x1
        and bbox.y0 >= container.y0
        and bbox.y1 <= container.y1
    )


def _bbox_hits_line_samples(bbox: Bbox, line_samples: np.ndarray) -> bool:
    """Return whether sampled line points collide with a text box."""
    if line_samples.size == 0:
        return False
    expanded = bbox.expanded(1.03, 1.12)
    x_values = line_samples[:, 0]
    y_values = line_samples[:, 1]
    hits = (
        (x_values >= expanded.x0)
        & (x_values <= expanded.x1)
        & (y_values >= expanded.y0)
        & (y_values <= expanded.y1)
    )
    return bool(np.any(hits))


def save_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    output_path: str | Path,
    title: str,
    colorbar_label: str,
) -> Path:
    """Save a compact heatmap."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.5, len(x_labels) * 0.55), 5.6))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)), y_labels)
    ax.set_title(title, fontsize=13, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
    return output
