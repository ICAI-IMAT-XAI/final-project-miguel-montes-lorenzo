"""HTGNN-style graph plotting for XAI node scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.analysis.common import build_graph, node_position_map


BLUE_RED_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "xai_blue_red",
    ["#1d4ed8", "#93c5fd", "#fca5a5", "#dc2626"],
)


def plot_graph_scores(
    scores: dict[str, float],
    output_path: str | Path,
    title: str,
) -> Path:
    """Draw node scores using the existing HTGNN graph layout style."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    node_names = list(scores)
    graph = build_graph(node_names=node_names)
    positions = node_position_map(node_names=node_names)
    values = np.array([float(scores[node]) for node in node_names], dtype=float)
    max_abs = float(max(np.max(np.abs(values)), 1.0e-8))
    sizes = 1100 + 8200 * (np.abs(values) / max_abs)
    fig, ax = plt.subplots(figsize=(11, 8))
    generic_edges = [
        (source, target)
        for source, target in graph.edges()
        if source.startswith("generic_latent_") or target.startswith("generic_latent_")
    ]
    other_edges = [edge for edge in graph.edges() if edge not in generic_edges]
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=generic_edges,
        ax=ax,
        edge_color="#94a3b8",
        arrows=True,
        alpha=0.25,
        width=0.9,
        style="dashed",
        arrowstyle="-|>",
        arrowsize=9,
        connectionstyle="arc3,rad=0.18",
    )
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=other_edges,
        ax=ax,
        edge_color="#64748b",
        arrows=True,
        alpha=0.45,
        width=1.2,
        arrowstyle="-|>",
        arrowsize=12,
        connectionstyle="arc3,rad=0.12",
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=node_names,
        node_color=values,
        cmap=BLUE_RED_CMAP,
        vmin=-max_abs,
        vmax=max_abs,
        node_size=sizes,
        linewidths=1.0,
        edgecolors="#0f172a",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos=positions,
        labels={node: node.replace("_", "\n") for node in node_names},
        font_size=8,
        font_weight="bold",
        font_color="#0f172a",
        ax=ax,
    )
    colorbar = fig.colorbar(nodes, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Signed importance")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)
    return output

