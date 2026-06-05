"""HTGNN-style auxiliary visualizations for XAI outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.common import plot_node_input_attention_pies


def input_attention_pie_charts(
    summary: dict[str, dict[str, float]],
    feature_attention: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save HTGNN-style small-multiple pies for every observed node."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_node_input_attention_pies(
        summary=summary,
        input_attention=feature_attention,
        output_path=output,
        title=title,
    )
    return output
