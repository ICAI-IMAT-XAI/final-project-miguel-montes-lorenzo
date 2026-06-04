"""Visualize node focus for a trained HTGNN checkpoint."""

from __future__ import annotations

from src.analysis.common import (
    attention_summary_payload,
    attribution_config_payload,
    build_analysis_dir,
    compute_focus_attributions,
    load_graph_model_and_dataset,
    parse_focus_args,
    plot_focus_graph,
    plot_focus_histogram,
    plot_node_input_attention_pies,
    summarize_focus_scores,
)
from src.models.pointwise.HTGNN.model import HTGNNModel


def main() -> None:
    """Create graph and histogram focus visualizations for HTGNN."""
    args = parse_focus_args(description="Visualize HTGNN node focus.")
    model, _, dataset, device, checkpoint_path = load_graph_model_and_dataset(
        checkpoint=args.checkpoint,
        processed_dir=args.processed_dir,
        split=args.split,
        date_start=args.date_start,
        date_end=args.date_end,
    )
    if not isinstance(model, HTGNNModel) or model.__class__.__name__ != "HTGNNModel":
        raise TypeError("The requested checkpoint is not an HTGNN checkpoint.")

    _, output_dir = build_analysis_dir(checkpoint=checkpoint_path)
    focus_distribution, input_attention = compute_focus_attributions(
        model=model,
        dataset=dataset,
        device=device,
        node_names=model.node_names,
        batch_size=args.batch_size,
        mc_samples=1,
        gradient_steps=args.gradient_steps,
        smooth_samples=args.smooth_samples,
        smooth_noise_std=args.smooth_noise_std,
    )
    summary = summarize_focus_scores(
        focus_distribution=focus_distribution,
        node_names=model.node_names,
    )
    plot_focus_graph(
        summary=summary,
        output_path=output_dir / "node_focus_graph.svg",
        title="HTGNN Node Focus by Smooth Gradient Attribution",
    )
    plot_focus_histogram(
        summary=summary,
        output_path=output_dir / "node_focus_histogram.svg",
        title="HTGNN Node Focus Histogram",
    )
    plot_node_input_attention_pies(
        summary=summary,
        input_attention=input_attention,
        output_path=output_dir / "node_input_attention_pies.svg",
        title="HTGNN Mean Input Attribution by Focused Node",
    )
    print(
        attention_summary_payload(
            summary=summary,
            checkpoint_path=checkpoint_path,
            input_attention=input_attention,
            attribution_config=attribution_config_payload(args=args, mc_samples=1),
        )
    )


if __name__ == "__main__":
    main()
