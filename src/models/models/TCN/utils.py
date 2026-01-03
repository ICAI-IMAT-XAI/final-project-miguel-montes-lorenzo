from collections.abc import Sequence


def receptive_field(
    *,
    kernel_size: int,
    dilations: Sequence[int],
    blocks_per_dilation: int,
) -> int:
    """Compute the temporal receptive field (in timesteps) of a stacked TCN.

    This assumes each residual block has two causal conv layers with the same
    kernel size and dilation, and that dilations are repeated
    `blocks_per_dilation` times (common in TCN designs).

    Args:
        kernel_size: Convolution kernel size (K). Must be >= 2 for temporal gain.
        dilations: Dilation factors used across depth (e.g. [1, 2, 4, 8, ...]).
        blocks_per_dilation: Number of residual blocks per dilation factor.

    Returns:
        Number of timesteps from the past that can influence the last output
        timestep (receptive field length).
    """
    assert kernel_size >= 1
    assert blocks_per_dilation >= 1
    assert len(dilations) >= 1

    # For one causal conv: adds (K-1)*d timesteps.
    # For two convs per block: adds 2*(K-1)*d.
    added: int = 0
    for d in dilations:
        added += blocks_per_dilation * (2 * (kernel_size - 1) * d)
    return 1 + added
