import numpy as np


def format_delay_compensation(
    delay_samples: int,
    sample_rate: int,
) -> str:
    """Format the delay-compensation banner line."""

    delay_ms = delay_samples / sample_rate * 1000.0

    return (
        f"Delay compensation: {delay_samples} samples "
        f"({delay_ms:.3f} ms)"
    )


def apply_evaluation_delay(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    delay_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align clean, noisy, and enhanced for objective evaluation.

    ``delay_samples`` is the known algorithmic delay of the enhanced
    output relative to the clean/noisy timeline. The first
    ``delay_samples`` enhanced samples are dropped; clean and noisy are
    truncated to the same overlap length.
    """

    if delay_samples < 0:
        raise ValueError(
            f"delay_samples must be >= 0, got {delay_samples}"
        )

    overlap_length = min(
        len(clean),
        len(noisy),
        len(enhanced) - delay_samples,
    )

    if overlap_length <= 0:
        raise ValueError(
            "No overlapping audio after applying "
            f"delay_samples={delay_samples}."
        )

    clean_aligned = clean[:overlap_length]
    noisy_aligned = noisy[:overlap_length]
    enhanced_aligned = enhanced[
        delay_samples : delay_samples + overlap_length
    ]

    return clean_aligned, noisy_aligned, enhanced_aligned
