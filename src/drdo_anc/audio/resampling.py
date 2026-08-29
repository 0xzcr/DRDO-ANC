import math

import numpy as np
from scipy.signal import resample_poly


def resample_mono(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """
    Deterministically resample mono audio to a target sample rate.

    This helper is intended for explicit model-input boundaries. The dataset
    layer keeps native source rates; callers resample before enhancement.
    """

    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive integers.")

    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)

    gcd = math.gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = source_sample_rate // gcd

    resampled = resample_poly(
        audio,
        up,
        down,
    )

    return resampled.astype(np.float32, copy=False)
