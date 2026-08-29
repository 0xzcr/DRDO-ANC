import numpy as np


def calculate_power(audio: np.ndarray) -> float:
    """Calculate mean-square signal power."""

    return float(np.mean(audio.astype(np.float64) ** 2))


def calculate_snr(
    clean: np.ndarray,
    noise: np.ndarray,
) -> float:
    """Calculate SNR in dB between clean speech and noise."""

    clean_power = calculate_power(clean)
    noise_power = calculate_power(noise)

    if clean_power <= 0:
        raise ValueError("Clean signal has no measurable power.")

    if noise_power <= 0:
        raise ValueError("Noise signal has no measurable power.")

    return 10.0 * np.log10(clean_power / noise_power)


def scale_noise_to_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """Scale noise to achieve the requested SNR against clean."""

    clean_power = calculate_power(clean)
    noise_power = calculate_power(noise)

    target_noise_power = (
        clean_power / (10.0 ** (target_snr_db / 10.0))
    )

    scale = np.sqrt(target_noise_power / noise_power)

    return noise * scale


def align_noise_to_clean_length(
    noise: np.ndarray,
    clean_length: int,
    mixing_seed: int,
) -> np.ndarray:
    """
    Align noise to the clean length using deterministic crop or repeat.

    If noise is longer than clean, crop using a start offset derived from
    ``mixing_seed``. If noise is shorter, tile cyclically and crop using a
    phase offset derived from ``mixing_seed``.
    """

    if clean_length <= 0:
        raise ValueError("clean_length must be positive.")

    noise = noise.astype(np.float32, copy=False)

    if len(noise) == clean_length:
        return noise.copy()

    if len(noise) > clean_length:
        max_start = len(noise) - clean_length
        start = mixing_seed % (max_start + 1)
        return noise[start : start + clean_length].copy()

    if len(noise) == 0:
        raise ValueError("Noise signal is empty.")

    repeats = int(np.ceil(clean_length / len(noise)))
    tiled = np.tile(noise, repeats)
    start = mixing_seed % len(noise)
    aligned = tiled[start : start + clean_length]

    if len(aligned) < clean_length:
        aligned = np.concatenate(
            [
                aligned,
                tiled[: clean_length - len(aligned)],
            ]
        )

    return aligned.astype(np.float32, copy=False)


def create_mixture(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
    mixing_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Create a noisy mixture at the requested SNR.

    Returns ``(noisy, scaled_noise, achieved_snr_db)``. The clean waveform is
    not modified.
    """

    aligned_noise = align_noise_to_clean_length(
        noise,
        len(clean),
        mixing_seed,
    )

    scaled_noise = scale_noise_to_snr(
        clean,
        aligned_noise,
        target_snr_db,
    )

    noisy = clean + scaled_noise

    achieved_snr = calculate_snr(
        clean,
        scaled_noise,
    )

    return noisy, scaled_noise, achieved_snr
