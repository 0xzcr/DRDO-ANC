import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly

from .delay import apply_evaluation_delay


def calculate_snr(clean, estimate):
    """Calculate SNR in dB using clean speech as reference."""

    noise = estimate - clean

    signal_power = np.sum(clean**2)
    noise_power = np.sum(noise**2)

    return 10.0 * np.log10(
        signal_power / (noise_power + 1e-12)
    )


def calculate_si_sdr(reference, estimate):
    """Calculate scale-invariant SDR in dB."""

    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    reference_energy = np.sum(reference**2)

    if reference_energy < 1e-12:
        raise ValueError(
            "Reference signal has almost no energy."
        )

    scale = (
        np.sum(estimate * reference)
        / reference_energy
    )

    target = scale * reference
    noise = estimate - target

    target_energy = np.sum(target**2)
    noise_energy = np.sum(noise**2)

    return 10.0 * np.log10(
        target_energy / (noise_energy + 1e-12)
    )


def calculate_stoi(clean, estimate, sample_rate):
    """Calculate STOI."""

    return stoi(
        clean,
        estimate,
        sample_rate,
        extended=False,
    )


def calculate_pesq(clean, estimate, sample_rate):
    """Calculate wideband PESQ at 16 kHz."""

    target_sample_rate = 16_000

    if sample_rate != target_sample_rate:
        clean = resample_poly(
            clean,
            target_sample_rate,
            sample_rate,
        )

        estimate = resample_poly(
            estimate,
            target_sample_rate,
            sample_rate,
        )

    return pesq(
        target_sample_rate,
        clean,
        estimate,
        "wb",
    )


def evaluate_pair(
    clean: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """Calculate all objective metrics for one signal pair."""

    return {
        "snr": calculate_snr(
            clean,
            estimate,
        ),
        "si_sdr": calculate_si_sdr(
            clean,
            estimate,
        ),
        "stoi": calculate_stoi(
            clean,
            estimate,
            sample_rate,
        ),
        "pesq": calculate_pesq(
            clean,
            estimate,
            sample_rate,
        ),
    }


def evaluate_model(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    delay_samples: int = 0,
) -> dict[str, float]:
    """Evaluate noisy and enhanced outputs against a clean reference."""

    clean_aligned, noisy_aligned, enhanced_aligned = (
        apply_evaluation_delay(
            clean,
            noisy,
            enhanced,
            delay_samples,
        )
    )

    noisy_metrics = evaluate_pair(
        clean_aligned,
        noisy_aligned,
        sample_rate,
    )
    enhanced_metrics = evaluate_pair(
        clean_aligned,
        enhanced_aligned,
        sample_rate,
    )

    return {
        "delay_samples": delay_samples,
        "overlap_samples": len(clean_aligned),
        "noisy_snr": noisy_metrics["snr"],
        "enhanced_snr": enhanced_metrics["snr"],
        "noisy_si_sdr": noisy_metrics["si_sdr"],
        "enhanced_si_sdr": enhanced_metrics["si_sdr"],
        "noisy_stoi": noisy_metrics["stoi"],
        "enhanced_stoi": enhanced_metrics["stoi"],
        "noisy_pesq": noisy_metrics["pesq"],
        "enhanced_pesq": enhanced_metrics["pesq"],
    }
