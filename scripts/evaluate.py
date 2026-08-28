from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = PROJECT_ROOT / "data" / "raw" / "clean_freesound_33711.wav"
NOISY_PATH = PROJECT_ROOT / "data" / "raw" / "noisy_snr0.wav"
ENHANCED_PATH = (
    PROJECT_ROOT / "data" / "enhanced" / "noisy_snr0_abstraction.wav"
)


def load_audio(path: Path):
    """Load a mono float32 WAV file."""
    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio in {path.name}, "
            f"got shape {audio.shape}"
        )

    return audio, sample_rate


def validate_audio(clean, noisy, enhanced, sample_rates):
    """Verify that all evaluation signals are compatible."""

    clean_sr, noisy_sr, enhanced_sr = sample_rates

    if not (clean_sr == noisy_sr == enhanced_sr):
        raise ValueError(
            f"Sample rates do not match: "
            f"clean={clean_sr}, noisy={noisy_sr}, enhanced={enhanced_sr}"
        )

    if not (len(clean) == len(noisy) == len(enhanced)):
        raise ValueError(
            f"Audio lengths do not match: "
            f"clean={len(clean)}, "
            f"noisy={len(noisy)}, "
            f"enhanced={len(enhanced)}"
        )


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
        raise ValueError("Reference signal has almost no energy.")

    scale = np.sum(estimate * reference) / reference_energy

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


def print_results(
    noisy_snr,
    enhanced_snr,
    noisy_si_sdr,
    enhanced_si_sdr,
    noisy_stoi,
    enhanced_stoi,
    noisy_pesq,
    enhanced_pesq,
):
    print()
    print("=" * 70)
    print("DRDO-ANC | Objective Evaluation")
    print("=" * 70)

    print()
    print(f"{'Metric':<20} {'Noisy':>15} {'Enhanced':>15} {'Change':>15}")
    print("-" * 70)

    print(
        f"{'SNR (dB)':<20}"
        f"{noisy_snr:>15.3f}"
        f"{enhanced_snr:>15.3f}"
        f"{enhanced_snr - noisy_snr:>15.3f}"
    )

    print(
        f"{'SI-SDR (dB)':<20}"
        f"{noisy_si_sdr:>15.3f}"
        f"{enhanced_si_sdr:>15.3f}"
        f"{enhanced_si_sdr - noisy_si_sdr:>15.3f}"
    )

    print(
        f"{'STOI':<20}"
        f"{noisy_stoi:>15.4f}"
        f"{enhanced_stoi:>15.4f}"
        f"{enhanced_stoi - noisy_stoi:>15.4f}"
    )

    print(
        f"{'PESQ':<20}"
        f"{noisy_pesq:>15.4f}"
        f"{enhanced_pesq:>15.4f}"
        f"{enhanced_pesq - noisy_pesq:>15.4f}"
    )

    print("=" * 70)

def main():
    print("=" * 70)
    print("DRDO-ANC | Objective Evaluation")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load signals
    # ---------------------------------------------------------

    clean, clean_sr = load_audio(CLEAN_PATH)
    noisy, noisy_sr = load_audio(NOISY_PATH)
    enhanced, enhanced_sr = load_audio(ENHANCED_PATH)

    validate_audio(
        clean,
        noisy,
        enhanced,
        (clean_sr, noisy_sr, enhanced_sr),
    )

    print(f"Clean:     {CLEAN_PATH.name}")
    print(f"Noisy:     {NOISY_PATH.name}")
    print(f"Enhanced:  {ENHANCED_PATH.name}")
    print(f"Sample rate: {clean_sr} Hz")
    print(f"Samples:     {len(clean)}")
    print(f"Duration:    {len(clean) / clean_sr:.3f} s")

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    print("\nCalculating metrics...")

    noisy_snr = calculate_snr(clean, noisy)
    enhanced_snr = calculate_snr(clean, enhanced)

    noisy_si_sdr = calculate_si_sdr(clean, noisy)
    enhanced_si_sdr = calculate_si_sdr(clean, enhanced)

    noisy_stoi = calculate_stoi(clean, noisy, clean_sr)
    enhanced_stoi = calculate_stoi(clean, enhanced, clean_sr)

    noisy_pesq = calculate_pesq(
        clean,
        noisy,
        clean_sr,
    )

    enhanced_pesq = calculate_pesq(
        clean,
        enhanced,
        clean_sr,
    )

    print_results(
        noisy_snr,
        enhanced_snr,
        noisy_si_sdr,
        enhanced_si_sdr,
        noisy_stoi,
        enhanced_stoi,
        noisy_pesq,
        enhanced_pesq,
    )


if __name__ == "__main__":
    main()