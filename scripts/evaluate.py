import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_audio(path: Path):
    """Load a mono float32 WAV file."""

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
    )

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
            f"clean={clean_sr}, "
            f"noisy={noisy_sr}, "
            f"enhanced={enhanced_sr}"
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


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a speech-enhancement model."
    )

    parser.add_argument(
        "--clean",
        type=Path,
        required=True,
        help="Clean reference WAV file.",
    )

    parser.add_argument(
        "--noisy",
        type=Path,
        required=True,
        help="Noisy input WAV file.",
    )

    parser.add_argument(
        "--enhanced",
        type=Path,
        required=True,
        help="Enhanced output WAV file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name.",
    )

    return parser.parse_args()


def print_results(
    model,
    noisy_snr,
    enhanced_snr,
    noisy_si_sdr,
    enhanced_si_sdr,
    noisy_stoi,
    enhanced_stoi,
    noisy_pesq,
    enhanced_pesq,
):
    """Print evaluation results."""

    print()
    print("=" * 75)
    print("DRDO-ANC | Objective Evaluation")
    print("=" * 75)

    print(f"Model: {model}")

    print()
    print(
        f"{'Metric':<20}"
        f"{'Noisy':>15}"
        f"{'Enhanced':>15}"
        f"{'Change':>15}"
    )

    print("-" * 75)

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

    print("=" * 75)


def main():
    args = parse_args()

    clean_path = args.clean.resolve()
    noisy_path = args.noisy.resolve()
    enhanced_path = args.enhanced.resolve()

    print("=" * 75)
    print("DRDO-ANC | Objective Evaluation")
    print("=" * 75)

    clean, clean_sr = load_audio(clean_path)
    noisy, noisy_sr = load_audio(noisy_path)
    enhanced, enhanced_sr = load_audio(enhanced_path)

    validate_audio(
        clean,
        noisy,
        enhanced,
        (clean_sr, noisy_sr, enhanced_sr),
    )

    print(f"Model:        {args.model}")
    print(f"Clean:        {clean_path}")
    print(f"Noisy:        {noisy_path}")
    print(f"Enhanced:     {enhanced_path}")
    print(f"Sample rate:  {clean_sr} Hz")
    print(f"Samples:      {len(clean)}")
    print(f"Duration:     {len(clean) / clean_sr:.3f} s")

    print("\nCalculating metrics...")

    noisy_snr = calculate_snr(
        clean,
        noisy,
    )

    enhanced_snr = calculate_snr(
        clean,
        enhanced,
    )

    noisy_si_sdr = calculate_si_sdr(
        clean,
        noisy,
    )

    enhanced_si_sdr = calculate_si_sdr(
        clean,
        enhanced,
    )

    noisy_stoi = calculate_stoi(
        clean,
        noisy,
        clean_sr,
    )

    enhanced_stoi = calculate_stoi(
        clean,
        enhanced,
        clean_sr,
    )

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
        model=args.model,
        noisy_snr=noisy_snr,
        enhanced_snr=enhanced_snr,
        noisy_si_sdr=noisy_si_sdr,
        enhanced_si_sdr=enhanced_si_sdr,
        noisy_stoi=noisy_stoi,
        enhanced_stoi=enhanced_stoi,
        noisy_pesq=noisy_pesq,
        enhanced_pesq=enhanced_pesq,
    )


if __name__ == "__main__":
    main()