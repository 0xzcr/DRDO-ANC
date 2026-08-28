import argparse
from pathlib import Path

import soundfile as sf

from drdo_anc.evaluation import (
    apply_evaluation_delay,
    evaluate_model,
    evaluate_pair,
    format_delay_compensation,
    calculate_pesq,
    calculate_si_sdr,
    calculate_snr,
    calculate_stoi,
)


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

    parser.add_argument(
        "--delay-samples",
        type=int,
        default=0,
        help=(
            "Known algorithmic delay of the enhanced output in "
            "samples. Drops the first N enhanced samples before "
            "metric calculation."
        ),
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
    print(
        format_delay_compensation(
            args.delay_samples,
            clean_sr,
        )
    )

    print("\nCalculating metrics...")

    metrics = evaluate_model(
        clean,
        noisy,
        enhanced,
        clean_sr,
        delay_samples=args.delay_samples,
    )

    print(
        f"Aligned samples: {metrics['overlap_samples']} "
        f"({metrics['overlap_samples'] / clean_sr:.3f} s)"
    )

    print_results(
        model=args.model,
        noisy_snr=metrics["noisy_snr"],
        enhanced_snr=metrics["enhanced_snr"],
        noisy_si_sdr=metrics["noisy_si_sdr"],
        enhanced_si_sdr=metrics["enhanced_si_sdr"],
        noisy_stoi=metrics["noisy_stoi"],
        enhanced_stoi=metrics["enhanced_stoi"],
        noisy_pesq=metrics["noisy_pesq"],
        enhanced_pesq=metrics["enhanced_pesq"],
    )


if __name__ == "__main__":
    main()
