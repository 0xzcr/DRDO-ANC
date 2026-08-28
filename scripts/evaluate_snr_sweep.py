import argparse
from pathlib import Path

import soundfile as sf

from evaluate import (
    apply_evaluation_delay,
    evaluate_pair,
    format_delay_compensation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "clean_freesound_33711.wav"
)

INPUT_DIR = PROJECT_ROOT / "data" / "generated"
ENHANCED_DIR = PROJECT_ROOT / "data" / "enhanced"

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


def load_audio(path: Path):
    """Load mono float32 WAV audio."""

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


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DeepFilterNet3 across an SNR sweep."
        )
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


def main():
    args = parse_args()

    clean, clean_sr = load_audio(
        CLEAN_PATH
    )

    print("=" * 95)
    print("DRDO-ANC | DF3 SNR Sweep Evaluation")
    print("=" * 95)

    print(f"Clean reference: {CLEAN_PATH}")
    print(f"Sample rate:     {clean_sr} Hz")
    print(f"Duration:        {len(clean) / clean_sr:.3f} s")
    print(
        format_delay_compensation(
            args.delay_samples,
            clean_sr,
        )
    )

    results = []

    for snr_db in SNR_LEVELS:
        noisy_path = (
            INPUT_DIR
            / (
                "clean_freesound_33711_"
                "noise_573577_"
                f"snr{snr_db}.wav"
            )
        )

        enhanced_path = (
            ENHANCED_DIR
            / (
                "clean_freesound_33711_"
                "noise_573577_"
                f"snr{snr_db}_df3.wav"
            )
        )

        if not noisy_path.exists():
            raise FileNotFoundError(
                f"Noisy file not found: {noisy_path}"
            )

        if not enhanced_path.exists():
            raise FileNotFoundError(
                f"Enhanced file not found: {enhanced_path}"
            )

        noisy, noisy_sr = load_audio(
            noisy_path
        )

        enhanced, enhanced_sr = load_audio(
            enhanced_path
        )

        if clean_sr != noisy_sr or clean_sr != enhanced_sr:
            raise ValueError(
                f"Sample rates do not match for SNR {snr_db}."
            )

        if not (
            len(clean) == len(noisy) == len(enhanced)
        ):
            raise ValueError(
                f"Audio lengths do not match for SNR {snr_db}."
            )

        clean_aligned, noisy_aligned, enhanced_aligned = (
            apply_evaluation_delay(
                clean,
                noisy,
                enhanced,
                args.delay_samples,
            )
        )

        noisy_metrics = evaluate_pair(
            clean_aligned,
            noisy_aligned,
            clean_sr,
        )

        enhanced_metrics = evaluate_pair(
            clean_aligned,
            enhanced_aligned,
            clean_sr,
        )

        results.append(
            {
                "target_snr": snr_db,
                "input_snr": noisy_metrics["snr"],
                "enhanced_snr": enhanced_metrics["snr"],
                "snr_improvement": (
                    enhanced_metrics["snr"]
                    - noisy_metrics["snr"]
                ),
                "noisy_si_sdr": noisy_metrics["si_sdr"],
                "enhanced_si_sdr": enhanced_metrics["si_sdr"],
                "si_sdr_improvement": (
                    enhanced_metrics["si_sdr"]
                    - noisy_metrics["si_sdr"]
                ),
                "noisy_stoi": noisy_metrics["stoi"],
                "enhanced_stoi": enhanced_metrics["stoi"],
                "stoi_improvement": (
                    enhanced_metrics["stoi"]
                    - noisy_metrics["stoi"]
                ),
                "noisy_pesq": noisy_metrics["pesq"],
                "enhanced_pesq": enhanced_metrics["pesq"],
                "pesq_improvement": (
                    enhanced_metrics["pesq"]
                    - noisy_metrics["pesq"]
                ),
            }
        )

        print(
            f"Evaluated SNR {snr_db:+d} dB"
        )

    print()
    print("=" * 95)
    print("DF3 SNR SWEEP RESULTS")
    print("=" * 95)

    print()
    print(
        f"{'Input':>8}"
        f"{'Out SNR':>10}"
        f"{'ΔSNR':>10}"
        f"{'SI-SDR':>10}"
        f"{'STOI':>10}"
        f"{'PESQ':>10}"
    )

    print("-" * 60)

    for result in results:
        print(
            f"{result['target_snr']:>+7.0f}"
            f"{result['enhanced_snr']:>10.3f}"
            f"{result['snr_improvement']:>10.3f}"
            f"{result['enhanced_si_sdr']:>10.3f}"
            f"{result['enhanced_stoi']:>10.4f}"
            f"{result['enhanced_pesq']:>10.4f}"
        )

    print("=" * 95)


if __name__ == "__main__":
    main()
