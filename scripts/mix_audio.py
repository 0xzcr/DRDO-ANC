import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


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


def calculate_power(audio):
    """Calculate mean-square signal power."""

    return np.mean(audio.astype(np.float64) ** 2)


def calculate_snr(clean, noise):
    """Calculate SNR between clean speech and noise."""

    clean_power = calculate_power(clean)
    noise_power = calculate_power(noise)

    if clean_power <= 0:
        raise ValueError("Clean signal has no measurable power.")

    if noise_power <= 0:
        raise ValueError("Noise signal has no measurable power.")

    return 10.0 * np.log10(clean_power / noise_power)


def scale_noise_to_snr(clean, noise, target_snr_db):
    """Scale noise to achieve the requested SNR."""

    clean_power = calculate_power(clean)
    noise_power = calculate_power(noise)

    target_noise_power = (
        clean_power / (10.0 ** (target_snr_db / 10.0))
    )

    scale = np.sqrt(
        target_noise_power / noise_power
    )

    return noise * scale


def create_mixture(clean, noise, target_snr_db):
    """Create a noisy mixture at the requested SNR."""

    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)

    noise = noise[: len(clean)]

    scaled_noise = scale_noise_to_snr(
        clean,
        noise,
        target_snr_db,
    )

    noisy = clean + scaled_noise

    actual_snr = calculate_snr(
        clean,
        scaled_noise,
    )

    return noisy, actual_snr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an SNR-controlled noisy speech mixture."
    )

    parser.add_argument(
        "--clean",
        type=Path,
        required=True,
        help="Clean speech WAV.",
    )

    parser.add_argument(
        "--noise",
        type=Path,
        required=True,
        help="Noise WAV.",
    )

    parser.add_argument(
        "--snr",
        type=float,
        required=True,
        help="Target SNR in dB.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output noisy WAV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    clean, clean_sr = load_audio(
        args.clean
    )

    noise, noise_sr = load_audio(
        args.noise
    )

    if clean_sr != noise_sr:
        raise ValueError(
            f"Sample rates do not match: "
            f"clean={clean_sr}, noise={noise_sr}"
        )

    noisy, actual_snr = create_mixture(
        clean,
        noise,
        args.snr,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        args.output,
        noisy,
        clean_sr,
    )

    print("=" * 60)
    print("DRDO-ANC | SNR-Controlled Audio Mixer")
    print("=" * 60)
    print(f"Clean:        {args.clean}")
    print(f"Noise:        {args.noise}")
    print(f"Output:       {args.output}")
    print(f"Sample rate:  {clean_sr} Hz")
    print(f"Target SNR:   {args.snr:.3f} dB")
    print(f"Actual SNR:   {actual_snr:.3f} dB")
    print(f"Duration:     {len(clean) / clean_sr:.3f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()