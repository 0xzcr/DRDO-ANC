from pathlib import Path

from mix_audio import create_mixture, load_audio
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "clean_freesound_33711.wav"
)

NOISE_PATH = (
    PROJECT_ROOT
    / "data"
    / "noise"
    / "noise_freesound_573577.wav"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


def main():
    clean, clean_sr = load_audio(CLEAN_PATH)
    noise, noise_sr = load_audio(NOISE_PATH)

    if clean_sr != noise_sr:
        raise ValueError(
            f"Sample rates do not match: "
            f"clean={clean_sr}, noise={noise_sr}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DRDO-ANC | Controlled SNR Sweep")
    print("=" * 70)

    print(f"Clean:       {CLEAN_PATH}")
    print(f"Noise:       {NOISE_PATH}")
    print(f"Sample rate: {clean_sr} Hz")
    print(f"SNR levels:  {SNR_LEVELS}")

    print()

    for snr_db in SNR_LEVELS:
        output_path = (
            OUTPUT_DIR
            / (
                "clean_freesound_33711_"
                "noise_573577_"
                f"snr{snr_db}.wav"
            )
        )

        noisy, actual_snr = create_mixture(
            clean,
            noise,
            snr_db,
        )

        sf.write(
            output_path,
            noisy,
            clean_sr,
        )

        print(
            f"Target: {snr_db:>4} dB | "
            f"Actual: {actual_snr:>8.3f} dB | "
            f"{output_path.name}"
        )

    print()
    print("SNR sweep complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()