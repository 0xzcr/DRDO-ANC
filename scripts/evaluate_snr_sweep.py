from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly


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


def validate_audio(clean, estimate, clean_sr, estimate_sr):
    """Validate that two signals can be compared."""

    if clean_sr != estimate_sr:
        raise ValueError(
            f"Sample rates do not match: "
            f"clean={clean_sr}, estimate={estimate_sr}"
        )

    if len(clean) != len(estimate):
        raise ValueError(
            f"Audio lengths do not match: "
            f"clean={len(clean)}, estimate={len(estimate)}"
        )


def calculate_snr(clean, estimate):
    """Calculate SNR in dB."""

    noise = estimate - clean

    signal_power = np.sum(
        clean.astype(np.float64) ** 2
    )

    noise_power = np.sum(
        noise.astype(np.float64) ** 2
    )

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


def evaluate_pair(clean, estimate, sample_rate):
    """Calculate all objective metrics for one signal."""

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


def main():
    clean, clean_sr = load_audio(
        CLEAN_PATH
    )

    print("=" * 95)
    print("DRDO-ANC | DF3 SNR Sweep Evaluation")
    print("=" * 95)

    print(f"Clean reference: {CLEAN_PATH}")
    print(f"Sample rate:     {clean_sr} Hz")
    print(f"Duration:        {len(clean) / clean_sr:.3f} s")

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

        validate_audio(
            clean,
            noisy,
            clean_sr,
            noisy_sr,
        )

        validate_audio(
            clean,
            enhanced,
            clean_sr,
            enhanced_sr,
        )

        noisy_metrics = evaluate_pair(
            clean,
            noisy,
            clean_sr,
        )

        enhanced_metrics = evaluate_pair(
            clean,
            enhanced,
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