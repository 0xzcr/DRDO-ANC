from pathlib import Path

import numpy as np

from evaluate import (
    apply_evaluation_delay,
    evaluate_model,
    load_audio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "clean_freesound_33711.wav"
)

NOISY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "clean_freesound_33711_noise_573577_snr0.wav"
)

STREAMING_PATH = (
    PROJECT_ROOT
    / "data"
    / "enhanced"
    / "clean_freesound_33711_noise_573577_snr0_streaming_df3.wav"
)


def test_apply_evaluation_delay_shapes() -> None:
    """Aligned segments must have equal length."""

    clean = np.arange(10, dtype=np.float32)
    noisy = np.arange(10, dtype=np.float32) + 100
    enhanced = np.arange(10, dtype=np.float32) + 200

    clean_aligned, noisy_aligned, enhanced_aligned = (
        apply_evaluation_delay(
            clean,
            noisy,
            enhanced,
            delay_samples=3,
        )
    )

    assert len(clean_aligned) == 7
    assert len(noisy_aligned) == 7
    assert len(enhanced_aligned) == 7

    np.testing.assert_array_equal(
        clean_aligned,
        np.arange(7, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        noisy_aligned,
        np.arange(7, dtype=np.float32) + 100,
    )
    np.testing.assert_array_equal(
        enhanced_aligned,
        np.arange(3, 10, dtype=np.float32) + 200,
    )


def test_delay_samples_zero_matches_alignment_investigation() -> None:
    """Delay 0 should reproduce the known misaligned metrics."""

    clean, sample_rate = load_audio(CLEAN_PATH)
    noisy, _ = load_audio(NOISY_PATH)
    streaming, _ = load_audio(STREAMING_PATH)

    metrics = evaluate_model(
        clean,
        noisy,
        streaming,
        sample_rate,
        delay_samples=0,
    )

    assert metrics["overlap_samples"] == len(clean)
    assert abs(metrics["enhanced_snr"] - (-3.140)) < 0.05
    assert abs(metrics["enhanced_si_sdr"] - (-42.130)) < 0.05
    assert abs(metrics["enhanced_stoi"] - 0.5043) < 0.01


def test_delay_samples_1440_matches_alignment_investigation() -> None:
    """Delay 1440 should reproduce the aligned streaming metrics."""

    clean, sample_rate = load_audio(CLEAN_PATH)
    noisy, _ = load_audio(NOISY_PATH)
    streaming, _ = load_audio(STREAMING_PATH)

    metrics = evaluate_model(
        clean,
        noisy,
        streaming,
        sample_rate,
        delay_samples=1440,
    )

    assert metrics["overlap_samples"] == len(clean) - 1440
    assert abs(metrics["enhanced_snr"] - 9.740) < 0.05
    assert abs(metrics["enhanced_si_sdr"] - 9.567) < 0.05
    assert abs(metrics["enhanced_stoi"] - 0.9747) < 0.01


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Evaluate Delay Compensation Tests")
    print("=" * 70)

    test_apply_evaluation_delay_shapes()
    print("PASS: apply_evaluation_delay shapes")

    test_delay_samples_zero_matches_alignment_investigation()
    print("PASS: delay_samples=0 streaming metrics")

    test_delay_samples_1440_matches_alignment_investigation()
    print("PASS: delay_samples=1440 streaming metrics")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
