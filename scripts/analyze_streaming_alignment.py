from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "clean_freesound_33711.wav"
)

STREAMING_PATH = (
    PROJECT_ROOT
    / "data"
    / "enhanced"
    / "clean_freesound_33711_noise_573577_snr0_streaming_df3.wav"
)


def normalize(x: np.ndarray) -> np.ndarray:
    """Remove DC and normalize signal energy."""
    x = x.astype(np.float64)
    x = x - np.mean(x)

    energy = np.sqrt(np.sum(x ** 2))

    if energy < 1e-12:
        raise ValueError("Signal has almost no energy.")

    return x / energy


def correlation_at_offset(
    clean: np.ndarray,
    enhanced: np.ndarray,
    offset: int,
) -> float:
    """
    Calculate normalized correlation at a given offset.

    Positive offset means the enhanced signal starts later
    than the clean reference.
    """

    if offset >= 0:
        clean_start = offset
        enhanced_start = 0
    else:
        clean_start = 0
        enhanced_start = -offset

    available_clean = len(clean) - clean_start
    available_enhanced = len(enhanced) - enhanced_start

    length = min(
        available_clean,
        available_enhanced,
    )

    if length <= 0:
        return float("-inf")

    clean_segment = clean[
        clean_start : clean_start + length
    ]

    enhanced_segment = enhanced[
        enhanced_start : enhanced_start + length
    ]

    clean_segment = normalize(clean_segment)
    enhanced_segment = normalize(enhanced_segment)

    return float(
        np.dot(
            clean_segment,
            enhanced_segment,
        )
    )


def main():
    print("=" * 70)
    print("DRDO-ANC | Streaming Alignment Analysis")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load signals
    # ---------------------------------------------------------

    clean, clean_sr = sf.read(
        CLEAN_PATH,
        dtype="float32",
    )

    enhanced, enhanced_sr = sf.read(
        STREAMING_PATH,
        dtype="float32",
    )

    if clean.ndim != 1:
        raise ValueError(
            f"Expected mono clean audio, got {clean.shape}"
        )

    if enhanced.ndim != 1:
        raise ValueError(
            f"Expected mono streaming audio, got {enhanced.shape}"
        )

    if clean_sr != enhanced_sr:
        raise ValueError(
            f"Sample rates differ: "
            f"clean={clean_sr}, "
            f"enhanced={enhanced_sr}"
        )

    print(f"Sample rate:       {clean_sr} Hz")
    print(f"Clean samples:     {len(clean)}")
    print(f"Streaming samples: {len(enhanced)}")

    # ---------------------------------------------------------
    # Search offsets
    # ---------------------------------------------------------

    # Search ±100 ms.
    max_offset = int(
        clean_sr * 0.100
    )

    print(
        f"\nSearching offsets from "
        f"{-max_offset} to +{max_offset} samples..."
    )

    results = []

    for offset in range(
        -max_offset,
        max_offset + 1,
    ):
        correlation = correlation_at_offset(
            clean,
            enhanced,
            offset,
        )

        results.append(
            (offset, correlation)
        )

    best_offset, best_correlation = max(
        results,
        key=lambda x: x[1],
    )

    latency_ms = (
        best_offset
        / clean_sr
        * 1000
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALIGNMENT RESULT")
    print("=" * 70)

    print(f"Best offset:       {best_offset} samples")
    print(f"Estimated latency:  {latency_ms:.3f} ms")
    print(f"Correlation:       {best_correlation:.6f}")

    if best_offset > 0:
        print(
            "\nInterpretation:"
            "\nThe streaming output appears delayed "
            "relative to the clean reference."
        )
    elif best_offset < 0:
        print(
            "\nInterpretation:"
            "\nThe streaming output appears ahead "
            "of the clean reference."
        )
    else:
        print(
            "\nInterpretation:"
            "\nNo measurable sample offset was found."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()