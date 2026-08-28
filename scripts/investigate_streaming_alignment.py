"""
Temporary alignment investigation for native DF3 streaming output.

Determines whether poor SI-SDR/STOI metrics are primarily caused by
sample misalignment between the clean reference and streaming enhanced
audio.

Does not modify production code or audio files.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from evaluate import (
    calculate_si_sdr,
    calculate_snr,
    calculate_stoi,
)


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

MIN_OFFSET = -2880
MAX_OFFSET = 2880
OFFSET_STEP = 480


def align_segments(
    clean: np.ndarray,
    streaming: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping segments for a relative offset.

    Offset convention (matches analyze_streaming_alignment.py):
      offset > 0  -> streaming starts later than clean at equal indices
                     (compare clean[offset:] with streaming[:])
      offset < 0  -> streaming starts earlier than clean at equal indices
                     (compare clean[:] with streaming[-offset:])
    """

    if offset >= 0:
        clean_start = offset
        streaming_start = 0
    else:
        clean_start = 0
        streaming_start = -offset

    length = min(
        len(clean) - clean_start,
        len(streaming) - streaming_start,
    )

    if length <= 0:
        raise ValueError(
            f"No overlap for offset {offset}."
        )

    clean_segment = clean[
        clean_start : clean_start + length
    ]
    streaming_segment = streaming[
        streaming_start : streaming_start + length
    ]

    return clean_segment, streaming_segment


def correlation_coefficient(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Normalized correlation after DC removal and energy normalization."""

    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    ref_energy = np.sqrt(np.sum(reference**2))
    est_energy = np.sqrt(np.sum(estimate**2))

    if ref_energy < 1e-12 or est_energy < 1e-12:
        return float("nan")

    return float(
        np.dot(reference, estimate)
        / (ref_energy * est_energy)
    )


def evaluate_at_offset(
    clean: np.ndarray,
    streaming: np.ndarray,
    sample_rate: int,
    offset: int,
) -> dict:
    """Compute metrics on the aligned overlap at a given offset."""

    clean_seg, streaming_seg = align_segments(
        clean,
        streaming,
        offset,
    )

    return {
        "offset": offset,
        "overlap_samples": len(clean_seg),
        "overlap_seconds": len(clean_seg) / sample_rate,
        "correlation": correlation_coefficient(
            clean_seg,
            streaming_seg,
        ),
        "snr_db": calculate_snr(
            clean_seg,
            streaming_seg,
        ),
        "si_sdr_db": calculate_si_sdr(
            clean_seg,
            streaming_seg,
        ),
        "stoi": calculate_stoi(
            clean_seg,
            streaming_seg,
            sample_rate,
        ),
    }


def print_table(
    title: str,
    rows: list[dict],
    center_offset: int,
) -> None:
    """Print a compact table centered on a chosen offset."""

    print()
    print(title)
    print("-" * 88)
    print(
        f"{'Offset':>8}"
        f"{'ms':>10}"
        f"{'Overlap':>10}"
        f"{'Corr':>10}"
        f"{'SNR':>10}"
        f"{'SI-SDR':>10}"
        f"{'STOI':>10}"
    )
    print("-" * 88)

    for row in rows:
        marker = " <--" if row["offset"] == center_offset else ""
        print(
            f"{row['offset']:>8d}"
            f"{row['offset_ms']:>10.1f}"
            f"{row['overlap_samples']:>10d}"
            f"{row['correlation']:>10.4f}"
            f"{row['snr_db']:>10.3f}"
            f"{row['si_sdr_db']:>10.3f}"
            f"{row['stoi']:>10.4f}"
            f"{marker}"
        )

    print("-" * 88)


def main() -> None:
    print("=" * 88)
    print("DRDO-ANC | Streaming Alignment Metric Investigation")
    print("=" * 88)

    clean, sample_rate = sf.read(
        CLEAN_PATH,
        dtype="float32",
    )
    streaming, streaming_sr = sf.read(
        STREAMING_PATH,
        dtype="float32",
    )

    if clean.ndim != 1 or streaming.ndim != 1:
        raise ValueError("Expected mono WAV files.")

    if sample_rate != streaming_sr:
        raise ValueError(
            f"Sample-rate mismatch: "
            f"clean={sample_rate}, streaming={streaming_sr}"
        )

    print(f"Clean path:      {CLEAN_PATH}")
    print(f"Streaming path:  {STREAMING_PATH}")
    print(f"Sample rate:     {sample_rate} Hz")
    print(f"Clean samples:   {len(clean)}")
    print(f"Streaming samples: {len(streaming)}")
    print(
        f"Offset search:   {MIN_OFFSET} to {MAX_OFFSET} "
        f"in steps of {OFFSET_STEP}"
    )

    offsets = list(
        range(
            MIN_OFFSET,
            MAX_OFFSET + 1,
            OFFSET_STEP,
        )
    )

    results = []

    for offset in offsets:
        try:
            row = evaluate_at_offset(
                clean,
                streaming,
                sample_rate,
                offset,
            )
        except ValueError:
            continue

        row["offset_ms"] = (
            row["offset"] / sample_rate * 1000.0
        )
        results.append(row)

    if not results:
        raise RuntimeError("No valid offsets produced overlapping audio.")

    offset_zero = next(
        row for row in results if row["offset"] == 0
    )

    best_corr = max(
        results,
        key=lambda row: row["correlation"],
    )
    best_si_sdr = max(
        results,
        key=lambda row: row["si_sdr_db"],
    )
    best_stoi = max(
        results,
        key=lambda row: row["stoi"],
    )

    def neighbors(
        center: int,
        span: int = 2,
    ) -> list[dict]:
        wanted = {
            center + i * OFFSET_STEP
            for i in range(-span, span + 1)
        }
        return [
            row
            for row in results
            if row["offset"] in wanted
        ]

    print()
    print("=" * 88)
    print("BASELINE AT OFFSET 0 (evaluate.py assumption)")
    print("=" * 88)
    print(
        f"Overlap:   {offset_zero['overlap_samples']} samples "
        f"({offset_zero['overlap_seconds']:.3f} s)"
    )
    print(f"Correlation: {offset_zero['correlation']:.6f}")
    print(f"SNR:         {offset_zero['snr_db']:.3f} dB")
    print(f"SI-SDR:      {offset_zero['si_sdr_db']:.3f} dB")
    print(f"STOI:        {offset_zero['stoi']:.4f}")

    print_table(
        title=(
            "TABLE AROUND BEST CORRELATION "
            f"(offset={best_corr['offset']} samples, "
            f"{best_corr['offset_ms']:.1f} ms)"
        ),
        rows=neighbors(best_corr["offset"]),
        center_offset=best_corr["offset"],
    )

    print_table(
        title=(
            "TABLE AROUND BEST SI-SDR "
            f"(offset={best_si_sdr['offset']} samples, "
            f"{best_si_sdr['offset_ms']:.1f} ms)"
        ),
        rows=neighbors(best_si_sdr["offset"]),
        center_offset=best_si_sdr["offset"],
    )

    print()
    print("=" * 88)
    print("BEST-OFFSET SUMMARY")
    print("=" * 88)
    print(
        f"Best correlation: offset={best_corr['offset']:>5d} "
        f"({best_corr['offset_ms']:>6.1f} ms), "
        f"corr={best_corr['correlation']:.6f}, "
        f"SI-SDR={best_corr['si_sdr_db']:.3f} dB, "
        f"STOI={best_corr['stoi']:.4f}"
    )
    print(
        f"Best SI-SDR:      offset={best_si_sdr['offset']:>5d} "
        f"({best_si_sdr['offset_ms']:>6.1f} ms), "
        f"corr={best_si_sdr['correlation']:.6f}, "
        f"SI-SDR={best_si_sdr['si_sdr_db']:.3f} dB, "
        f"STOI={best_si_sdr['stoi']:.4f}"
    )
    print(
        f"Best STOI:        offset={best_stoi['offset']:>5d} "
        f"({best_stoi['offset_ms']:>6.1f} ms), "
        f"corr={best_stoi['correlation']:.6f}, "
        f"SI-SDR={best_stoi['si_sdr_db']:.3f} dB, "
        f"STOI={best_stoi['stoi']:.4f}"
    )

    si_sdr_gain = (
        best_si_sdr["si_sdr_db"]
        - offset_zero["si_sdr_db"]
    )
    stoi_gain = (
        best_stoi["stoi"]
        - offset_zero["stoi"]
    )
    snr_gain = (
        best_si_sdr["snr_db"]
        - offset_zero["snr_db"]
    )

    print()
    print("=" * 88)
    print("CONCLUSION")
    print("=" * 88)

    if abs(si_sdr_gain) >= 10.0 or abs(stoi_gain) >= 0.15:
        print(
            "Alignment appears to be a MAJOR factor in the poor metrics."
        )
        print(
            f"  SI-SDR improves by {si_sdr_gain:+.3f} dB when shifted "
            f"from offset 0 to {best_si_sdr['offset']}."
        )
        print(
            f"  STOI improves by {stoi_gain:+.4f} when shifted "
            f"from offset 0 to {best_si_sdr['offset']}."
        )
        print(
            f"  SNR improves by {snr_gain:+.3f} dB over the same shift."
        )
    elif abs(si_sdr_gain) >= 3.0 or abs(stoi_gain) >= 0.05:
        print(
            "Alignment is a MODERATE factor; shifting helps, but does "
            "not fully explain the gap."
        )
        print(
            f"  SI-SDR gain from alignment: {si_sdr_gain:+.3f} dB"
        )
        print(
            f"  STOI gain from alignment: {stoi_gain:+.4f}"
        )
    else:
        print(
            "Alignment is NOT the primary cause of catastrophic SI-SDR/STOI."
        )
        print(
            f"  SI-SDR change from best offset: {si_sdr_gain:+.3f} dB"
        )
        print(
            f"  STOI change from best offset: {stoi_gain:+.4f}"
        )

    if best_si_sdr["si_sdr_db"] > 0:
        print(
            "After best alignment, SI-SDR is positive, suggesting the "
            "streaming output is broadly usable once shifted."
        )
    else:
        print(
            "Even after best alignment, SI-SDR remains poor, suggesting "
            "additional enhancement-quality issues beyond sample shift."
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
