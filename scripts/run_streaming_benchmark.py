from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "clean_freesound_33711_noise_573577_snr0.wav"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "enhanced"
    / "clean_freesound_33711_noise_573577_snr0_streaming_df3.wav"
)


def main():
    print("=" * 70)
    print("DRDO-ANC | Full Streaming DF3 Benchmark")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load input
    # ---------------------------------------------------------

    audio, sample_rate = sf.read(
        INPUT_PATH,
        dtype="float32",
    )

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio, got shape {audio.shape}"
        )

    # ---------------------------------------------------------
    # Load enhancer
    # ---------------------------------------------------------

    enhancer = DeepFilterNetEnhancer()
    enhancer.load()

    if sample_rate != enhancer.sample_rate():
        raise ValueError(
            f"Sample-rate mismatch: "
            f"audio={sample_rate}, "
            f"enhancer={enhancer.sample_rate()}"
        )

    print(f"\nModel:       {enhancer.name()}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Input:       {INPUT_PATH.name}")
    print(f"Samples:     {len(audio)}")
    print(f"Duration:    {len(audio) / sample_rate:.3f} s")

    # ---------------------------------------------------------
    # Simulate arbitrary incoming chunks
    # ---------------------------------------------------------

    chunk_sizes = [
        300,
        700,
        250,
        1000,
        137,
        911,
        2048,
        512,
        1536,
        800,
        1200,
    ]

    outputs = []

    position = 0
    chunk_index = 0

    total_input = 0
    total_output = 0

    print("\nRunning streaming enhancement...")

    start_time = time.perf_counter()

    while position < len(audio):
        requested_size = chunk_sizes[
            chunk_index % len(chunk_sizes)
        ]

        end = min(
            position + requested_size,
            len(audio),
        )

        chunk = audio[position:end]

        output = enhancer.process_stream(
            torch.from_numpy(chunk)
        )

        output_np = (
            output.detach()
            .cpu()
            .numpy()
        )

        if len(output_np) > 0:
            outputs.append(output_np)

        total_input += len(chunk)
        total_output += len(output_np)

        position = end
        chunk_index += 1

    # ---------------------------------------------------------
    # Flush final partial frame
    # ---------------------------------------------------------

    pending_before_flush = (
        total_input - total_output
    )

    print(
        f"\nPending before flush: "
        f"{pending_before_flush} samples"
    )

    flush_output = enhancer.flush()

    flush_output_np = (
        flush_output.detach()
        .cpu()
        .numpy()
    )

    if len(flush_output_np) > 0:
        outputs.append(flush_output_np)

    total_output += len(flush_output_np)

    print(
        f"Final flush output:   "
        f"{len(flush_output_np)} samples"
    )

    elapsed = time.perf_counter() - start_time

    # ---------------------------------------------------------
    # Concatenate complete output
    # ---------------------------------------------------------

    if outputs:
        enhanced = np.concatenate(outputs)
    else:
        enhanced = np.empty(
            0,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Final accounting
    # ---------------------------------------------------------

    pending_after_flush = (
        total_input - total_output
    )

    duration = len(audio) / sample_rate
    output_duration = len(enhanced) / sample_rate
    rtf = duration / elapsed

    print("\n" + "=" * 70)
    print("STREAMING ENHANCEMENT COMPLETE")
    print("=" * 70)

    print(f"Input samples:       {total_input}")
    print(f"Output samples:      {total_output}")
    print(f"Pending samples:     {pending_after_flush}")

    print(f"Input duration:      {duration:.3f} s")
    print(f"Output duration:     {output_duration:.3f} s")

    print(f"Inference time:      {elapsed:.3f} s")
    print(f"RTF:                 {rtf:.2f}x")

    # ---------------------------------------------------------
    # Verify sample count
    # ---------------------------------------------------------

    if len(enhanced) != len(audio):
        raise RuntimeError(
            f"Streaming output length mismatch: "
            f"input={len(audio)}, "
            f"output={len(enhanced)}"
        )

    print("\nLength verification: PASSED")

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        OUTPUT_PATH,
        enhanced,
        sample_rate,
    )

    print(f"\nOutput: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()