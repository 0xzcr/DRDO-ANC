from pathlib import Path

import torch

from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "clean_freesound_33711_noise_573577_snr0.wav"
)


def main():
    print("=" * 70)
    print("DRDO-ANC | Enhancer Streaming Integration Test")
    print("=" * 70)

    enhancer = DeepFilterNetEnhancer()

    print("\nLoading enhancer...")
    enhancer.load()

    print(f"\nName:        {enhancer.name()}")
    print(f"Sample rate: {enhancer.sample_rate()} Hz")

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    import soundfile as sf

    audio_np, sample_rate = sf.read(
        INPUT_PATH,
        dtype="float32",
    )

    if sample_rate != enhancer.sample_rate():
        raise ValueError(
            f"Sample-rate mismatch: "
            f"audio={sample_rate}, "
            f"enhancer={enhancer.sample_rate()}"
        )

    audio = torch.from_numpy(audio_np)

    # ---------------------------------------------------------
    # Send deliberately awkward chunk sizes
    # ---------------------------------------------------------

    chunk_sizes = [
        300,
        700,
        250,
        1000,
        137,
        911,
        2048,
    ]

    position = 0

    total_input = 0
    total_output = 0

    print("\nProcessing arbitrary chunks...")

    for index, requested_size in enumerate(
        chunk_sizes,
        start=1,
    ):
        if position >= len(audio):
            break

        end = min(
            position + requested_size,
            len(audio),
        )

        chunk = audio[position:end]

        output = enhancer.process_stream(chunk)

        total_input += len(chunk)
        total_output += len(output)

        print(
            f"Chunk {index}: "
            f"input={len(chunk):4d}, "
            f"output={len(output):4d}"
        )

        position = end

    print("\nStreaming test complete.")

    print(f"Input consumed:  {total_input}")
    print(f"Output produced: {total_output}")

    # ---------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------

    print("\nTesting reset...")

    enhancer.reset()

    print("Reset successful.")

    print("\nAll Enhancer streaming tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()