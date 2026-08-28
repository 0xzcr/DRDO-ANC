import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

from drdo_anc.enhancement import DeepFilterNetEnhancer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a speech-enhancement model."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input noisy WAV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output enhanced WAV file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="DeepFilterNet3",
        help="Model name. Currently only DeepFilterNet3 is supported.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if args.model != "DeepFilterNet3":
        raise ValueError(
            f"Unsupported model: {args.model}. "
            "Currently only DeepFilterNet3 is available."
        )

    print("=" * 60)
    print("DRDO-ANC | Model-Agnostic Enhancement")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    audio_np, sample_rate = sf.read(
        input_path,
        dtype="float32",
    )

    if audio_np.ndim != 1:
        raise ValueError(
            f"Expected mono input, got shape {audio_np.shape}"
        )

    print(f"Input:       {input_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples:     {len(audio_np)}")

    # ---------------------------------------------------------
    # Initialize enhancer
    # ---------------------------------------------------------

    enhancer = DeepFilterNetEnhancer()
    enhancer.load()

    expected_sample_rate = enhancer.sample_rate()

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Expected {expected_sample_rate} Hz input, "
            f"got {sample_rate} Hz"
        )

    # ---------------------------------------------------------
    # Convert NumPy → Tensor
    # ---------------------------------------------------------

    audio = torch.from_numpy(audio_np).float()

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # ---------------------------------------------------------
    # Run enhancement
    # ---------------------------------------------------------

    print("\nRunning enhancement...")

    if enhancer.device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    enhanced = enhancer.process(audio)

    if enhancer.device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------

    enhanced_np = (
        enhanced
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        output_path,
        enhanced_np,
        sample_rate,
    )

    duration = len(audio_np) / sample_rate
    realtime_factor = duration / elapsed

    print("\nEnhancement complete.")
    print(f"Model:       {enhancer.name()}")
    print(f"Output:      {output_path}")
    print(f"Duration:    {duration:.3f} s")
    print(f"Inference:  {elapsed:.3f} s")
    print(f"RTF:         {realtime_factor:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()