from pathlib import Path
import time

import soundfile as sf
import torch

from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "noisy_snr0.wav"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "enhanced" / "noisy_snr0_abstraction.wav"
)


def main():
    print("=" * 60)
    print("DRDO-ANC | Model-Agnostic Enhancement")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    audio_np, sample_rate = sf.read(
        INPUT_PATH,
        dtype="float32",
    )

    print(f"Input:       {INPUT_PATH}")
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

    sf.write(
        OUTPUT_PATH,
        enhanced_np,
        sample_rate,
    )

    duration = len(audio_np) / sample_rate
    realtime_factor = duration / elapsed

    print("\nEnhancement complete.")
    print(f"Model:       {enhancer.name()}")
    print(f"Output:      {OUTPUT_PATH}")
    print(f"Duration:    {duration:.3f} s")
    print(f"Inference:   {elapsed:.3f} s")
    print(f"RTF:         {realtime_factor:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()