from pathlib import Path
import time

import soundfile as sf
import torch

from df import enhance, init_df


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "noisy_snr0.wav"
OUTPUT_PATH = PROJECT_ROOT / "data" / "enhanced" / "noisy_snr0_deepfilternet3.wav"


def main():
    print("=" * 60)
    print("DRDO-ANC | DeepFilterNet3 Inference")
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

    if sample_rate != 48_000:
        raise ValueError(
            f"Expected 48000 Hz input, got {sample_rate} Hz"
        )

    # DeepFilterNet expects [channels, samples].
    audio = torch.from_numpy(audio_np).float()

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # ---------------------------------------------------------
    # Initialize DeepFilterNet3
    # ---------------------------------------------------------
    print("\nLoading DeepFilterNet3...")

    model, df_state, suffix, epoch = init_df()

    device = next(model.parameters()).device

    print(f"Model:       {suffix}")
    print(f"Checkpoint:  epoch {epoch}")
    print(f"Device:      {device}")
    print(f"DF rate:     {df_state.sr()} Hz")

    # ---------------------------------------------------------
    # Run enhancement
    # ---------------------------------------------------------
    print("\nRunning enhancement...")

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    enhanced = enhance(
        model,
        df_state,
        audio,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    enhanced_np = enhanced.squeeze(0).detach().cpu().numpy()

    sf.write(
        OUTPUT_PATH,
        enhanced_np,
        sample_rate,
    )

    duration = len(audio_np) / sample_rate
    realtime_factor = duration / elapsed

    print("\nEnhancement complete.")
    print(f"Output:      {OUTPUT_PATH}")
    print(f"Duration:    {duration:.3f} s")
    print(f"Inference:   {elapsed:.3f} s")
    print(f"RTF:         {realtime_factor:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()