from pathlib import Path
import time

import numpy as np

import soundfile as sf
import torch

from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enhanced"

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


def synchronize_cuda(device):
    """Wait for queued CUDA operations to finish."""

    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    print("=" * 70)
    print("DRDO-ANC | DF3 SNR Enhancement Benchmark")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load model once
    # ---------------------------------------------------------

    enhancer = DeepFilterNetEnhancer()
    enhancer.load()

    print()
    print(f"Model:       {enhancer.name()}")
    print(f"Device:      {enhancer.device}")
    print(f"Sample rate: {enhancer.sample_rate()} Hz")

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    warmup_snr = SNR_LEVELS[0]

    warmup_path = (
        INPUT_DIR
        / (
            "clean_freesound_33711_"
            "noise_573577_"
            f"snr{warmup_snr}.wav"
        )
    )

    warmup_np, warmup_sr = sf.read(
        warmup_path,
        dtype="float32",
    )

    warmup_audio = (
        torch.from_numpy(warmup_np)
        .float()
        .unsqueeze(0)
    )

    print()
    print("Running GPU/model warm-up...")

    synchronize_cuda(enhancer.device)

    warmup_start = time.perf_counter()

    enhancer.process(warmup_audio)

    synchronize_cuda(enhancer.device)

    warmup_elapsed = (
        time.perf_counter() - warmup_start
    )

    print(
        f"Warm-up time: {warmup_elapsed:.3f} s"
    )

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    results = []

    print()
    print("Running steady-state benchmark...")

    for snr_db in SNR_LEVELS:
        input_path = (
            INPUT_DIR
            / (
                "clean_freesound_33711_"
                "noise_573577_"
                f"snr{snr_db}.wav"
            )
        )

        output_path = (
            OUTPUT_DIR
            / (
                "clean_freesound_33711_"
                "noise_573577_"
                f"snr{snr_db}_df3.wav"
            )
        )

        audio_np, sample_rate = sf.read(
            input_path,
            dtype="float32",
        )

        if audio_np.ndim != 1:
            raise ValueError(
                f"Expected mono audio: {input_path}"
            )

        if sample_rate != enhancer.sample_rate():
            raise ValueError(
                f"Expected {enhancer.sample_rate()} Hz, "
                f"got {sample_rate} Hz"
            )

        audio = (
            torch.from_numpy(audio_np)
            .float()
            .unsqueeze(0)
        )

        synchronize_cuda(enhancer.device)

        start = time.perf_counter()

        enhanced = enhancer.process(audio)

        synchronize_cuda(enhancer.device)

        elapsed = time.perf_counter() - start

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

        duration = (
            len(audio_np) / sample_rate
        )

        rtf = duration / elapsed

        results.append(
            {
                "snr": snr_db,
                "duration": duration,
                "elapsed": elapsed,
                "rtf": rtf,
            }
        )

        print(
            f"SNR {snr_db:+d} dB | "
            f"{elapsed:.3f} s | "
            f"RTF {rtf:.2f}x"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    rtf_values = [
        result["rtf"]
        for result in results
    ]

    elapsed_values = [
        result["elapsed"]
        for result in results
    ]

    print()
    print("=" * 70)
    print("STEADY-STATE TIMING RESULTS")
    print("=" * 70)

    print()
    print(
        f"{'SNR':>8}"
        f"{'Inference':>15}"
        f"{'RTF':>12}"
    )

    print("-" * 40)

    for result in results:
        print(
            f"{result['snr']:>+7.0f} dB"
            f"{result['elapsed']:>14.3f} s"
            f"{result['rtf']:>11.2f}x"
        )

    print("-" * 40)

    print(
        f"{'Median':>8}"
        f"{np.median(elapsed_values):>14.3f} s"
        f"{np.median(rtf_values):>11.2f}x"
    )

    print()
    print(f"Warm-up: {warmup_elapsed:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()