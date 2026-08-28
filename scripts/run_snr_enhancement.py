from pathlib import Path
import time

import soundfile as sf
import torch

from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enhanced"

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


def main():
    enhancer = DeepFilterNetEnhancer()
    enhancer.load()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("DRDO-ANC | DF3 SNR Enhancement Sweep")
    print("=" * 70)

    results = []

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

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
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
                f"got {sample_rate} Hz: {input_path}"
            )

        audio = torch.from_numpy(
            audio_np
        ).float().unsqueeze(0)

        print()
        print(f"Processing SNR {snr_db:+d} dB...")
        print(f"Input:  {input_path.name}")

        if enhancer.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        enhanced = enhancer.process(audio)

        if enhancer.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        enhanced_np = (
            enhanced
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        sf.write(
            output_path,
            enhanced_np,
            sample_rate,
        )

        duration = len(audio_np) / sample_rate
        rtf = duration / elapsed

        results.append(
            (snr_db, elapsed, rtf)
        )

        print(f"Output: {output_path.name}")
        print(f"Inference: {elapsed:.3f} s")
        print(f"RTF:       {rtf:.2f}x")

    print()
    print("=" * 70)
    print("SNR Enhancement Sweep Complete")
    print("=" * 70)

    print()
    print(
        f"{'SNR':>8} "
        f"{'Inference':>15} "
        f"{'RTF':>10}"
    )
    print("-" * 40)

    for snr_db, elapsed, rtf in results:
        print(
            f"{snr_db:>+7.0f} dB "
            f"{elapsed:>14.3f} s "
            f"{rtf:>9.2f}x"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()