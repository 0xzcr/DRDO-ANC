from pathlib import Path

import numpy as np

from drdo_anc.enhancement.native import NativeDF3Backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DLL_PATH = (
    PROJECT_ROOT
    / "external"
    / "DeepFilterNet"
    / "target"
    / "release"
    / "df.dll"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "external"
    / "DeepFilterNet"
    / "models"
    / "DeepFilterNet3_onnx.tar.gz"
)


def main():
    print("=" * 70)
    print("DRDO-ANC | Native DF3 Latency Test")
    print("=" * 70)

    backend = NativeDF3Backend(
        dll_path=DLL_PATH,
        model_path=MODEL_PATH,
    )

    backend.load()

    frame_length = backend.frame_length()

    print(f"Frame length: {frame_length} samples")
    print(f"Frame duration: {frame_length / 48000 * 1000:.2f} ms")

    # ---------------------------------------------------------
    # Create impulse
    # ---------------------------------------------------------

    num_frames = 100
    total_samples = num_frames * frame_length

    impulse_position = 10 * frame_length

    signal = np.zeros(
        total_samples,
        dtype=np.float32,
    )

    signal[impulse_position] = 1.0

    print(
        f"Impulse position: {impulse_position} samples "
        f"({impulse_position / 48000:.3f} s)"
    )

    # ---------------------------------------------------------
    # Process frame by frame
    # ---------------------------------------------------------

    output = np.zeros_like(signal)

    local_snrs = []

    for frame_index in range(num_frames):
        start = frame_index * frame_length
        end = start + frame_length

        frame_output = backend.process_frame(
            signal[start:end]
        )

        output[start:end] = frame_output

    backend.reset()

    # ---------------------------------------------------------
    # Find strongest output peak
    # ---------------------------------------------------------

    abs_output = np.abs(output)

    peak_index = int(
        np.argmax(abs_output)
    )

    peak_value = float(
        abs_output[peak_index]
    )

    delay_samples = (
        peak_index - impulse_position
    )

    delay_ms = (
        delay_samples
        / 48000
        * 1000
    )

    print("\n" + "=" * 70)
    print("LATENCY RESULT")
    print("=" * 70)

    print(f"Input impulse:    {impulse_position}")
    print(f"Output peak:      {peak_index}")
    print(f"Peak magnitude:   {peak_value:.8f}")
    print(f"Delay:            {delay_samples} samples")
    print(f"Delay:            {delay_ms:.3f} ms")

    print("=" * 70)


if __name__ == "__main__":
    main()