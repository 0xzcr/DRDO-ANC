from pathlib import Path

import numpy as np

from drdo_anc.enhancement.native import NativeDF3Backend
from drdo_anc.enhancement.streaming import StreamingBuffer


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
    print("=" * 60)
    print("DRDO-ANC | Streaming Backend Test")
    print("=" * 60)

    backend = NativeDF3Backend(
        dll_path=DLL_PATH,
        model_path=MODEL_PATH,
    )

    backend.load()

    try:
        frame_length = backend.frame_length()

        print(f"Frame length: {frame_length}")

        buffer = StreamingBuffer(
            frame_length
        )

        # -----------------------------------------------------
        # Test arbitrary chunks
        # -----------------------------------------------------

        chunks = [
            np.zeros(300, dtype=np.float32),
            np.zeros(700, dtype=np.float32),
            np.zeros(250, dtype=np.float32),
        ]

        total_input = 0
        total_output = 0

        for i, chunk in enumerate(chunks, start=1):
            frames = buffer.append(chunk)

            total_input += len(chunk)

            output_samples = 0

            for frame in frames:
                enhanced = backend.process_frame(frame)

                assert enhanced.shape == (
                    frame_length,
                )

                output_samples += len(enhanced)

            total_output += output_samples

            print(
                f"Chunk {i}: "
                f"input={len(chunk)}, "
                f"frames={len(frames)}, "
                f"output={output_samples}, "
                f"pending={buffer.pending_samples()}"
            )

        print()
        print(f"Total input:   {total_input}")
        print(f"Total output:  {total_output}")
        print(
            f"Pending:       {buffer.pending_samples()}"
        )

        # 300 + 700 + 250 = 1250
        # 2 × 480 = 960 processed
        # 290 remain buffered.

        assert total_input == 1250
        assert total_output == 960
        assert buffer.pending_samples() == 290

        # -----------------------------------------------------
        # Test reset
        # -----------------------------------------------------

        print("\nTesting reset...")

        backend.reset()

        assert backend.frame_length() == frame_length

        print("Reset successful.")

    finally:
        backend.close()

    print("\nAll streaming backend tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()