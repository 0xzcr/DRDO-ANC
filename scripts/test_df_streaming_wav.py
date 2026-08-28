from pathlib import Path
import ctypes

import numpy as np
import soundfile as sf


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
    / "native_streaming_test.wav"
)


def main():
    print("=" * 70)
    print("DRDO-ANC | Native DF3 Streaming WAV Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    audio, sample_rate = sf.read(
        INPUT_PATH,
        dtype="float32",
    )

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio, got shape {audio.shape}"
        )

    print(f"Input:       {INPUT_PATH}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples:     {len(audio)}")

    # We deliberately test exactly 1 second.
    test_samples = sample_rate

    if len(audio) < test_samples:
        raise ValueError("Input audio is shorter than 1 second.")

    audio = np.ascontiguousarray(
        audio[:test_samples],
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Load native library
    # ---------------------------------------------------------

    lib = ctypes.CDLL(str(DLL_PATH))

    # ---------------------------------------------------------
    # Define C API
    # ---------------------------------------------------------

    lib.df_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
    ]
    lib.df_create.restype = ctypes.c_void_p

    lib.df_get_frame_length.argtypes = [
        ctypes.c_void_p,
    ]
    lib.df_get_frame_length.restype = ctypes.c_size_t

    lib.df_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.df_process_frame.restype = ctypes.c_float

    lib.df_free.argtypes = [
        ctypes.c_void_p,
    ]
    lib.df_free.restype = None

    # ---------------------------------------------------------
    # Create streaming state
    # ---------------------------------------------------------

    attenuation_limit = ctypes.c_float(0.0)

    state = lib.df_create(
        str(MODEL_PATH).encode("utf-8"),
        ctypes.byref(attenuation_limit),
        None,
    )

    if not state:
        raise RuntimeError(
            "df_create() returned a null pointer."
        )

    try:
        # -----------------------------------------------------
        # Determine frame size
        # -----------------------------------------------------

        frame_length = lib.df_get_frame_length(state)

        print(f"Frame length: {frame_length} samples")
        print(
            f"Frame duration: "
            f"{frame_length / sample_rate * 1000:.2f} ms"
        )

        if test_samples % frame_length != 0:
            raise ValueError(
                f"Test audio length ({test_samples}) is not "
                f"divisible by frame length ({frame_length})."
            )

        num_frames = test_samples // frame_length

        print(f"Frames:       {num_frames}")
        print(f"Duration:     {test_samples / sample_rate:.3f} s")

        # -----------------------------------------------------
        # Streaming processing
        # -----------------------------------------------------

        enhanced = np.zeros_like(audio)

        local_snrs = []

        print("\nProcessing stream...")

        for frame_index in range(num_frames):
            start = frame_index * frame_length
            end = start + frame_length

            input_frame = np.ascontiguousarray(
                audio[start:end],
                dtype=np.float32,
            )

            output_frame = np.zeros(
                frame_length,
                dtype=np.float32,
            )

            local_snr = lib.df_process_frame(
                state,
                input_frame.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
                output_frame.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
            )

            enhanced[start:end] = output_frame
            local_snrs.append(local_snr)

            if (
                frame_index == 0
                or frame_index == num_frames - 1
                or (frame_index + 1) % 25 == 0
            ):
                print(
                    f"Frame {frame_index + 1:3d}/{num_frames} | "
                    f"Local SNR: {local_snr:7.3f} dB"
                )

        # -----------------------------------------------------
        # Save output
        # -----------------------------------------------------

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            OUTPUT_PATH,
            enhanced,
            sample_rate,
        )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        input_rms = np.sqrt(
            np.mean(audio ** 2)
        )

        output_rms = np.sqrt(
            np.mean(enhanced ** 2)
        )

        print("\nStreaming processing complete.")
        print(f"Output:       {OUTPUT_PATH}")
        print(f"Output shape: {enhanced.shape}")
        print(f"Input RMS:    {input_rms:.8f}")
        print(f"Output RMS:   {output_rms:.8f}")
        print(
            f"Local SNR:    "
            f"{min(local_snrs):.3f} to "
            f"{max(local_snrs):.3f} dB"
        )

    finally:
        lib.df_free(state)

    print("\nNative DF3 state destroyed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()