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

AUDIO_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "clean_freesound_33711_noise_573577_snr0.wav"
)


def main():
    print("=" * 60)
    print("DRDO-ANC | Native DeepFilterNet Frame Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
    )

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    print(f"Audio:       {AUDIO_PATH.name}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples:     {len(audio)}")

    # ---------------------------------------------------------
    # Load DLL
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
    # Create native DF3 state
    # ---------------------------------------------------------

    attenuation_limit = ctypes.c_float(0.0)

    state = lib.df_create(
        str(MODEL_PATH).encode("utf-8"),
        ctypes.byref(attenuation_limit),
        None,
    )

    if not state:
        raise RuntimeError("df_create() returned a null pointer.")

    try:
        # -----------------------------------------------------
        # Get required frame size
        # -----------------------------------------------------

        frame_length = lib.df_get_frame_length(state)

        print(f"Frame length: {frame_length} samples")

        if len(audio) < frame_length:
            raise ValueError("Audio is shorter than one DF frame.")
        # ---------------------------------------------------------
        # Process consecutive frames
        # ---------------------------------------------------------

        num_frames = 100
        total_samples = num_frames * frame_length

        if len(audio) < total_samples:
            raise ValueError(
                f"Audio is too short for {num_frames} frames."
            )

        input_audio = np.ascontiguousarray(
            audio[:total_samples],
            dtype=np.float32,
        )

        output_audio = np.zeros(
            total_samples,
            dtype=np.float32,
        )

        print(f"\nProcessing {num_frames} consecutive frames...")
        print(f"Total samples: {total_samples}")
        print(f"Total duration: {total_samples / sample_rate:.3f} s")

        local_snrs = []

        for i in range(num_frames):
            start = i * frame_length
            end = start + frame_length

            input_frame = input_audio[start:end]
            output_frame = output_audio[start:end]

            local_snr = lib.df_process_frame(
                state,
                input_frame.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
                output_frame.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
            )

            local_snrs.append(local_snr)

        # ---------------------------------------------------------
        # Inspect streaming output
        # ---------------------------------------------------------

        print("\nStreaming processing complete.")

        print(f"Output shape: {output_audio.shape}")
        print(f"Output min:   {output_audio.min():.8f}")
        print(f"Output max:   {output_audio.max():.8f}")
        print(
            f"Output RMS:   "
            f"{np.sqrt(np.mean(output_audio ** 2)):.8f}"
        )

        print(
            f"First frame RMS: "
            f"{np.sqrt(np.mean(output_audio[:frame_length] ** 2)):.8f}"
        )

        print(
            f"Last frame RMS:  "
            f"{np.sqrt(np.mean(output_audio[-frame_length:] ** 2)):.8f}"
        )

        print(
            f"Local SNR range: "
            f"{min(local_snrs):.3f} to {max(local_snrs):.3f} dB"
        )
        
        # -----------------------------------------------------
        # Inspect result
        # -----------------------------------------------------

        print("Frame processed successfully.")
        print(f"Input shape:  {input_frame.shape}")
        print(f"Output shape: {output_frame.shape}")
        print(f"Local SNR:    {local_snr:.3f} dB")
        print(f"Input RMS:    {np.sqrt(np.mean(input_frame ** 2)):.6f}")
        print(f"Output RMS:   {np.sqrt(np.mean(output_frame ** 2)):.6f}")

    finally:
        lib.df_free(state)

    print("\nNative DF3 state destroyed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()