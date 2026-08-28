from pathlib import Path
import ctypes


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
    print("DRDO-ANC | Native DeepFilterNet Streaming Smoke Test")
    print("=" * 60)

    print(f"DLL:   {DLL_PATH}")
    print(f"Model: {MODEL_PATH}")

    if not DLL_PATH.exists():
        raise FileNotFoundError(f"DLL not found: {DLL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # ---------------------------------------------------------
    # Load native library
    # ---------------------------------------------------------

    lib = ctypes.CDLL(str(DLL_PATH))

    # ---------------------------------------------------------
    # Declare C API signatures
    # ---------------------------------------------------------

    lib.df_create.argtypes = [
        ctypes.c_char_p,                   # path
        ctypes.POINTER(ctypes.c_float),    # attenuation limit
        ctypes.c_char_p,                   # log level
    ]
    lib.df_create.restype = ctypes.c_void_p

    lib.df_get_frame_length.argtypes = [
        ctypes.c_void_p,
    ]
    lib.df_get_frame_length.restype = ctypes.c_size_t

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
        raise RuntimeError("df_create() returned a null state pointer.")

    print("\nNative DF3 state created successfully.")

    # ---------------------------------------------------------
    # Query frame size
    # ---------------------------------------------------------

    frame_length = lib.df_get_frame_length(state)

    print(f"Frame length: {frame_length} samples")
    print(f"Frame duration: {frame_length / 48000 * 1000:.2f} ms")

    # ---------------------------------------------------------
    # Destroy state
    # ---------------------------------------------------------

    lib.df_free(state)

    print("Native DF3 state destroyed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()