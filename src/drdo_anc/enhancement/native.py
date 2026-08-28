from pathlib import Path
import ctypes
from typing import Optional

import numpy as np


class NativeDF3Backend:
    """Thin ctypes wrapper around DeepFilterNet's native C API."""

    def __init__(
        self,
        dll_path: Path,
        model_path: Path,
        attenuation_limit_db: float = 100.0,
    ) -> None:
        self.dll_path = Path(dll_path)
        self.model_path = Path(model_path)
        self.attenuation_limit_db = attenuation_limit_db

        self._lib = None
        self._state = None
        self._frame_length: Optional[int] = None

    def load(self) -> None:
        if not self.dll_path.exists():
            raise FileNotFoundError(
                f"DeepFilterNet DLL not found: {self.dll_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DeepFilterNet model not found: {self.model_path}"
            )

        self._lib = ctypes.CDLL(str(self.dll_path))

        self._configure_api()
        self._create_state()

    def _configure_api(self) -> None:
        assert self._lib is not None

        self._lib.df_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.c_char_p,
        ]
        self._lib.df_create.restype = ctypes.c_void_p

        self._lib.df_get_frame_length.argtypes = [
            ctypes.c_void_p,
        ]
        self._lib.df_get_frame_length.restype = ctypes.c_size_t

        self._lib.df_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.df_process_frame.restype = ctypes.c_float

        self._lib.df_free.argtypes = [
            ctypes.c_void_p,
        ]
        self._lib.df_free.restype = None

    def _create_state(self) -> None:
        assert self._lib is not None

        state = self._lib.df_create(
            str(self.model_path).encode("utf-8"),
            ctypes.c_float(self.attenuation_limit_db),
            None,
        )

        if not state:
            raise RuntimeError(
                "df_create() returned a null state pointer."
            )

        self._state = state

        self._frame_length = int(
            self._lib.df_get_frame_length(self._state)
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._lib is None or self._state is None:
            raise RuntimeError(
                "Native DF3 backend is not loaded."
            )

        if self._frame_length is None:
            raise RuntimeError(
                "Native DF3 frame length is unknown."
            )

        frame = np.asarray(
            frame,
            dtype=np.float32,
        )

        if frame.ndim != 1:
            raise ValueError(
                f"Expected mono frame [T], got shape {frame.shape}"
            )

        if len(frame) != self._frame_length:
            raise ValueError(
                f"Expected {self._frame_length} samples, "
                f"got {len(frame)}"
            )

        frame = np.ascontiguousarray(
            frame,
            dtype=np.float32,
        )

        output = np.zeros(
            self._frame_length,
            dtype=np.float32,
        )

        self._lib.df_process_frame(
            self._state,
            frame.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            ),
            output.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            ),
        )

        return output

    def frame_length(self) -> int:
        if self._frame_length is None:
            raise RuntimeError(
                "Native DF3 backend is not loaded."
            )

        return self._frame_length

    def reset(self) -> None:
        if self._lib is None:
            raise RuntimeError(
                "Native DF3 backend is not loaded."
            )

        self.close()
        self._create_state()

    def close(self) -> None:
        if self._lib is not None and self._state is not None:
            self._lib.df_free(self._state)

        self._state = None
        self._frame_length = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass