from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono float32 WAV file."""

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
    )

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio in {path.name}, "
            f"got shape {audio.shape}"
        )

    return audio, sample_rate


def save_mono_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """Save a mono float32 WAV file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        path,
        audio.astype(np.float32, copy=False),
        sample_rate,
    )
