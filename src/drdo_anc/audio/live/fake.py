import numpy as np

from .interfaces import AudioInput, AudioOutput


class FakeAudioInput(AudioInput):
    """
    In-memory audio input for tests.

    Each ``read()`` call returns the next scripted chunk. Chunk sizes are
    arbitrary and independent of ``max_samples`` (like many host APIs).
    After the scripted chunks are exhausted, ``read()`` returns an empty
    array to signal end-of-stream.
    """

    def __init__(
        self,
        chunks: list[np.ndarray],
        sample_rate: int,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        self._chunks = [
            np.asarray(chunk, dtype=np.float32).reshape(-1)
            for chunk in chunks
        ]
        self._sample_rate = sample_rate
        self._index = 0
        self._closed = False

    def sample_rate(self) -> int:
        return self._sample_rate

    def read(self, max_samples: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AudioInput is closed.")

        if max_samples <= 0:
            return np.empty(0, dtype=np.float32)

        if self._index >= len(self._chunks):
            return np.empty(0, dtype=np.float32)

        chunk = self._chunks[self._index]
        self._index += 1

        return chunk.astype(np.float32, copy=False)

    def close(self) -> None:
        self._closed = True


class FakeAudioOutput(AudioOutput):
    """In-memory audio output that records every ``write()`` call."""

    def __init__(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        self._sample_rate = sample_rate
        self._written: list[np.ndarray] = []
        self._closed = False

    def sample_rate(self) -> int:
        return self._sample_rate

    def write(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("AudioOutput is closed.")

        array = np.asarray(audio, dtype=np.float32).reshape(-1)

        if array.size == 0:
            return

        self._written.append(array.copy())

    def close(self) -> None:
        self._closed = True

    @property
    def written_chunks(self) -> tuple[np.ndarray, ...]:
        return tuple(self._written)

    def all_written(self) -> np.ndarray:
        if not self._written:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(self._written)
