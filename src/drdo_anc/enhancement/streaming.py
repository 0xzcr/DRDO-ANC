import numpy as np


class StreamingBuffer:
    """Convert arbitrary audio chunks into fixed-size frames."""

    def __init__(self, frame_length: int) -> None:
        if frame_length <= 0:
            raise ValueError(
                "frame_length must be positive."
            )

        self.frame_length = frame_length
        self._buffer = np.empty(
            0,
            dtype=np.float32,
        )

    def append(self, audio: np.ndarray) -> list[np.ndarray]:
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim != 1:
            raise ValueError(
                f"Expected mono audio [T], got shape {audio.shape}"
            )

        if audio.size == 0:
            return []

        self._buffer = np.concatenate(
            (self._buffer, audio)
        )

        num_frames = (
            len(self._buffer)
            // self.frame_length
        )

        if num_frames == 0:
            return []

        split_point = (
            num_frames * self.frame_length
        )

        complete = self._buffer[:split_point]

        self._buffer = self._buffer[split_point:]

        return [
            complete[
                i : i + self.frame_length
            ]
            for i in range(
                0,
                len(complete),
                self.frame_length,
            )
        ]

    def clear(self) -> None:
        self._buffer = np.empty(
            0,
            dtype=np.float32,
        )

    def pending_samples(self) -> int:
        return len(self._buffer)

    def flush(self) -> np.ndarray:
        """
        Return all currently buffered samples and clear the buffer.

        This is used when the input stream has ended and the remaining
        samples do not form a complete frame.
        """
        buffered = self._buffer.copy()

        self.clear()

        return buffered