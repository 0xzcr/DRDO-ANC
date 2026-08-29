from abc import ABC, abstractmethod

import numpy as np


class AudioInput(ABC):
    """Hardware-independent mono audio capture interface."""

    @abstractmethod
    def sample_rate(self) -> int:
        """Return the capture sample rate in Hz."""

    @abstractmethod
    def read(self, max_samples: int) -> np.ndarray:
        """
        Read up to ``max_samples`` of mono float32 audio.

        Implementations may return fewer samples than requested.
        An empty array signals that no more input is available.
        """

    @abstractmethod
    def close(self) -> None:
        """Release capture resources."""


class AudioOutput(ABC):
    """Hardware-independent mono audio playback interface."""

    @abstractmethod
    def sample_rate(self) -> int:
        """Return the playback sample rate in Hz."""

    @abstractmethod
    def write(self, audio: np.ndarray) -> None:
        """Write mono float32 audio to the output device."""

    @abstractmethod
    def close(self) -> None:
        """Release playback resources."""
