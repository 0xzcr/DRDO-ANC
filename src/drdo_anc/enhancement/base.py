from abc import ABC, abstractmethod

import torch


class Enhancer(ABC):
    """Abstract interface for speech-enhancement models."""

    @abstractmethod
    def load(self) -> None:
        """Load the model and any required processing state."""
        pass

    @abstractmethod
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance a complete audio tensor."""
        pass

    @abstractmethod
    def process_stream(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """
        Enhance one streaming audio chunk.

        Streaming implementations must preserve the necessary
        model and processing state between calls.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal processing state."""
        pass

    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate expected by the enhancer."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable model name."""
        pass