import torch

from df import enhance, init_df

from .base import Enhancer


class DeepFilterNetEnhancer(Enhancer):
    """DeepFilterNet3 implementation of the Enhancer interface."""

    def __init__(self):
        self.model = None
        self.df_state = None
        self._sample_rate = None
        self._name = "DeepFilterNet3"

    def load(self) -> None:
        """Load DeepFilterNet3 and its processing state."""

        self.model, self.df_state, suffix, epoch = init_df()

        self._name = suffix
        self._sample_rate = self.df_state.sr()

        self.device = next(self.model.parameters()).device

        print(f"Model:       {self._name}")
        print(f"Checkpoint:  epoch {epoch}")
        print(f"Device:      {self.device}")
        print(f"DF rate:     {self._sample_rate} Hz")

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance audio using DeepFilterNet."""

        if self.model is None or self.df_state is None:
            raise RuntimeError(
                "Enhancer is not loaded. Call load() before process()."
            )

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        return enhance(
            self.model,
            self.df_state,
            audio,
        )

    def reset(self) -> None:
        """Reset DeepFilterNet processing state."""

        # DeepFilterNet's processing state is currently tied to df_state.
        # We reload it to obtain a fresh state.
        if self.model is not None:
            self.load()

    def sample_rate(self) -> int:
        """Return the DeepFilterNet sample rate."""

        if self._sample_rate is None:
            raise RuntimeError(
                "Enhancer is not loaded. Call load() first."
            )

        return self._sample_rate

    def name(self) -> str:
        """Return the model name."""

        return self._name