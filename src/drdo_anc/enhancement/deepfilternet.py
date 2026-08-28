import torch

from df import enhance, init_df

from .base import Enhancer


class DeepFilterNetEnhancer(Enhancer):
    """DeepFilterNet3 implementation of the Enhancer interface."""

    def __init__(self):
        self.model = None
        self.df_state = None
        self.device = None
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
        """Enhance an audio tensor using DeepFilterNet."""

        if self.model is None or self.df_state is None:
            raise RuntimeError(
                "Enhancer is not loaded. Call load() before process()."
            )

        if not isinstance(audio, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(audio).__name__}"
            )

        if audio.ndim not in (1, 2):
            raise ValueError(
                f"Expected audio with 1 or 2 dimensions, got {audio.ndim}"
            )

        if not audio.is_floating_point():
            audio = audio.float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        return enhance(
            self.model,
            self.df_state,
            audio,
        )

    def reset(self) -> None:
        """
        Reset processing state.

        DeepFilterNet maintains internal state required for continuous
        processing. The exact state-reset mechanism will be handled when
        the streaming API is implemented.

        For now, reset is intentionally not implemented rather than
        incorrectly reloading the entire model.
        """

        raise NotImplementedError(
            "DeepFilterNet state reset will be implemented with the "
            "streaming processing API."
        )

    def sample_rate(self) -> int:
        """Return the sample rate expected by DeepFilterNet."""

        if self._sample_rate is None:
            raise RuntimeError(
                "Enhancer is not loaded. Call load() first."
            )

        return self._sample_rate

    def name(self) -> str:
        """Return the model name."""

        return self._name