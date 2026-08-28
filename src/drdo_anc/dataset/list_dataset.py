from .sample import AudioSample


class ListDataset:
    """Dataset backed by an explicit list of samples."""

    def __init__(
        self,
        samples: list[AudioSample],
    ) -> None:
        self._samples = list(samples)

    def __iter__(self):
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
