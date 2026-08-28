from typing import Iterator, Protocol

from .sample import AudioSample
from .source_sample import SourceSample


class Dataset(Protocol):
    """Minimal dataset interface over benchmark audio samples."""

    def __iter__(self) -> Iterator[AudioSample]: ...

    def __len__(self) -> int: ...


class SourceDataset(Protocol):
    """Minimal dataset interface over source-pool clips."""

    def __iter__(self) -> Iterator[SourceSample]: ...

    def __len__(self) -> int: ...