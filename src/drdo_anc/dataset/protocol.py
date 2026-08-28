from typing import Iterator, Protocol

from .sample import AudioSample


class Dataset(Protocol):
    """Minimal dataset interface over audio sample metadata."""

    def __iter__(self) -> Iterator[AudioSample]: ...

    def __len__(self) -> int: ...
