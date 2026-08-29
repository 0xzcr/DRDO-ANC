from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from drdo_anc.audio.io import load_mono_wav_bytes

from .manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
    load_metadata_rows,
    row_to_source_sample,
)
from .source_sample import SourceSample
from .zip_access import ZipArchiveCache


class ZipManifestDataset:
    """
    Dataset adapter for ZIP-backed Hugging Face audio manifests.

    Construction loads only ``metadata.csv`` into a small in-memory index.
    ZIP archives and WAV members are accessed lazily when ``load_audio`` is
    called for a specific sample.

  * No full-dataset download is triggered automatically.
  * No whole-archive extraction is performed.
  * Individual WAV files are read from the ZIP in memory and are not written
    to disk unless the caller does so separately.

    When ``archive_dir`` is ``None`` and ``repo_id`` is set, the first audio
    request for a given archive may call ``huggingface_hub.hf_hub_download``
    to fetch **that archive only** into the Hugging Face cache.
    """

    def __init__(
        self,
        metadata_path: Path,
        *,
        repo_id: str | None = SIH26_REPO_ID,
        archive_dir: Path | None = None,
        cache_dir: Path | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        if archive_dir is None and repo_id is None:
            raise ValueError(
                "Provide either archive_dir for local ZIP files "
                "or repo_id for Hugging Face archive resolution."
            )

        self._metadata_path = metadata_path.resolve()
        self._repo_id = repo_id
        self._archive_dir = (
            archive_dir.resolve()
            if archive_dir is not None
            else None
        )
        self._cache_dir = cache_dir
        self._zip_cache = ZipArchiveCache()
        self._resolved_archives: dict[str, Path] = {}

        all_rows = load_metadata_rows(self._metadata_path)

        if indices is None:
            self._rows = all_rows
            self._indices = list(range(len(all_rows)))
        else:
            normalized = list(indices)

            for index in normalized:
                if index < 0 or index >= len(all_rows):
                    raise IndexError(
                        f"Manifest index out of range: {index}"
                    )

            self._rows = [all_rows[index] for index in normalized]
            self._indices = normalized

    @property
    def metadata_path(self) -> Path:
        return self._metadata_path

    @property
    def repo_id(self) -> str | None:
        return self._repo_id

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[SourceSample]:
        for index in range(len(self._rows)):
            yield self.get_source(index)

    def __getitem__(
        self,
        index: int,
    ) -> SourceSample:
        return self.get_source(index)

    def manifest_index(self, index: int) -> int:
        """Return the original metadata row index for a dataset position."""

        return self._indices[index]

    def get_source(
        self,
        index: int,
    ) -> SourceSample:
        """Return metadata for one source clip without loading audio."""

        if index < 0 or index >= len(self._rows):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        return row_to_source_sample(self._rows[index])

    def load_audio(
        self,
        index: int | SourceSample,
    ) -> tuple[np.ndarray, int]:
        """
        Load mono float32 audio for one source clip.

        This is the first point where ZIP/audio access may occur.
        """

        source = (
            index
            if isinstance(index, SourceSample)
            else self.get_source(index)
        )

        archive_path = self._resolve_archive_path(
            source.archive_name,
        )

        audio_bytes = self._zip_cache.read_member_bytes(
            archive_path,
            source.internal_path,
        )

        return load_mono_wav_bytes(audio_bytes)

    def close(self) -> None:
        """Close any open ZIP handles."""

        self._zip_cache.close()

    def _resolve_archive_path(
        self,
        archive_name: str,
    ) -> Path:
        if archive_name in self._resolved_archives:
            return self._resolved_archives[archive_name]

        if self._archive_dir is not None:
            archive_path = self._archive_dir / archive_name

            if not archive_path.exists():
                raise FileNotFoundError(
                    f"ZIP archive not found in archive_dir: "
                    f"{archive_path}"
                )

            self._resolved_archives[archive_name] = archive_path
            return archive_path

        archive_path = self._download_archive(archive_name)
        self._resolved_archives[archive_name] = archive_path
        return archive_path

    def _download_archive(
        self,
        archive_name: str,
    ) -> Path:
        if self._repo_id is None:
            raise FileNotFoundError(
                f"No local archive available for {archive_name}"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError(
                "huggingface_hub is required to download archives. "
                "Install it or provide archive_dir with local ZIP files."
            ) from error

        downloaded = hf_hub_download(
            repo_id=self._repo_id,
            filename=archive_name,
            repo_type="dataset",
            cache_dir=(
                str(self._cache_dir)
                if self._cache_dir is not None
                else None
            ),
        )

        return Path(downloaded)

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = SIH26_REPO_ID,
        metadata_filename: str = SIH26_METADATA_FILENAME,
        *,
        cache_dir: Path | None = None,
        archive_dir: Path | None = None,
        indices: Sequence[int] | None = None,
    ) -> "ZipManifestDataset":
        """
        Build a dataset by downloading only ``metadata.csv`` from Hugging Face.

        ZIP archives are still accessed lazily via ``load_audio``.
        """

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError(
                "huggingface_hub is required to fetch metadata.csv"
            ) from error

        metadata_file = hf_hub_download(
            repo_id=repo_id,
            filename=metadata_filename,
            repo_type="dataset",
            cache_dir=(
                str(cache_dir)
                if cache_dir is not None
                else None
            ),
        )

        return cls(
            metadata_path=Path(metadata_file),
            repo_id=repo_id,
            archive_dir=archive_dir,
            cache_dir=cache_dir,
            indices=indices,
        )
