import zipfile
from pathlib import Path


class ZipArchiveCache:
    """
    Lazily open ZIP archives and read individual members.

    Archives are expected to already exist on disk (for example after
    ``huggingface_hub.hf_hub_download`` fetches a single ``.zip`` file).
    This class does not download archives and does not extract all members.
    """

    def __init__(self) -> None:
        self._open_archives: dict[Path, zipfile.ZipFile] = {}

    def read_member_bytes(
        self,
        archive_path: Path,
        internal_path: str,
    ) -> bytes:
        """Read one member from a ZIP archive without extracting it."""

        if not archive_path.exists():
            raise FileNotFoundError(
                f"ZIP archive not found: {archive_path}"
            )

        archive = self._get_archive(archive_path)

        try:
            with archive.open(internal_path) as member:
                return member.read()
        except KeyError as error:
            raise FileNotFoundError(
                f"Member not found in {archive_path.name}: "
                f"{internal_path}"
            ) from error

    def close(self) -> None:
        """Close any open ZIP handles."""

        for archive in self._open_archives.values():
            archive.close()

        self._open_archives.clear()

    def _get_archive(
        self,
        archive_path: Path,
    ) -> zipfile.ZipFile:
        resolved = archive_path.resolve()

        if resolved not in self._open_archives:
            self._open_archives[resolved] = zipfile.ZipFile(
                resolved,
                mode="r",
            )

        return self._open_archives[resolved]

    def __del__(self) -> None:
        self.close()
