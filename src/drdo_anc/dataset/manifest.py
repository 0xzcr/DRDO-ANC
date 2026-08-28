import csv
from pathlib import Path
from typing import TextIO


METADATA_COLUMNS = (
    "archive_name",
    "internal_path",
    "filename",
    "parent_folder",
    "file_size_bytes",
    "audio_class",
    "dataset_source",
    "inferred_subclass",
)

SIH26_REPO_ID = "Panav-Payappagoudar/sih-26-processed-audio"
SIH26_METADATA_FILENAME = "metadata.csv"


def read_metadata_rows(
    metadata_file: TextIO,
) -> list[dict[str, str]]:
    """Parse a metadata CSV into row dictionaries."""

    reader = csv.DictReader(metadata_file)

    if reader.fieldnames is None:
        raise ValueError("Metadata CSV has no header row.")

    missing = [
        column
        for column in METADATA_COLUMNS
        if column not in reader.fieldnames
    ]

    if missing:
        raise ValueError(
            "Metadata CSV is missing required columns: "
            f"{missing}"
        )

    return list(reader)


def load_metadata_rows(
    metadata_path: Path,
) -> list[dict[str, str]]:
    """Load metadata rows from a local CSV file."""

    with metadata_path.open(
        encoding="utf-8",
        newline="",
    ) as metadata_file:
        return read_metadata_rows(metadata_file)


def make_source_sample_id(row: dict[str, str]) -> str:
    """Build a stable identifier for one manifest row."""

    return (
        f"{row['dataset_source']}/"
        f"{row['internal_path']}"
    )


def row_to_source_sample(
    row: dict[str, str],
) -> "SourceSample":
    """Convert one metadata row into a ``SourceSample``."""

    from .source_sample import SourceSample

    return SourceSample(
        sample_id=make_source_sample_id(row),
        archive_name=row["archive_name"],
        internal_path=row["internal_path"],
        filename=row["filename"],
        parent_folder=row["parent_folder"],
        file_size_bytes=int(row["file_size_bytes"]),
        audio_class=row["audio_class"],
        dataset_source=row["dataset_source"],
        inferred_subclass=row["inferred_subclass"],
        tags={
            key: value
            for key, value in row.items()
            if key not in METADATA_COLUMNS
        },
    )
