from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceSample:
    """
    One source clip from a ZIP-backed manifest.

    This represents raw source material (clean speech or noise), not a
    benchmark-ready clean/noisy pair. Use ``AudioSample`` only after a
    future mixture-generation step.

    The Hugging Face SIH-26 dataset stores ``audio_class`` in metadata.csv.
    Approximately 24k MS-SNSD files under ``clean_train`` / ``clean_test``
    are labeled ``noise`` in that column. The original ``audio_class`` is
    preserved here; use ``parent_folder`` and ``inferred_subclass`` in a
    future filter to exclude them from the noise pool without rewriting rows.
    """

    sample_id: str

    archive_name: str
    internal_path: str
    filename: str
    parent_folder: str
    file_size_bytes: int
    audio_class: str
    dataset_source: str
    inferred_subclass: str

    tags: dict[str, str] = field(default_factory=dict)
