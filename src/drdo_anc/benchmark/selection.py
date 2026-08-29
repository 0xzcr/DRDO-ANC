from collections import defaultdict
from pathlib import Path

from drdo_anc.dataset.manifest import (
    SIH26_REPO_ID,
    load_metadata_rows,
    row_to_source_sample,
)
from drdo_anc.dataset.source_pool import (
    ENGLISH_SOURCE,
    NOISE_CATEGORY_SOURCES,
    derive_mixing_seed,
    english_speaker_id,
    is_clean_source,
    is_noise_source,
    source_sample_id_from_row,
)
from drdo_anc.dataset.source_sample import SourceSample

from .case import BenchmarkCase
from .evaluation_manifest import EvaluationManifest


def _select_english_clean_sources(
    clean_rows: list[dict[str, str]],
    num_speakers: int,
) -> list[SourceSample]:
    english_rows = [
        row
        for row in clean_rows
        if row["dataset_source"] == ENGLISH_SOURCE
    ]

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(
        list
    )

    for row in english_rows:
        speaker = english_speaker_id(row)
        by_speaker[speaker].append(row)

    selected: list[SourceSample] = []

    for speaker in sorted(by_speaker):
        rows = sorted(
            by_speaker[speaker],
            key=source_sample_id_from_row,
        )
        selected.append(row_to_source_sample(rows[0]))

        if len(selected) >= num_speakers:
            break

    if len(selected) < num_speakers:
        raise ValueError(
            f"Requested {num_speakers} English speakers, "
            f"but only found {len(selected)}."
        )

    return selected


def _group_noise_rows(
    noise_rows: list[dict[str, str]],
    dataset_source: str,
) -> list[dict[str, str]]:
    return sorted(
        [
            row
            for row in noise_rows
            if row["dataset_source"] == dataset_source
        ],
        key=source_sample_id_from_row,
    )


def _select_noise_source(
    noise_rows: list[dict[str, str]],
    clean_index: int,
    category_index: int,
) -> SourceSample:
    by_subclass: dict[str, list[dict[str, str]]] = defaultdict(
        list
    )

    for row in noise_rows:
        by_subclass[row["inferred_subclass"]].append(row)

    subclasses = sorted(by_subclass)

    if not subclasses:
        raise ValueError("No noise rows available for category.")

    subclass = subclasses[
        (clean_index + category_index) % len(subclasses)
    ]

    rows = sorted(
        by_subclass[subclass],
        key=source_sample_id_from_row,
    )

    return row_to_source_sample(rows[0])


def build_evaluation_manifest(
    metadata_path: Path,
    *,
    selection_seed: int,
    rules_version: str,
    split_name: str,
    num_clean_speakers: int,
    noise_categories: tuple[str, ...],
    snr_levels_db: tuple[float, ...],
    dataset_repo_id: str = SIH26_REPO_ID,
    dataset_revision: str | None = None,
) -> EvaluationManifest:
    """
    Build a deterministic evaluation manifest from metadata only.

    This function does not access ZIP archives or load audio.
    """

    if selection_seed < 0:
        raise ValueError("selection_seed must be non-negative.")

    rows = load_metadata_rows(metadata_path)

    clean_rows = sorted(
        [row for row in rows if is_clean_source(row)],
        key=source_sample_id_from_row,
    )
    noise_rows = sorted(
        [row for row in rows if is_noise_source(row)],
        key=source_sample_id_from_row,
    )

    clean_sources = _select_english_clean_sources(
        clean_rows,
        num_clean_speakers,
    )

    cases: list[BenchmarkCase] = []
    case_counter = 0

    for clean_index, clean_source in enumerate(clean_sources):
        for category_index, category in enumerate(
            noise_categories
        ):
            dataset_source = NOISE_CATEGORY_SOURCES[category]
            category_rows = _group_noise_rows(
                noise_rows,
                dataset_source,
            )
            noise_source = _select_noise_source(
                category_rows,
                clean_index,
                category_index,
            )

            for snr_db in snr_levels_db:
                case_counter += 1
                case_id = (
                    f"benchmark_eval_{rules_version}_"
                    f"{case_counter:05d}"
                )

                cases.append(
                    BenchmarkCase(
                        case_id=case_id,
                        clean_source=clean_source,
                        noise_source=noise_source,
                        noise_category=category,
                        snr_db=float(snr_db),
                        mixing_seed=derive_mixing_seed(
                            case_id
                        ),
                    )
                )

    return EvaluationManifest(
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        selection_seed=selection_seed,
        rules_version=rules_version,
        split_name=split_name,
        snr_levels_db=snr_levels_db,
        noise_categories=noise_categories,
        cases=tuple(cases),
    )


def build_development_manifest(
    metadata_path: Path,
    *,
    dataset_revision: str | None = None,
) -> EvaluationManifest:
    """Build the approved 60-case development evaluation manifest."""

    from drdo_anc.dataset.source_pool import (
        DEVELOPMENT_NOISE_CATEGORIES,
        DEVELOPMENT_NUM_CLEAN_SPEAKERS,
        DEVELOPMENT_SNR_LEVELS_DB,
        RULES_VERSION,
        SELECTION_SEED,
        SPLIT_NAME,
    )

    return build_evaluation_manifest(
        metadata_path,
        selection_seed=SELECTION_SEED,
        rules_version=RULES_VERSION,
        split_name=SPLIT_NAME,
        num_clean_speakers=DEVELOPMENT_NUM_CLEAN_SPEAKERS,
        noise_categories=DEVELOPMENT_NOISE_CATEGORIES,
        snr_levels_db=DEVELOPMENT_SNR_LEVELS_DB,
        dataset_revision=dataset_revision,
    )
