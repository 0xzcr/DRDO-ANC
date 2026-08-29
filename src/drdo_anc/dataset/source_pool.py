import hashlib
from typing import Mapping

from .manifest import make_source_sample_id
from .source_sample import SourceSample


MS_SNSD_SOURCE = "MS-SNSD-Complex-Noise"
ENGLISH_SOURCE = "English-with-various-accents"

MS_SNSD_CLEAN_SUBCLASSES = frozenset(
    {"clean_train", "clean_test"}
)
MS_SNSD_EXCLUDED_NOISE_SUBCLASSES = frozenset(
    {
        "clean_train",
        "clean_test",
        "Test_Triplets",
        "Training_Files",
    }
)

NOISE_CATEGORY_SOURCES: dict[str, str] = {
    "uav_drone": "Drone-Noise-Audio-set",
    "impulsive_firearms": (
        "firearms-audio-dataset-contains-58-guntypes"
    ),
    "vehicle_engine": (
        "Vehicle-Engine-Wind-Electronic-Electrical-Noise"
    ),
    "environmental": "DEMAND-Background-Noise",
    "general_noise": MS_SNSD_SOURCE,
}

DEVELOPMENT_NOISE_CATEGORIES = (
    "uav_drone",
    "impulsive_firearms",
    "vehicle_engine",
)

DEVELOPMENT_SNR_LEVELS_DB = (0.0, 5.0)
DEVELOPMENT_NUM_CLEAN_SPEAKERS = 10

RULES_VERSION = "sih26-eval-v1"
SPLIT_NAME = "benchmark_evaluation"
SELECTION_SEED = 42
MIXING_POLICY_VERSION = "duration-align-v1"


def _as_mapping(
    row_or_sample: Mapping[str, str] | SourceSample,
) -> Mapping[str, str]:
    if isinstance(row_or_sample, SourceSample):
        return {
            "archive_name": row_or_sample.archive_name,
            "internal_path": row_or_sample.internal_path,
            "filename": row_or_sample.filename,
            "parent_folder": row_or_sample.parent_folder,
            "file_size_bytes": str(
                row_or_sample.file_size_bytes
            ),
            "audio_class": row_or_sample.audio_class,
            "dataset_source": row_or_sample.dataset_source,
            "inferred_subclass": row_or_sample.inferred_subclass,
        }

    return row_or_sample


def is_ms_snsd_clean_row(
    row_or_sample: Mapping[str, str] | SourceSample,
) -> bool:
    row = _as_mapping(row_or_sample)

    return (
        row["dataset_source"] == MS_SNSD_SOURCE
        and row["inferred_subclass"] in MS_SNSD_CLEAN_SUBCLASSES
    )


def is_clean_source(
    row_or_sample: Mapping[str, str] | SourceSample,
) -> bool:
    row = _as_mapping(row_or_sample)

    if row["audio_class"] == "clean_speech":
        return True

    return is_ms_snsd_clean_row(row)


def is_noise_source(
    row_or_sample: Mapping[str, str] | SourceSample,
) -> bool:
    row = _as_mapping(row_or_sample)

    if is_ms_snsd_clean_row(row):
        return False

    if row["audio_class"] != "noise":
        return False

    if (
        row["dataset_source"] == MS_SNSD_SOURCE
        and row["inferred_subclass"]
        in MS_SNSD_EXCLUDED_NOISE_SUBCLASSES
    ):
        return False

    return True


def english_speaker_id(
    row_or_sample: Mapping[str, str] | SourceSample,
) -> str:
    row = _as_mapping(row_or_sample)

    if row["dataset_source"] != ENGLISH_SOURCE:
        raise ValueError(
            "english_speaker_id() requires "
            "English-with-various-accents source."
        )

    return row["inferred_subclass"]


def source_sample_id_from_row(
    row: Mapping[str, str],
) -> str:
    return make_source_sample_id(row)


def derive_mixing_seed(case_id: str) -> int:
    digest = hashlib.sha256(
        case_id.encode("utf-8")
    ).digest()

    return int.from_bytes(digest[:8], byteorder="big")
