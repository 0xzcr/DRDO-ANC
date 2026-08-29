import json
from dataclasses import dataclass
from pathlib import Path

from drdo_anc.dataset.source_sample import SourceSample

from .case import BenchmarkCase

def _source_sample_to_dict(
    source: SourceSample,
) -> dict:
    return {
        "sample_id": source.sample_id,
        "archive_name": source.archive_name,
        "internal_path": source.internal_path,
        "filename": source.filename,
        "parent_folder": source.parent_folder,
        "file_size_bytes": source.file_size_bytes,
        "audio_class": source.audio_class,
        "dataset_source": source.dataset_source,
        "inferred_subclass": source.inferred_subclass,
        "tags": dict(source.tags),
    }


def _source_sample_from_dict(
    payload: dict,
) -> SourceSample:
    return SourceSample(
        sample_id=payload["sample_id"],
        archive_name=payload["archive_name"],
        internal_path=payload["internal_path"],
        filename=payload["filename"],
        parent_folder=payload["parent_folder"],
        file_size_bytes=int(payload["file_size_bytes"]),
        audio_class=payload["audio_class"],
        dataset_source=payload["dataset_source"],
        inferred_subclass=payload["inferred_subclass"],
        tags=dict(payload.get("tags", {})),
    )


def _benchmark_case_to_dict(
    case: BenchmarkCase,
) -> dict:
    return {
        "case_id": case.case_id,
        "clean_source": _source_sample_to_dict(
            case.clean_source
        ),
        "noise_source": _source_sample_to_dict(
            case.noise_source
        ),
        "noise_category": case.noise_category,
        "snr_db": case.snr_db,
        "mixing_seed": case.mixing_seed,
        "mixing_policy_version": case.mixing_policy_version,
    }


def _benchmark_case_from_dict(
    payload: dict,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=payload["case_id"],
        clean_source=_source_sample_from_dict(
            payload["clean_source"]
        ),
        noise_source=_source_sample_from_dict(
            payload["noise_source"]
        ),
        noise_category=payload["noise_category"],
        snr_db=float(payload["snr_db"]),
        mixing_seed=int(payload["mixing_seed"]),
        mixing_policy_version=payload["mixing_policy_version"],
    )


@dataclass(frozen=True)
class EvaluationManifest:
    """Fixed evaluation manifest for reproducible benchmark cases."""

    dataset_repo_id: str
    dataset_revision: str | None
    selection_seed: int
    rules_version: str
    split_name: str
    snr_levels_db: tuple[float, ...]
    noise_categories: tuple[str, ...]
    cases: tuple[BenchmarkCase, ...]

    def to_dict(self) -> dict:
        return {
            "manifest_version": "sih26-benchmark-eval-v1",
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_revision": self.dataset_revision,
            "selection_seed": self.selection_seed,
            "rules_version": self.rules_version,
            "split_name": self.split_name,
            "snr_levels_db": list(self.snr_levels_db),
            "noise_categories": list(self.noise_categories),
            "cases": [
                _benchmark_case_to_dict(case)
                for case in self.cases
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EvaluationManifest":
        return cls(
            dataset_repo_id=payload["dataset_repo_id"],
            dataset_revision=payload.get("dataset_revision"),
            selection_seed=int(payload["selection_seed"]),
            rules_version=payload["rules_version"],
            split_name=payload["split_name"],
            snr_levels_db=tuple(
                float(value)
                for value in payload["snr_levels_db"]
            ),
            noise_categories=tuple(
                payload["noise_categories"]
            ),
            cases=tuple(
                _benchmark_case_from_dict(case)
                for case in payload["cases"]
            ),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
            )
            handle.write("\n")

    @classmethod
    def load_json(cls, path: Path) -> "EvaluationManifest":
        with path.open(
            encoding="utf-8",
        ) as handle:
            return cls.from_dict(json.load(handle))