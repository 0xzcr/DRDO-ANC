import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from drdo_anc.audio.resampling import resample_mono
from drdo_anc.dataset.source_pool import english_speaker_id
from drdo_anc.dataset.zip_manifest_dataset import (
    ZipManifestDataset,
)
from drdo_anc.evaluation import apply_evaluation_delay, evaluate_pair

from .case import BenchmarkCase
from .config import BenchmarkMode, STREAMING_CHUNK_SIZES
from .evaluation_manifest import EvaluationManifest
from .mixture import MixtureGenerator

if TYPE_CHECKING:
    from drdo_anc.enhancement.base import Enhancer

STREAMING_DELAY_SAMPLES = 1440


@dataclass(frozen=True)
class ManifestCaseResult:
    case_id: str
    clean_source_id: str
    noise_source_id: str
    noise_category: str
    snr_db: float
    model_name: str
    mode: str
    delay_samples: int
    sample_rate: int
    num_samples: int
    achieved_mixture_snr_db: float | None
    snr: float | None
    si_sdr: float | None
    stoi: float | None
    pesq: float | None
    inference_s: float | None
    rtf: float | None
    status: str = "success"
    error: str | None = None


@dataclass
class ManifestBenchmarkReport:
    manifest_rules_version: str
    manifest_split_name: str
    modes: list[str]
    case_results: list[ManifestCaseResult] = field(
        default_factory=list,
    )

    def successful_results(self) -> list[ManifestCaseResult]:
        return [
            result
            for result in self.case_results
            if result.status == "success"
        ]

    def summary_overall(self) -> dict[str, float]:
        return _mean_metrics(self.successful_results())

    def summary_by_snr(self) -> dict[str, dict[str, float]]:
        groups: dict[float, list[ManifestCaseResult]] = {}

        for result in self.successful_results():
            groups.setdefault(result.snr_db, []).append(result)

        return {
            _format_snr_key(snr_db): _mean_metrics(results)
            for snr_db, results in sorted(groups.items())
        }

    def summary_by_noise_category(
        self,
    ) -> dict[str, dict[str, float]]:
        groups: dict[str, list[ManifestCaseResult]] = {}

        for result in self.successful_results():
            groups.setdefault(
                result.noise_category,
                [],
            ).append(result)

        return {
            category: _mean_metrics(results)
            for category, results in sorted(groups.items())
        }

    def summary_rtf(self) -> dict[str, float]:
        rtfs = [
            result.rtf
            for result in self.successful_results()
            if result.rtf is not None
        ]

        if not rtfs:
            return {}

        return {
            "median_rtf": median(rtfs),
            "mean_rtf": mean(rtfs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_rules_version": self.manifest_rules_version,
            "manifest_split_name": self.manifest_split_name,
            "modes": self.modes,
            "successful_cases": len(self.successful_results()),
            "failed_cases": len(self.case_results)
            - len(self.successful_results()),
            "case_results": [
                asdict(result) for result in self.case_results
            ],
            "summary_overall": self.summary_overall(),
            "summary_by_snr": self.summary_by_snr(),
            "summary_by_noise_category": (
                self.summary_by_noise_category()
            ),
            "summary_rtf": self.summary_rtf(),
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
            )
            handle.write("\n")

    def save_csv(self, path: Path) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "case_id",
            "clean_source_id",
            "noise_source_id",
            "noise_category",
            "snr_db",
            "model_name",
            "mode",
            "delay_samples",
            "sample_rate",
            "num_samples",
            "achieved_mixture_snr_db",
            "snr",
            "si_sdr",
            "stoi",
            "pesq",
            "inference_s",
            "rtf",
            "status",
            "error",
        ]

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for result in self.case_results:
                writer.writerow(asdict(result))


def validate_development_manifest(
    manifest: EvaluationManifest,
) -> None:
    """Validate the approved 60-case development manifest."""

    if len(manifest.cases) != 60:
        raise ValueError(
            f"Expected 60 cases, found {len(manifest.cases)}."
        )

    speakers = {
        english_speaker_id(case.clean_source)
        for case in manifest.cases
    }
    categories = {case.noise_category for case in manifest.cases}
    snr_counts = Counter(case.snr_db for case in manifest.cases)
    case_ids = [case.case_id for case in manifest.cases]

    if len(speakers) != 10:
        raise ValueError(
            f"Expected 10 clean speakers, found {len(speakers)}."
        )

    if len(categories) != 3:
        raise ValueError(
            f"Expected 3 noise categories, found {len(categories)}."
        )

    if snr_counts.get(0.0) != 30 or snr_counts.get(5.0) != 30:
        raise ValueError(
            "Expected 30 cases at 0 dB and 30 cases at +5 dB."
        )

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs found in manifest.")


def select_smoke_cases(
    manifest: EvaluationManifest,
) -> tuple[BenchmarkCase, ...]:
    """
    Select the approved smoke subset:

    1 clean speaker × 1 noise category × 2 SNR levels = 2 cases.
    """

    if not manifest.cases:
        raise ValueError("Manifest has no cases.")

    first_clean_id = manifest.cases[0].clean_source.sample_id
    first_category = manifest.cases[0].noise_category

    smoke_cases = tuple(
        case
        for case in manifest.cases
        if case.clean_source.sample_id == first_clean_id
        and case.noise_category == first_category
    )

    if len(smoke_cases) != 2:
        raise ValueError(
            f"Expected 2 smoke cases, found {len(smoke_cases)}."
        )

    return smoke_cases


def resample_mixture_for_enhancer(
    clean: np.ndarray,
    noisy: np.ndarray,
    source_sample_rate: int,
    enhancer_sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Resample a generated 16 kHz mixture to the enhancer input rate.

  * Resampling happens here at the model-input boundary.
  * ``ZipManifestDataset`` and ``MixtureGenerator`` remain at native rate.
    """

    clean_resampled = resample_mono(
        clean,
        source_sample_rate,
        enhancer_sample_rate,
    )
    noisy_resampled = resample_mono(
        noisy,
        source_sample_rate,
        enhancer_sample_rate,
    )

    return (
        clean_resampled,
        noisy_resampled,
        enhancer_sample_rate,
    )


def delay_samples_for_mode(
    mode: BenchmarkMode,
    *,
    override_delay_samples: int | None = None,
) -> int:
    if override_delay_samples is not None:
        return override_delay_samples

    if mode == BenchmarkMode.OFFLINE:
        return 0

    if mode == BenchmarkMode.STREAMING:
        return STREAMING_DELAY_SAMPLES

    raise ValueError(f"Unsupported mode: {mode}")


class ManifestBenchmarkRunner:
    """
    Run DF3 evaluation over manifest-defined benchmark cases.

    Timing includes only model inference (offline ``process`` or streaming
    ``process_stream`` + ``flush``). It excludes manifest parsing, ZIP I/O,
    mixture generation, and resampling.
    """

    def __init__(
        self,
        enhancer: "Enhancer",
        mixture_generator: MixtureGenerator,
        *,
        modes: tuple[BenchmarkMode, ...] = (
            BenchmarkMode.OFFLINE,
            BenchmarkMode.STREAMING,
        ),
        measure_timing: bool = True,
    ) -> None:
        self._enhancer = enhancer
        self._mixture_generator = mixture_generator
        self._modes = modes
        self._measure_timing = measure_timing

    def run(
        self,
        manifest: EvaluationManifest,
        cases: tuple[BenchmarkCase, ...] | None = None,
        *,
        streaming_delay_override: int | None = None,
    ) -> ManifestBenchmarkReport:
        selected_cases = cases or manifest.cases

        report = ManifestBenchmarkReport(
            manifest_rules_version=manifest.rules_version,
            manifest_split_name=manifest.split_name,
            modes=[mode.value for mode in self._modes],
        )

        for case_index, case in enumerate(selected_cases, start=1):
            print(
                f"Case {case_index}/{len(selected_cases)}: "
                f"{case.case_id}",
                flush=True,
            )

            try:
                mixture = self._mixture_generator.generate(case)
            except Exception as exc:
                for mode in self._modes:
                    report.case_results.append(
                        _failed_case_result(
                            case,
                            self._enhancer.name(),
                            mode,
                            delay_samples_for_mode(
                                mode,
                                override_delay_samples=(
                                    streaming_delay_override
                                    if mode
                                    == BenchmarkMode.STREAMING
                                    else None
                                ),
                            ),
                            self._enhancer.sample_rate(),
                            error=(
                                f"mixture_generation: {exc}"
                            ),
                        )
                    )
                continue

            clean_ref, noisy_input, eval_sample_rate = (
                resample_mixture_for_enhancer(
                    mixture.clean,
                    mixture.noisy,
                    mixture.sample_rate,
                    self._enhancer.sample_rate(),
                )
            )

            for mode in self._modes:
                delay_samples = delay_samples_for_mode(
                    mode,
                    override_delay_samples=(
                        streaming_delay_override
                        if mode == BenchmarkMode.STREAMING
                        else None
                    ),
                )

                try:
                    result = self._evaluate_case(
                        case,
                        clean_ref,
                        noisy_input,
                        eval_sample_rate,
                        mode,
                        delay_samples,
                        mixture.achieved_snr_db,
                    )
                except Exception as exc:
                    result = _failed_case_result(
                        case,
                        self._enhancer.name(),
                        mode,
                        delay_samples,
                        eval_sample_rate,
                        error=str(exc),
                    )

                report.case_results.append(result)

        return report

    def _evaluate_case(
        self,
        case: BenchmarkCase,
        clean: np.ndarray,
        noisy: np.ndarray,
        sample_rate: int,
        mode: BenchmarkMode,
        delay_samples: int,
        achieved_mixture_snr_db: float,
    ) -> ManifestCaseResult:
        self._enhancer.reset()

        inference_s: float | None = None

        if self._measure_timing:
            start = time.perf_counter()

        if mode == BenchmarkMode.OFFLINE:
            enhanced = self._enhance_offline(noisy)
        elif mode == BenchmarkMode.STREAMING:
            enhanced = self._enhance_streaming(noisy)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if self._measure_timing:
            inference_s = time.perf_counter() - start

        if not np.isfinite(enhanced).all():
            raise ValueError(
                f"Enhanced output contains non-finite values "
                f"for {case.case_id}."
            )

        if len(enhanced) != len(noisy):
            raise RuntimeError(
                f"Enhanced length mismatch for {case.case_id}: "
                f"input={len(noisy)}, output={len(enhanced)}"
            )

        clean_aligned, _noisy_aligned, enhanced_aligned = (
            apply_evaluation_delay(
                clean,
                noisy,
                enhanced,
                delay_samples,
            )
        )

        metrics = evaluate_pair(
            clean_aligned,
            enhanced_aligned,
            sample_rate,
        )

        duration_s = len(noisy) / sample_rate
        rtf = (
            duration_s / inference_s
            if inference_s and inference_s > 0
            else None
        )

        return ManifestCaseResult(
            case_id=case.case_id,
            clean_source_id=case.clean_source.sample_id,
            noise_source_id=case.noise_source.sample_id,
            noise_category=case.noise_category,
            snr_db=case.snr_db,
            model_name=self._enhancer.name(),
            mode=mode.value,
            delay_samples=delay_samples,
            sample_rate=sample_rate,
            num_samples=len(noisy),
            achieved_mixture_snr_db=achieved_mixture_snr_db,
            snr=metrics["snr"],
            si_sdr=metrics["si_sdr"],
            stoi=metrics["stoi"],
            pesq=metrics["pesq"],
            inference_s=inference_s,
            rtf=rtf,
        )

    def _enhance_offline(
        self,
        noisy: np.ndarray,
    ) -> np.ndarray:
        audio = torch.from_numpy(noisy).float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        enhanced_tensor = self._enhancer.process(audio)

        return _tensor_to_mono_numpy(enhanced_tensor)

    def _enhance_streaming(
        self,
        noisy: np.ndarray,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        position = 0
        chunk_index = 0

        while position < len(noisy):
            chunk_size = STREAMING_CHUNK_SIZES[
                chunk_index % len(STREAMING_CHUNK_SIZES)
            ]

            end = min(position + chunk_size, len(noisy))
            chunk = noisy[position:end]

            output_tensor = self._enhancer.process_stream(
                torch.from_numpy(chunk).float()
            )

            output = _tensor_to_mono_numpy(output_tensor)

            if len(output) > 0:
                outputs.append(output)

            position = end
            chunk_index += 1

        flush_output = _tensor_to_mono_numpy(
            self._enhancer.flush(),
        )

        if len(flush_output) > 0:
            outputs.append(flush_output)

        if outputs:
            enhanced = np.concatenate(outputs)
        else:
            enhanced = np.empty(0, dtype=np.float32)

        return enhanced.astype(np.float32, copy=False)


def build_manifest_dataset(
    manifest: EvaluationManifest,
    metadata_path: Path,
    *,
    archive_dir: Path | None = None,
) -> ZipManifestDataset:
    return ZipManifestDataset(
        metadata_path=metadata_path,
        repo_id=manifest.dataset_repo_id,
        archive_dir=archive_dir,
    )


def _mean_metrics(
    results: list[ManifestCaseResult],
) -> dict[str, float]:
    if not results:
        return {}

    summary: dict[str, float] = {}

    for key in ("snr", "si_sdr", "stoi", "pesq"):
        values = [
            getattr(result, key)
            for result in results
            if getattr(result, key) is not None
        ]

        if values:
            summary[f"mean_{key}"] = mean(values)

    return summary


def _format_snr_key(snr_db: float) -> str:
    if snr_db == int(snr_db):
        sign = "+" if snr_db > 0 else ""
        return f"{sign}{int(snr_db)} dB"

    return f"{snr_db} dB"


def _failed_case_result(
    case: BenchmarkCase,
    model_name: str,
    mode: BenchmarkMode,
    delay_samples: int,
    sample_rate: int,
    *,
    error: str,
) -> ManifestCaseResult:
    return ManifestCaseResult(
        case_id=case.case_id,
        clean_source_id=case.clean_source.sample_id,
        noise_source_id=case.noise_source.sample_id,
        noise_category=case.noise_category,
        snr_db=case.snr_db,
        model_name=model_name,
        mode=mode.value,
        delay_samples=delay_samples,
        sample_rate=sample_rate,
        num_samples=0,
        achieved_mixture_snr_db=None,
        snr=None,
        si_sdr=None,
        stoi=None,
        pesq=None,
        inference_s=None,
        rtf=None,
        status="failed",
        error=error,
    )


def _tensor_to_mono_numpy(
    audio: torch.Tensor,
) -> np.ndarray:
    array = (
        audio.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError("Expected mono audio tensor.")

        array = array.squeeze(0)

    return array
