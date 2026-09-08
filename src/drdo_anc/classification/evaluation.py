"""Benchmark evaluation helpers for the noise classifier."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean

import numpy as np

from drdo_anc.benchmark.case import BenchmarkCase
from drdo_anc.benchmark.config import STREAMING_CHUNK_SIZES
from drdo_anc.benchmark.evaluation_manifest import EvaluationManifest
from drdo_anc.benchmark.mixture import MixtureGenerator
from drdo_anc.dataset.zip_manifest_dataset import ZipManifestDataset

from .categories import DEFENCE_NOISE_CATEGORIES, NOISE_CLASSES, UNKNOWN_CLASS
from .classifier import ClassificationResult, NoiseClassifier


@dataclass(frozen=True)
class ClassifierCaseResult:
    case_id: str
    noise_source_id: str
    ground_truth: str
    predicted_class: str
    probabilities: dict[str, float]
    inference_s: float
    num_samples: int
    sample_rate: int


@dataclass
class ClassifierBenchmarkReport:
    manifest_rules_version: str
    manifest_split_name: str
    case_results: list[ClassifierCaseResult] = field(
        default_factory=list,
    )

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {
            truth: {pred: 0 for pred in NOISE_CLASSES}
            for truth in DEFENCE_NOISE_CATEGORIES
        }

        for result in self.case_results:
            matrix[result.ground_truth][result.predicted_class] += 1

        return matrix

    def per_class_metrics(self) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}

        for truth in DEFENCE_NOISE_CATEGORIES:
            predicted_positive = [
                result
                for result in self.case_results
                if result.predicted_class == truth
            ]
            actual_positive = [
                result
                for result in self.case_results
                if result.ground_truth == truth
            ]
            true_positive = sum(
                1
                for result in self.case_results
                if result.ground_truth == truth
                and result.predicted_class == truth
            )

            precision = (
                true_positive / len(predicted_positive)
                if predicted_positive
                else 0.0
            )
            recall = (
                true_positive / len(actual_positive)
                if actual_positive
                else 0.0
            )
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )

            metrics[truth] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(len(actual_positive)),
            }

        return metrics

    def overall_accuracy(self) -> float:
        if not self.case_results:
            return 0.0

        correct = sum(
            1
            for result in self.case_results
            if result.ground_truth == result.predicted_class
        )

        return correct / len(self.case_results)

    def unknown_predictions(self) -> list[ClassifierCaseResult]:
        return [
            result
            for result in self.case_results
            if result.predicted_class == UNKNOWN_CLASS
        ]

    def timing_summary(self) -> dict[str, float]:
        if not self.case_results:
            return {}

        durations = [result.inference_s for result in self.case_results]
        audio_seconds = [
            result.num_samples / result.sample_rate
            for result in self.case_results
            if result.sample_rate > 0
        ]

        return {
            "mean_inference_s": mean(durations),
            "total_inference_s": float(sum(durations)),
            "mean_audio_s": mean(audio_seconds) if audio_seconds else 0.0,
            "mean_rtf": (
                mean(
                    duration / audio_s
                    for duration, audio_s in zip(durations, audio_seconds)
                    if audio_s > 0.0
                )
                if audio_seconds
                else 0.0
            ),
        }


def classify_case_noise(
    classifier: NoiseClassifier,
    dataset: ZipManifestDataset,
    case: BenchmarkCase,
    *,
    chunk_sizes: tuple[int, ...] = STREAMING_CHUNK_SIZES,
) -> tuple[ClassificationResult, float]:
    """
    Classify the benchmark noise source for one case using streaming chunks.

    Ground-truth labels come from ``case.noise_category``. Audio is loaded
    from the deterministic noise source referenced by the case.
    """

    noise, sample_rate = dataset.load_audio(case.noise_source)

    if sample_rate != classifier.sample_rate:
        raise ValueError(
            f"Sample rate mismatch for {case.case_id}: "
            f"noise={sample_rate}, classifier={classifier.sample_rate}"
        )

    classifier.reset()

    start = time.perf_counter()

    offset = 0
    chunk_index = 0

    while offset < noise.size:
        chunk_size = chunk_sizes[chunk_index % len(chunk_sizes)]
        chunk = noise[offset : offset + chunk_size]
        classifier.process_chunk(chunk)
        offset += chunk_size
        chunk_index += 1

    result = classifier.classify_buffered()
    elapsed = time.perf_counter() - start

    return result, elapsed


def run_classifier_benchmark(
    manifest: EvaluationManifest,
    dataset: ZipManifestDataset,
    *,
    classifier: NoiseClassifier | None = None,
    chunk_sizes: tuple[int, ...] = STREAMING_CHUNK_SIZES,
) -> ClassifierBenchmarkReport:
    """Evaluate the classifier across all manifest cases."""

    if classifier is None:
        classifier = NoiseClassifier()

    report = ClassifierBenchmarkReport(
        manifest_rules_version=manifest.rules_version,
        manifest_split_name=manifest.split_name,
    )

    for case in manifest.cases:
        result, elapsed = classify_case_noise(
            classifier,
            dataset,
            case,
            chunk_sizes=chunk_sizes,
        )
        noise, sample_rate = dataset.load_audio(case.noise_source)

        report.case_results.append(
            ClassifierCaseResult(
                case_id=case.case_id,
                noise_source_id=case.noise_source.sample_id,
                ground_truth=case.noise_category,
                predicted_class=result.predicted_class,
                probabilities=result.probabilities,
                inference_s=elapsed,
                num_samples=int(noise.size),
                sample_rate=sample_rate,
            )
        )

    return report
