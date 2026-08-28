from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .config import BenchmarkMode


@dataclass
class SampleResult:
    sample_id: str
    metrics: dict[str, float]
    inference_s: float | None
    rtf: float | None
    enhanced_path: Path | None
    split: str | None = None
    snr_db: float | None = None
    noise_type: str | None = None


@dataclass
class BenchmarkResult:
    model_name: str
    mode: BenchmarkMode
    delay_samples: int
    sample_results: list[SampleResult]

    def _mean_enhanced_metrics(
        self,
        results: list[SampleResult],
    ) -> dict[str, float]:
        if not results:
            return {}

        keys = [
            "enhanced_snr",
            "enhanced_si_sdr",
            "enhanced_stoi",
            "enhanced_pesq",
        ]

        summary: dict[str, float] = {}

        for key in keys:
            values = [
                result.metrics[key]
                for result in results
                if key in result.metrics
            ]

            if values:
                summary[key] = mean(values)

        return summary

    def summary_overall(self) -> dict[str, float]:
        """Mean enhanced metrics across all samples."""

        return self._mean_enhanced_metrics(
            self.sample_results,
        )

    def summary_by_snr(self) -> dict[float, dict[str, float]]:
        """Mean enhanced metrics grouped by ``snr_db``."""

        groups: dict[float, list[SampleResult]] = {}

        for result in self.sample_results:
            if result.snr_db is None:
                continue

            groups.setdefault(
                result.snr_db,
                [],
            ).append(result)

        return {
            snr_db: self._mean_enhanced_metrics(results)
            for snr_db, results in sorted(
                groups.items(),
            )
        }

    def summary_by_noise_type(
        self,
    ) -> dict[str, dict[str, float]]:
        """Mean enhanced metrics grouped by ``noise_type``."""

        groups: dict[str, list[SampleResult]] = {}

        for result in self.sample_results:
            if result.noise_type is None:
                continue

            groups.setdefault(
                result.noise_type,
                [],
            ).append(result)

        return {
            noise_type: self._mean_enhanced_metrics(results)
            for noise_type, results in sorted(
                groups.items(),
            )
        }
