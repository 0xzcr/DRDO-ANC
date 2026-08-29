from dataclasses import dataclass

import numpy as np

from drdo_anc.audio.mixing import create_mixture
from drdo_anc.dataset.zip_manifest_dataset import (
    ZipManifestDataset,
)

from .case import BenchmarkCase


@dataclass(frozen=True)
class MixtureResult:
    """Generated mixture audio for one benchmark case."""

    case_id: str
    clean: np.ndarray
    noisy: np.ndarray
    sample_rate: int
    achieved_snr_db: float


class MixtureGenerator:
    """
    Deterministically generate noisy mixtures for benchmark cases.

    Mixtures are cached in memory keyed by case identity and mixing policy.
    """

    def __init__(
        self,
        dataset: ZipManifestDataset,
    ) -> None:
        self._dataset = dataset
        self._cache: dict[tuple[str, str], MixtureResult] = {}

    def generate(
        self,
        case: BenchmarkCase,
    ) -> MixtureResult:
        cache_key = (
            case.case_id,
            case.mixing_policy_version,
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        clean, sample_rate = self._dataset.load_audio(
            case.clean_source
        )
        noise, noise_sr = self._dataset.load_audio(
            case.noise_source
        )

        if sample_rate != noise_sr:
            raise ValueError(
                f"Sample rate mismatch for {case.case_id}: "
                f"clean={sample_rate}, noise={noise_sr}"
            )

        noisy, _scaled_noise, achieved_snr = create_mixture(
            clean,
            noise,
            case.snr_db,
            case.mixing_seed,
        )

        result = MixtureResult(
            case_id=case.case_id,
            clean=clean,
            noisy=noisy,
            sample_rate=sample_rate,
            achieved_snr_db=achieved_snr,
        )

        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()
