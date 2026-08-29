from dataclasses import dataclass

from drdo_anc.dataset.source_sample import SourceSample


@dataclass(frozen=True)
class BenchmarkCase:
    """Reproducible benchmark case metadata without stored audio."""

    case_id: str
    clean_source: SourceSample
    noise_source: SourceSample
    noise_category: str
    snr_db: float
    mixing_seed: int
    mixing_policy_version: str = "duration-align-v1"
