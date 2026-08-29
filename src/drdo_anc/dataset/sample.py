from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AudioSample:
    """Metadata record for one evaluation utterance."""

    sample_id: str

    noisy_path: Path | None = None
    clean_path: Path | None = None
    enhanced_path: Path | None = None

    sample_rate: int | None = None
    num_samples: int | None = None
    duration_s: float | None = None

    split: Literal["train", "validation", "test"] | None = None
    snr_db: float | None = None
    noise_type: str | None = None
    speaker_id: str | None = None
    scene_id: str | None = None

    tags: dict[str, str] = field(default_factory=dict)
