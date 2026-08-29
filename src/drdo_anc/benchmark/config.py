from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BenchmarkMode(Enum):
    OFFLINE = "offline"
    STREAMING = "streaming"


STREAMING_CHUNK_SIZES = (
    300,
    700,
    250,
    1000,
    137,
    911,
    2048,
    512,
    1536,
    800,
    1200,
)


@dataclass
class BenchmarkConfig:
    mode: BenchmarkMode
    delay_samples: int = 0
    output_dir: Path | None = None
    save_enhanced: bool = True
    overwrite: bool = False
    measure_timing: bool = True
