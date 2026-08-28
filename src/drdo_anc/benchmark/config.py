from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BenchmarkMode(Enum):
    OFFLINE = "offline"
    STREAMING = "streaming"


@dataclass
class BenchmarkConfig:
    mode: BenchmarkMode
    delay_samples: int = 0
    output_dir: Path | None = None
    save_enhanced: bool = True
    overwrite: bool = False
    measure_timing: bool = True
