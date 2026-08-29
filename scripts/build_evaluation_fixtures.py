import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from drdo_anc.dataset.manifest import METADATA_COLUMNS


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "evaluation_manifest"
METADATA_PATH = FIXTURE_DIR / "metadata.csv"
ARCHIVE_PATH = FIXTURE_DIR / "test_archive.zip"
ARCHIVE_NAME = "test_archive.zip"


def build_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()

    sample_rate = 16_000
    rows: list[list[str]] = []

    for speaker_index in range(10):
        speaker = f"p{100 + speaker_index:03d}"
        internal_path = f"english/{speaker}/clip.wav"
        audio = np.linspace(
            -0.1,
            0.1,
            num=sample_rate + speaker_index * 100,
            dtype=np.float32,
        )
        _write_zip_member(internal_path, audio, sample_rate)
        rows.append(
            [
                ARCHIVE_NAME,
                internal_path,
                "clip.wav",
                speaker,
                str(32044 + speaker_index * 100),
                "clean_speech",
                "English-with-various-accents",
                speaker,
            ]
        )

    ms_clean = [
        "MS-SNSD-Complex-Noise/clean_train/p999_001.wav",
        "MS-SNSD-Complex-Noise/clean_test/p998_001.wav",
    ]
    for path in ms_clean:
        audio = np.ones(8000, dtype=np.float32) * 0.01
        _write_zip_member(path, audio, sample_rate)
        subclass = path.split("/")[1]
        rows.append(
            [
                ARCHIVE_NAME,
                path,
                path.split("/")[-1],
                subclass,
                "16044",
                "noise",
                "MS-SNSD-Complex-Noise",
                subclass,
            ]
        )

    noise_specs = [
        ("Drone-Noise-Audio-set", "drone", "drone_a.wav", 4000),
        ("Drone-Noise-Audio-set", "drone", "drone_b.wav", 6000),
        ("firearms-audio-dataset-contains-58-guntypes", "ak-47", "ak.wav", 2000),
        ("firearms-audio-dataset-contains-58-guntypes", "m16", "m16.wav", 2500),
        ("Vehicle-Engine-Wind-Electronic-Electrical-Noise", "1", "veh1.wav", 8000),
        ("Vehicle-Engine-Wind-Electronic-Electrical-Noise", "2", "veh2.wav", 12000),
        ("MS-SNSD-Complex-Noise", "noise_train", "noise_train.wav", 5000),
        ("MS-SNSD-Complex-Noise", "Test_Triplets", "triplet.wav", 5000),
    ]

    for source, subclass, filename, length in noise_specs:
        internal_path = f"noise/{source}/{subclass}/{filename}"
        audio = np.random.default_rng(length).standard_normal(
            length,
            dtype=np.float32,
        )
        _write_zip_member(internal_path, audio, sample_rate)
        rows.append(
            [
                ARCHIVE_NAME,
                internal_path,
                filename,
                subclass,
                str(length * 2 + 44),
                "noise",
                source,
                subclass,
            ]
        )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(METADATA_COLUMNS)
        writer.writerows(rows)


def _write_zip_member(
    internal_path: str,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    memory = io.BytesIO()
    sf.write(memory, audio, sample_rate, format="WAV")
    payload = memory.getvalue()

    mode = "a" if ARCHIVE_PATH.exists() else "w"
    with zipfile.ZipFile(
        ARCHIVE_PATH,
        mode=mode,
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(internal_path, payload)


if __name__ == "__main__":
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    build_fixtures()
    print(f"Wrote {METADATA_PATH} and {ARCHIVE_PATH}")
