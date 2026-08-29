import os
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from drdo_anc.dataset import (
    METADATA_COLUMNS,
    ZipManifestDataset,
    load_metadata_rows,
    row_to_source_sample,
)
from drdo_anc.dataset.manifest import make_source_sample_id


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "zip_manifest"
METADATA_PATH = FIXTURE_DIR / "metadata.csv"
ARCHIVE_PATH = FIXTURE_DIR / "test_archive.zip"
ARCHIVE_NAME = "test_archive.zip"


def _build_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    sample_rate = 16_000
    clean_audio = np.linspace(
        -0.25,
        0.25,
        num=sample_rate,
        dtype=np.float32,
    )
    noise_audio = np.linspace(
        0.5,
        -0.5,
        num=sample_rate // 2,
        dtype=np.float32,
    )

    clean_internal = "fixture/clean/example_clean.wav"
    noise_internal = "fixture/noise/example_noise.wav"

    with zipfile.ZipFile(
        ARCHIVE_PATH,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        import io

        for internal_path, audio in (
            (clean_internal, clean_audio),
            (noise_internal, noise_audio),
        ):
            memory = io.BytesIO()
            sf.write(
                memory,
                audio,
                sample_rate,
                format="WAV",
            )
            archive.writestr(
                internal_path,
                memory.getvalue(),
            )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as metadata_file:
        metadata_file.write(",".join(METADATA_COLUMNS) + "\n")
        metadata_file.write(
            f"{ARCHIVE_NAME},{clean_internal},example_clean.wav,clean,12345,clean_speech,fixture-clean,clean\n"
        )
        metadata_file.write(
            f"{ARCHIVE_NAME},{noise_internal},example_noise.wav,noise,6789,noise,fixture-noise,impulse\n"
        )


def setup_module() -> None:
    if not METADATA_PATH.exists() or not ARCHIVE_PATH.exists():
        _build_fixtures()


def test_metadata_parsing_and_row_count() -> None:
    rows = load_metadata_rows(METADATA_PATH)

    assert len(rows) == 2
    assert list(rows[0].keys()) == list(METADATA_COLUMNS)


def test_source_sample_preserves_metadata() -> None:
    rows = load_metadata_rows(METADATA_PATH)
    source = row_to_source_sample(rows[0])

    assert source.sample_id == make_source_sample_id(rows[0])
    assert source.archive_name == ARCHIVE_NAME
    assert source.internal_path == "fixture/clean/example_clean.wav"
    assert source.audio_class == "clean_speech"
    assert source.dataset_source == "fixture-clean"
    assert source.inferred_subclass == "clean"
    assert source.file_size_bytes == 12345


def test_dataset_index_without_audio_access() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    assert len(dataset) == 2
    assert dataset.get_source(0).filename == "example_clean.wav"
    assert dataset.manifest_index(1) == 1


def test_subset_indices() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
        indices=[1],
    )

    assert len(dataset) == 1
    assert dataset.get_source(0).filename == "example_noise.wav"
    assert dataset.manifest_index(0) == 1


def test_lazy_audio_load_clean() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    audio, sample_rate = dataset.load_audio(0)

    assert sample_rate == 16_000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 16_000


def test_lazy_audio_load_noise_subset() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
        indices=[1],
    )

    audio, sample_rate = dataset.load_audio(dataset.get_source(0))

    assert sample_rate == 16_000
    assert len(audio) == 8_000


def test_missing_archive_raises() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR / "missing",
    )

    try:
        dataset.load_audio(0)
    except FileNotFoundError as error:
        assert "ZIP archive not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_missing_member_raises() -> None:
    rows = load_metadata_rows(METADATA_PATH)
    rows[0] = {
        **rows[0],
        "internal_path": "fixture/missing.wav",
    }

    broken_metadata = FIXTURE_DIR / "broken_metadata.csv"
    with broken_metadata.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as metadata_file:
        metadata_file.write(",".join(METADATA_COLUMNS) + "\n")
        metadata_file.write(
            ",".join(rows[0][column] for column in METADATA_COLUMNS)
            + "\n"
        )

    dataset = ZipManifestDataset(
        metadata_path=broken_metadata,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    try:
        dataset.load_audio(0)
    except FileNotFoundError as error:
        assert "Member not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_construction_does_not_open_zip() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    assert dataset._zip_cache._open_archives == {}


def test_integration_real_dataset_sample() -> None:
    if os.environ.get("SIH26_INTEGRATION") != "1":
        return

    full = ZipManifestDataset.from_huggingface()
    drone_index = next(
        index
        for index in range(len(full))
        if full.get_source(index).archive_name
        == "Drone-Noise-Audio-set.zip"
    )

    dataset = ZipManifestDataset(
        metadata_path=full.metadata_path,
        repo_id=full.repo_id,
        indices=[drone_index],
    )

    source = dataset.get_source(0)
    audio, sample_rate = dataset.load_audio(0)

    assert source.archive_name == "Drone-Noise-Audio-set.zip"
    assert sample_rate == 16_000
    assert audio.dtype == np.float32
    assert len(audio) > 0


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | ZipManifestDataset Tests")
    print("=" * 70)

    _build_fixtures()

    tests = [
        test_metadata_parsing_and_row_count,
        test_source_sample_preserves_metadata,
        test_dataset_index_without_audio_access,
        test_subset_indices,
        test_lazy_audio_load_clean,
        test_lazy_audio_load_noise_subset,
        test_missing_archive_raises,
        test_missing_member_raises,
        test_construction_does_not_open_zip,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    if os.environ.get("SIH26_INTEGRATION") == "1":
        test_integration_real_dataset_sample()
        print("PASS: test_integration_real_dataset_sample")
    else:
        print("SKIP: test_integration_real_dataset_sample (set SIH26_INTEGRATION=1)")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
