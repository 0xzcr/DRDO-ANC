import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from drdo_anc.audio.mixing import (
    align_noise_to_clean_length,
    calculate_snr,
    create_mixture,
)
from drdo_anc.benchmark import MixtureGenerator
from drdo_anc.benchmark.evaluation_manifest import (
    EvaluationManifest,
)
from drdo_anc.benchmark.selection import build_development_manifest
from drdo_anc.dataset import (
    ZipManifestDataset,
    derive_mixing_seed,
    english_speaker_id,
    is_clean_source,
    is_ms_snsd_clean_row,
    is_noise_source,
    load_metadata_rows,
    row_to_source_sample,
)
from drdo_anc.dataset.source_pool import (
    DEVELOPMENT_NOISE_CATEGORIES,
    DEVELOPMENT_SNR_LEVELS_DB,
    RULES_VERSION,
    SPLIT_NAME,
)

from build_evaluation_fixtures import (
    ARCHIVE_PATH,
    FIXTURE_DIR,
    METADATA_PATH,
    build_fixtures,
)


def setup_module() -> None:
    build_fixtures()


def test_clean_filter_rules() -> None:
    rows = load_metadata_rows(METADATA_PATH)

    clean_rows = [row for row in rows if is_clean_source(row)]
    assert len(clean_rows) == 12
    assert any(
        is_ms_snsd_clean_row(row)
        for row in clean_rows
    )


def test_noise_filter_excludes_ms_snsd_clean_and_triplets() -> None:
    rows = load_metadata_rows(METADATA_PATH)

    noise_rows = [row for row in rows if is_noise_source(row)]
    sources = {row["dataset_source"] for row in noise_rows}

    assert "MS-SNSD-Complex-Noise" in sources
    assert all(
        row["inferred_subclass"]
        not in {"clean_train", "clean_test", "Test_Triplets", "Training_Files"}
        for row in noise_rows
    )


def test_deterministic_manifest_generation() -> None:
    manifest_a = build_development_manifest(METADATA_PATH)
    manifest_b = build_development_manifest(METADATA_PATH)

    assert manifest_a.to_dict() == manifest_b.to_dict()
    assert len(manifest_a.cases) == 60


def test_manifest_json_round_trip() -> None:
    manifest = build_development_manifest(METADATA_PATH)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "manifest.json"
        manifest.save_json(path)
        loaded = EvaluationManifest.load_json(path)

    assert loaded.to_dict() == manifest.to_dict()


def test_manifest_generation_does_not_open_zip() -> None:
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )

    build_development_manifest(METADATA_PATH)

    assert dataset._zip_cache._open_archives == {}


def test_case_ids_and_mixing_seeds() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    case = manifest.cases[0]

    assert case.case_id == "benchmark_eval_sih26-eval-v1_00001"
    assert case.mixing_seed == derive_mixing_seed(case.case_id)


def test_manifest_distribution() -> None:
    manifest = build_development_manifest(METADATA_PATH)

    speakers = {
        english_speaker_id(case.clean_source)
        for case in manifest.cases
    }
    categories = {case.noise_category for case in manifest.cases}
    snrs = {case.snr_db for case in manifest.cases}
    case_ids = [case.case_id for case in manifest.cases]

    assert len(manifest.cases) == 60
    assert len(speakers) == 10
    assert categories == set(DEVELOPMENT_NOISE_CATEGORIES)
    assert snrs == set(DEVELOPMENT_SNR_LEVELS_DB)
    assert len(case_ids) == len(set(case_ids))


def test_no_demand_category_in_development_manifest() -> None:
    manifest = build_development_manifest(METADATA_PATH)

    assert all(
        case.noise_source.dataset_source != "DEMAND-Background-Noise"
        for case in manifest.cases
    )


def test_short_noise_cyclic_repetition() -> None:
    clean = np.ones(10, dtype=np.float32)
    noise = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    aligned = align_noise_to_clean_length(
        noise,
        len(clean),
        mixing_seed=7,
    )

    assert len(aligned) == 10
    assert not np.array_equal(aligned, np.tile(noise, 4)[:10])


def test_long_noise_deterministic_crop() -> None:
    noise = np.arange(20, dtype=np.float32)

    aligned_a = align_noise_to_clean_length(
        noise,
        8,
        mixing_seed=3,
    )
    aligned_b = align_noise_to_clean_length(
        noise,
        8,
        mixing_seed=3,
    )
    aligned_c = align_noise_to_clean_length(
        noise,
        8,
        mixing_seed=4,
    )

    assert np.array_equal(aligned_a, aligned_b)
    assert not np.array_equal(aligned_a, aligned_c)


def test_target_snr_accuracy() -> None:
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(16_000, dtype=np.float32)
    noise = rng.standard_normal(2_000, dtype=np.float32)

    for target in (0.0, 5.0):
        _noisy, scaled_noise, achieved = create_mixture(
            clean,
            noise,
            target,
            mixing_seed=42,
        )
        measured = calculate_snr(clean, scaled_noise)
        assert abs(measured - target) < 0.05


def test_mixture_generator_deterministic() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)

    first = generator.generate(manifest.cases[0])
    second = generator.generate(manifest.cases[0])

    assert np.array_equal(first.clean, second.clean)
    assert np.array_equal(first.noisy, second.noisy)
    assert first.sample_rate == 16_000


def test_different_snr_produces_different_waveform() -> None:
    manifest = build_development_manifest(METADATA_PATH)
    dataset = ZipManifestDataset(
        metadata_path=METADATA_PATH,
        repo_id=None,
        archive_dir=FIXTURE_DIR,
    )
    generator = MixtureGenerator(dataset)

    cases = [
        case
        for case in manifest.cases
        if case.clean_source.sample_id
        == manifest.cases[0].clean_source.sample_id
        and case.noise_category
        == manifest.cases[0].noise_category
    ]

    assert len(cases) == 2
    first = generator.generate(cases[0])
    second = generator.generate(cases[1])

    assert not np.array_equal(first.noisy, second.noisy)


def test_integration_real_metadata_manifest() -> None:
    if os.environ.get("SIH26_INTEGRATION") != "1":
        return

    import urllib.request

    url = (
        "https://huggingface.co/datasets/"
        "Panav-Payappagoudar/sih-26-processed-audio/"
        "resolve/main/metadata.csv"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        metadata_path = Path(tmp_dir) / "metadata.csv"
        with urllib.request.urlopen(url, timeout=180) as resp:
            metadata_path.write_bytes(resp.read())

        manifest = build_development_manifest(metadata_path)

    assert len(manifest.cases) == 60
    assert manifest.split_name == SPLIT_NAME
    assert manifest.rules_version == RULES_VERSION
    assert manifest.noise_categories == DEVELOPMENT_NOISE_CATEGORIES

    speakers = {
        english_speaker_id(case.clean_source)
        for case in manifest.cases
    }
    assert len(speakers) == 10

    category_counts = Counter(
        case.noise_category for case in manifest.cases
    )
    assert category_counts["uav_drone"] == 20
    assert category_counts["impulsive_firearms"] == 20
    assert category_counts["vehicle_engine"] == 20


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Evaluation Manifest + Mixture Tests")
    print("=" * 70)

    build_fixtures()

    tests = [
        test_clean_filter_rules,
        test_noise_filter_excludes_ms_snsd_clean_and_triplets,
        test_deterministic_manifest_generation,
        test_manifest_json_round_trip,
        test_manifest_generation_does_not_open_zip,
        test_case_ids_and_mixing_seeds,
        test_manifest_distribution,
        test_no_demand_category_in_development_manifest,
        test_short_noise_cyclic_repetition,
        test_long_noise_deterministic_crop,
        test_target_snr_accuracy,
        test_mixture_generator_deterministic,
        test_different_snr_produces_different_waveform,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    if os.environ.get("SIH26_INTEGRATION") == "1":
        test_integration_real_metadata_manifest()
        print("PASS: test_integration_real_metadata_manifest")
    else:
        print("SKIP: test_integration_real_metadata_manifest")

    manifest = build_development_manifest(METADATA_PATH)
    print("\nExample case:")
    print(json.dumps(manifest.cases[0].__dict__, default=str, indent=2)[:800])

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
