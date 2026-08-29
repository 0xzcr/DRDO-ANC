import argparse
import os
import tempfile
import urllib.request
from pathlib import Path

from drdo_anc.benchmark import (
    BenchmarkMode,
    EvaluationManifest,
    MixtureGenerator,
    build_development_manifest,
)
from drdo_anc.benchmark.manifest_benchmark import (
    ManifestBenchmarkRunner,
    build_manifest_dataset,
    select_smoke_cases,
    validate_development_manifest,
)
from drdo_anc.dataset.manifest import (
    SIH26_METADATA_FILENAME,
    SIH26_REPO_ID,
)
from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark_results"


def _resolve_metadata_path(
    metadata_path: Path | None,
) -> Path:
    if metadata_path is not None:
        return metadata_path.resolve()

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=SIH26_REPO_ID,
            repo_type="dataset",
            filename=SIH26_METADATA_FILENAME,
        )

        return Path(downloaded)
    except Exception:
        url = (
            "https://huggingface.co/datasets/"
            f"{SIH26_REPO_ID}/resolve/main/"
            f"{SIH26_METADATA_FILENAME}"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / SIH26_METADATA_FILENAME
            with urllib.request.urlopen(url, timeout=180) as resp:
                target.write_bytes(resp.read())

            persistent = (
                PROJECT_ROOT
                / "data"
                / "cache"
                / SIH26_METADATA_FILENAME
            )
            persistent.parent.mkdir(parents=True, exist_ok=True)
            persistent.write_bytes(target.read_bytes())

            return persistent


def _parse_modes(raw: str) -> tuple[BenchmarkMode, ...]:
    if raw == "both":
        return (
            BenchmarkMode.OFFLINE,
            BenchmarkMode.STREAMING,
        )

    return (BenchmarkMode(raw),)


def _print_summary(report) -> None:
    successful = report.successful_results()
    failed = len(report.case_results) - len(successful)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(report.case_results)}")
    print(f"Failed:     {failed}/{len(report.case_results)}")

    overall = report.summary_overall()
    if overall:
        print("\nOverall:")
        for key, value in overall.items():
            print(f"  {key}: {value:.4f}")

    rtf_summary = report.summary_rtf()
    if rtf_summary:
        print("\nRTF:")
        for key, value in rtf_summary.items():
            print(f"  {key}: {value:.4f}")

    by_snr = report.summary_by_snr()
    if by_snr:
        print("\nBy SNR:")
        for snr_key, metrics in by_snr.items():
            print(f"  {snr_key}:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")

    by_category = report.summary_by_noise_category()
    if by_category:
        print("\nBy noise category:")
        for category, metrics in by_category.items():
            print(f"  {category}:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")

    failures = [
        result
        for result in report.case_results
        if result.status != "success"
    ]

    if failures:
        print("\nFailures:")
        for result in failures:
            print(
                f"  {result.case_id} [{result.mode}]: "
                f"{result.error}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SIH-26 development manifest benchmark "
            "with DeepFilterNet3."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help=(
            "Local metadata.csv path. When omitted, metadata is "
            "resolved from Hugging Face."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional pre-built manifest JSON.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Optional local directory containing dataset ZIP archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON/CSV benchmark outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "streaming", "both"],
        default="both",
        help="Enhancement mode to evaluate.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the 2-case smoke subset.",
    )
    parser.add_argument(
        "--streaming-delay-override",
        type=int,
        default=None,
        help=(
            "Override streaming delay compensation. Intended for "
            "validation experiments only."
        ),
    )
    parser.add_argument(
        "--skip-smoke-validation",
        action="store_true",
        help=(
            "Skip the mandatory 2-case smoke validation before the "
            "full 60-case run."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DRDO-ANC | DF3 Manifest Benchmark")
    print("=" * 70)

    metadata_path = _resolve_metadata_path(args.metadata_path)

    if args.manifest_path is not None:
        manifest = EvaluationManifest.load_json(
            args.manifest_path,
        )
    else:
        manifest = build_development_manifest(metadata_path)

    validate_development_manifest(manifest)

    dataset = build_manifest_dataset(
        manifest,
        metadata_path,
        archive_dir=args.archive_dir,
    )
    mixture_generator = MixtureGenerator(dataset)

    enhancer = DeepFilterNetEnhancer()
    print("\nLoading DeepFilterNet3...")
    enhancer.load()

    # Resampling to 48 kHz happens in ManifestBenchmarkRunner after mixture
    # generation. ZipManifestDataset and MixtureGenerator remain at 16 kHz.
    print(
        "\nResampling boundary: MixtureGenerator (16 kHz) -> "
        f"ManifestBenchmarkRunner.resample_mixture_for_enhancer() -> "
        f"DF3 input ({enhancer.sample_rate()} Hz)"
    )
    print(
        "Timing includes only model inference "
        "(offline process or streaming process_stream + flush)."
    )

    modes = _parse_modes(args.mode)
    runner = ManifestBenchmarkRunner(
        enhancer,
        mixture_generator,
        modes=modes,
    )

    if args.smoke:
        cases = select_smoke_cases(manifest)
        label = "smoke"
    else:
        if not args.skip_smoke_validation:
            print("\nRunning mandatory 2-case smoke validation...")
            smoke_cases = select_smoke_cases(manifest)
            smoke_report = runner.run(
                manifest,
                smoke_cases,
                streaming_delay_override=(
                    args.streaming_delay_override
                ),
            )

            smoke_success = len(smoke_report.successful_results())
            smoke_total = len(smoke_report.case_results)

            print(
                f"Smoke validation: {smoke_success}/{smoke_total} "
                "successful."
            )

            if smoke_success != smoke_total:
                _print_summary(smoke_report)
                raise SystemExit(
                    "Smoke validation failed. Aborting full benchmark."
                )

            if (
                BenchmarkMode.STREAMING in modes
                and args.streaming_delay_override is None
            ):
                bad_delay_report = runner.run(
                    manifest,
                    smoke_cases,
                    streaming_delay_override=0,
                )
                streaming_results = [
                    result
                    for result in bad_delay_report.case_results
                    if result.mode == BenchmarkMode.STREAMING.value
                    and result.status == "success"
                ]

                if streaming_results:
                    worst_si_sdr = min(
                        result.si_sdr
                        for result in streaming_results
                        if result.si_sdr is not None
                    )
                    print(
                        "\nStreaming delay=0 validation "
                        f"(expected degraded metrics): "
                        f"min SI-SDR={worst_si_sdr:.3f} dB"
                    )

                    if worst_si_sdr > 5.0:
                        raise SystemExit(
                            "Streaming delay=0 did not produce "
                            "degraded alignment metrics."
                        )

        cases = manifest.cases
        label = "full"

    print(
        f"\nEvaluating {len(cases)} case(s) "
        f"across {len(modes)} mode(s)..."
    )

    report = runner.run(
        manifest,
        cases,
        streaming_delay_override=args.streaming_delay_override,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = (
        args.output_dir
        / f"df3_manifest_benchmark_{label}.json"
    )
    csv_path = (
        args.output_dir
        / f"df3_manifest_benchmark_{label}.csv"
    )

    report.save_json(json_path)
    report.save_csv(csv_path)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")

    _print_summary(report)


if __name__ == "__main__":
    main()
