from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch

from drdo_anc.benchmark import (
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkRunner,
)
from drdo_anc.dataset import AudioSample, ListDataset
from drdo_anc.enhancement.base import Enhancer
from drdo_anc.enhancement import DeepFilterNetEnhancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "clean_freesound_33711.wav"
)

NOISY_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "clean_freesound_33711_noise_573577_snr0.wav"
)


def snr0_sample() -> AudioSample:
    return AudioSample(
        sample_id="snr0",
        clean_path=CLEAN_PATH,
        noisy_path=NOISY_PATH,
        sample_rate=48_000,
        snr_db=0.0,
        noise_type="573577",
        split="test",
    )


class MockEnhancer(Enhancer):
    """Identity enhancer for fast runner tests."""

    def __init__(
        self,
        sample_rate: int = 48_000,
    ) -> None:
        self._sample_rate = sample_rate
        self.process_calls = 0
        self.stream_calls = 0
        self.reset_calls = 0

    def load(self) -> None:
        return None

    def reset(self) -> None:
        self.reset_calls += 1
        self.process_calls = 0
        self.stream_calls = 0

    def sample_rate(self) -> int:
        return self._sample_rate

    def name(self) -> str:
        return "MockEnhancer"

    def _to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2:
            if audio.shape[0] != 1:
                raise ValueError("Expected mono audio.")
            audio = audio.squeeze(0)

        return audio.float()

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        self.process_calls += 1
        mono = self._to_mono(audio)
        return mono.unsqueeze(0)

    def process_stream(
        self,
        audio_chunk: torch.Tensor,
    ) -> torch.Tensor:
        self.stream_calls += 1
        return self._to_mono(audio_chunk)

    def flush(self) -> torch.Tensor:
        return torch.empty(
            0,
            dtype=torch.float32,
        )


def write_test_wav(
    path: Path,
    num_samples: int,
    sample_rate: int = 48_000,
) -> None:
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(
        num_samples,
        dtype=np.float32,
    )

    sf.write(
        path,
        audio,
        sample_rate,
    )


def test_mock_offline_preserves_length() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        noisy_path = tmp_path / "noisy.wav"
        write_test_wav(noisy_path, 12_345)

        sample = AudioSample(
            sample_id="mock_offline",
            noisy_path=noisy_path,
            sample_rate=48_000,
        )

        enhancer = MockEnhancer()
        enhancer.load()

        runner = BenchmarkRunner(
            enhancer,
            BenchmarkConfig(
                mode=BenchmarkMode.OFFLINE,
                save_enhanced=False,
                measure_timing=False,
            ),
        )

        result = runner.run_sample(sample)

        assert enhancer.process_calls == 1
        assert result.metrics == {}

        noisy_audio, _ = sf.read(
            noisy_path,
            dtype="float32",
        )

        assert len(noisy_audio) == 12_345


def test_mock_streaming_arbitrary_chunks_and_length() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        noisy_path = tmp_path / "noisy.wav"
        num_samples = 12_345
        write_test_wav(noisy_path, num_samples)

        sample = AudioSample(
            sample_id="mock_streaming",
            noisy_path=noisy_path,
            sample_rate=48_000,
        )

        enhancer = MockEnhancer()
        enhancer.load()

        runner = BenchmarkRunner(
            enhancer,
            BenchmarkConfig(
                mode=BenchmarkMode.STREAMING,
                save_enhanced=True,
                output_dir=tmp_path / "enhanced",
                overwrite=True,
                measure_timing=False,
            ),
        )

        result = runner.run_sample(sample)

        assert enhancer.stream_calls > 1
        assert result.enhanced_path is not None

        enhanced, _ = sf.read(
            result.enhanced_path,
            dtype="float32",
        )

        assert len(enhanced) == num_samples


def test_mock_reset_between_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        samples = []

        for index in range(2):
            noisy_path = tmp_path / f"noisy_{index}.wav"
            write_test_wav(noisy_path, 1_000 + index)

            samples.append(
                AudioSample(
                    sample_id=f"sample_{index}",
                    noisy_path=noisy_path,
                    sample_rate=48_000,
                )
            )

        enhancer = MockEnhancer()
        enhancer.load()

        runner = BenchmarkRunner(
            enhancer,
            BenchmarkConfig(
                mode=BenchmarkMode.STREAMING,
                save_enhanced=False,
                measure_timing=False,
            ),
        )

        runner.run(ListDataset(samples))

        assert enhancer.reset_calls == 2


def test_snr0_offline_df3_metrics() -> None:
    enhancer = DeepFilterNetEnhancer()
    enhancer.load()

    runner = BenchmarkRunner(
        enhancer,
        BenchmarkConfig(
            mode=BenchmarkMode.OFFLINE,
            delay_samples=0,
            save_enhanced=False,
            measure_timing=False,
        ),
    )

    result = runner.run_sample(snr0_sample())
    metrics = result.metrics

    assert abs(metrics["enhanced_snr"] - 9.817) < 0.05
    assert abs(metrics["enhanced_si_sdr"] - 9.644) < 0.05
    assert abs(metrics["enhanced_stoi"] - 0.9751) < 0.01


def test_snr0_streaming_df3_metrics_and_length() -> None:
    enhancer = DeepFilterNetEnhancer()
    enhancer.load()

    noisy, sample_rate = sf.read(
        NOISY_PATH,
        dtype="float32",
    )

    runner = BenchmarkRunner(
        enhancer,
        BenchmarkConfig(
            mode=BenchmarkMode.STREAMING,
            delay_samples=1440,
            save_enhanced=False,
            measure_timing=False,
        ),
    )

    result = runner.run_sample(snr0_sample())
    metrics = result.metrics

    assert abs(metrics["enhanced_snr"] - 9.740) < 0.05
    assert abs(metrics["enhanced_si_sdr"] - 9.567) < 0.05
    assert abs(metrics["enhanced_stoi"] - 0.9747) < 0.01

    enhanced = runner._enhance_streaming(noisy)
    assert len(enhanced) == len(noisy)
    assert sample_rate == enhancer.sample_rate()


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Benchmark Runner Tests")
    print("=" * 70)

    test_mock_offline_preserves_length()
    print("PASS: mock offline length")

    test_mock_streaming_arbitrary_chunks_and_length()
    print("PASS: mock streaming chunks and length")

    test_mock_reset_between_samples()
    print("PASS: mock reset between samples")

    print("\nRunning DF3 offline integration test...")
    test_snr0_offline_df3_metrics()
    print("PASS: snr0 offline DF3 metrics")

    print("\nRunning DF3 streaming integration test...")
    test_snr0_streaming_df3_metrics_and_length()
    print("PASS: snr0 streaming DF3 metrics and length")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
