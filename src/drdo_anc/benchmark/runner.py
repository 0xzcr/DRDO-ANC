import time
from pathlib import Path

import numpy as np
import torch

from drdo_anc.audio.io import load_mono_wav, save_mono_wav
from drdo_anc.dataset.protocol import Dataset
from drdo_anc.dataset.sample import AudioSample
from drdo_anc.enhancement.base import Enhancer
from drdo_anc.evaluation import evaluate_model

from .config import BenchmarkConfig, BenchmarkMode
from .result import BenchmarkResult, SampleResult


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


class BenchmarkRunner:
    """Run enhancement and objective evaluation over a dataset."""

    def __init__(
        self,
        enhancer: Enhancer,
        config: BenchmarkConfig,
    ) -> None:
        self._enhancer = enhancer
        self._config = config

    def run_sample(
        self,
        sample: AudioSample,
    ) -> SampleResult:
        if sample.noisy_path is None:
            raise ValueError(
                f"Sample {sample.sample_id!r} has no noisy_path."
            )

        noisy_path = sample.noisy_path

        if not noisy_path.exists():
            raise FileNotFoundError(
                f"Noisy file not found: {noisy_path}"
            )

        self._enhancer.reset()

        noisy, sample_rate = load_mono_wav(noisy_path)
        expected_sample_rate = self._enhancer.sample_rate()

        if sample_rate != expected_sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {sample.sample_id}: "
                f"audio={sample_rate}, "
                f"enhancer={expected_sample_rate}"
            )

        inference_s: float | None = None

        if self._config.measure_timing:
            start = time.perf_counter()

        if self._config.mode == BenchmarkMode.OFFLINE:
            enhanced = self._enhance_offline(noisy)
        elif self._config.mode == BenchmarkMode.STREAMING:
            enhanced = self._enhance_streaming(noisy)
        else:
            raise ValueError(
                f"Unsupported benchmark mode: {self._config.mode}"
            )

        if self._config.measure_timing:
            inference_s = time.perf_counter() - start

        duration_s = len(noisy) / sample_rate
        rtf = (
            duration_s / inference_s
            if inference_s and inference_s > 0
            else None
        )

        enhanced_path = self._maybe_save_enhanced(
            sample,
            enhanced,
            sample_rate,
        )

        metrics = self._evaluate_sample(
            sample,
            noisy,
            enhanced,
            sample_rate,
        )

        return SampleResult(
            sample_id=sample.sample_id,
            metrics=metrics,
            inference_s=inference_s,
            rtf=rtf,
            enhanced_path=enhanced_path,
            split=sample.split,
            snr_db=sample.snr_db,
            noise_type=sample.noise_type,
        )

    def run(
        self,
        dataset: Dataset,
    ) -> BenchmarkResult:
        sample_results = [
            self.run_sample(sample)
            for sample in dataset
        ]

        return BenchmarkResult(
            model_name=self._enhancer.name(),
            mode=self._config.mode,
            delay_samples=self._config.delay_samples,
            sample_results=sample_results,
        )

    def _enhance_offline(
        self,
        noisy: np.ndarray,
    ) -> np.ndarray:
        audio = torch.from_numpy(noisy).float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        enhanced_tensor = self._enhancer.process(audio)

        return self._tensor_to_mono_numpy(
            enhanced_tensor,
        )

    def _enhance_streaming(
        self,
        noisy: np.ndarray,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        position = 0
        chunk_index = 0

        while position < len(noisy):
            chunk_size = STREAMING_CHUNK_SIZES[
                chunk_index % len(STREAMING_CHUNK_SIZES)
            ]

            end = min(
                position + chunk_size,
                len(noisy),
            )

            chunk = noisy[position:end]

            output_tensor = self._enhancer.process_stream(
                torch.from_numpy(chunk).float()
            )

            output = self._tensor_to_mono_numpy(
                output_tensor,
            )

            if len(output) > 0:
                outputs.append(output)

            position = end
            chunk_index += 1

        flush_tensor = self._enhancer.flush()
        flush_output = self._tensor_to_mono_numpy(
            flush_tensor,
        )

        if len(flush_output) > 0:
            outputs.append(flush_output)

        if outputs:
            enhanced = np.concatenate(outputs)
        else:
            enhanced = np.empty(
                0,
                dtype=np.float32,
            )

        if len(enhanced) != len(noisy):
            raise RuntimeError(
                "Streaming output length mismatch: "
                f"input={len(noisy)}, "
                f"output={len(enhanced)}"
            )

        return enhanced.astype(
            np.float32,
            copy=False,
        )

    def _maybe_save_enhanced(
        self,
        sample: AudioSample,
        enhanced: np.ndarray,
        sample_rate: int,
    ) -> Path | None:
        if not self._config.save_enhanced:
            return None

        if self._config.output_dir is None:
            return None

        enhanced_path = (
            self._config.output_dir
            / f"{sample.sample_id}.wav"
        )

        if (
            enhanced_path.exists()
            and not self._config.overwrite
        ):
            raise FileExistsError(
                f"Enhanced output already exists: {enhanced_path}"
            )

        save_mono_wav(
            enhanced_path,
            enhanced,
            sample_rate,
        )

        return enhanced_path

    def _evaluate_sample(
        self,
        sample: AudioSample,
        noisy: np.ndarray,
        enhanced: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        if sample.clean_path is None:
            return {}

        if not sample.clean_path.exists():
            raise FileNotFoundError(
                f"Clean file not found: {sample.clean_path}"
            )

        clean, clean_sr = load_mono_wav(
            sample.clean_path,
        )

        if clean_sr != sample_rate:
            raise ValueError(
                f"Clean sample rate mismatch for "
                f"{sample.sample_id}: "
                f"clean={clean_sr}, noisy={sample_rate}"
            )

        return evaluate_model(
            clean,
            noisy,
            enhanced,
            sample_rate,
            delay_samples=self._config.delay_samples,
        )

    @staticmethod
    def _tensor_to_mono_numpy(
        audio: torch.Tensor,
    ) -> np.ndarray:
        array = (
            audio.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        if array.ndim == 2:
            if array.shape[0] != 1:
                raise ValueError(
                    "Expected mono audio tensor."
                )

            array = array.squeeze(0)

        return array
