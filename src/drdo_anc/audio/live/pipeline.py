import time

import numpy as np
import torch

from drdo_anc.enhancement.base import Enhancer

from .interfaces import AudioInput, AudioOutput


class StreamingPipeline:
    """
    Synchronous microphone-to-speaker streaming through an ``Enhancer``.

    The pipeline repeatedly:

    1. Reads an arbitrary-sized chunk from ``AudioInput``
    2. Passes it to ``Enhancer.process_stream()`` (or pass-through)
    3. Writes any produced output to ``AudioOutput``

    Hardware chunk sizes are unrelated to model frame sizes. Frame
    assembly remains inside the enhancer via ``StreamingBuffer``.

    Shutdown semantics
    ------------------
    * ``run()`` resets the enhancer once at stream start.
    * On normal end-of-input, ``KeyboardInterrupt``, or ``request_stop()``,
      ``run()`` calls ``enhancer.flush()`` **exactly once** before closing
      I/O devices.
    * Pass-through mode (``enhancer=None``) skips enhancement and flush.
    """

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        enhancer: Enhancer | None = None,
        *,
        read_chunk_size: int = 1024,
    ) -> None:
        if read_chunk_size <= 0:
            raise ValueError("read_chunk_size must be positive.")

        input_rate = audio_input.sample_rate()
        output_rate = audio_output.sample_rate()

        if input_rate != output_rate:
            raise ValueError(
                f"AudioInput sample rate ({input_rate} Hz) does not "
                f"match AudioOutput sample rate ({output_rate} Hz)."
            )

        if enhancer is not None:
            enhancer_rate = enhancer.sample_rate()

            if input_rate != enhancer_rate:
                raise ValueError(
                    f"AudioInput sample rate ({input_rate} Hz) does not "
                    f"match enhancer sample rate ({enhancer_rate} Hz)."
                )

        self._audio_input = audio_input
        self._audio_output = audio_output
        self._enhancer = enhancer
        self._read_chunk_size = read_chunk_size
        self._stop_requested = False
        self._flushed = False
        self._shutdown_complete = False

    @property
    def sample_rate(self) -> int:
        return self._audio_input.sample_rate()

    @property
    def read_chunk_size(self) -> int:
        return self._read_chunk_size

    def request_stop(self) -> None:
        """Request graceful shutdown after the current read cycle."""

        self._stop_requested = True

    def run(
        self,
        *,
        diagnose: bool = False,
        diagnose_interval_s: float = 1.0,
        max_chunks: int | None = None,
    ) -> None:
        """
        Process audio until input is exhausted or shutdown is requested.

        ``KeyboardInterrupt`` triggers the same graceful shutdown path.

        When ``diagnose=True``, print ``SoundDeviceStreamStats`` (if
        available on the input device) every ``diagnose_interval_s``.

        When ``max_chunks`` is set, stop after that many successful reads.
        """

        if self._enhancer is not None:
            self._enhancer.reset()

        last_report = time.perf_counter()
        chunks_processed = 0

        try:
            while not self._stop_requested:
                chunk = self._audio_input.read(
                    self._read_chunk_size,
                )

                if len(chunk) == 0:
                    break

                self._process_chunk(chunk)
                chunks_processed += 1

                if (
                    max_chunks is not None
                    and chunks_processed >= max_chunks
                ):
                    break

                if diagnose:
                    now = time.perf_counter()

                    if now - last_report >= diagnose_interval_s:
                        self._print_diagnostics()
                        last_report = now
        except KeyboardInterrupt:
            pass
        finally:
            if diagnose:
                self._print_diagnostics()
            self._shutdown()

    def _print_diagnostics(self) -> None:
        stats = getattr(self._audio_input, "stats", None)

        if stats is None:
            return

        import json

        payload = stats.as_dict()
        host_input = getattr(
            self._audio_input,
            "host_input_channels",
            None,
        )
        host_output = getattr(
            self._audio_output,
            "host_output_channels",
            None,
        )

        if host_input is not None:
            payload["host_input_channels"] = host_input

        if host_output is not None:
            payload["host_output_channels"] = host_output

        print(json.dumps(payload, indent=2), flush=True)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        if self._enhancer is None:
            self._audio_output.write(chunk)
            return

        output_tensor = self._enhancer.process_stream(
            torch.from_numpy(chunk).float(),
        )

        self._write_tensor(output_tensor)

    def _write_tensor(self, audio: torch.Tensor) -> None:
        array = _tensor_to_mono_numpy(audio)

        if len(array) > 0:
            self._audio_output.write(array)

    def _shutdown(self) -> None:
        if self._shutdown_complete:
            return

        try:
            if self._enhancer is not None and not self._flushed:
                flush_tensor = self._enhancer.flush()
                self._write_tensor(flush_tensor)
                self._flushed = True
        finally:
            self._audio_input.close()
            self._audio_output.close()
            self._shutdown_complete = True


def _tensor_to_mono_numpy(audio: torch.Tensor) -> np.ndarray:
    array = (
        audio.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError("Expected mono audio tensor.")

        array = array.squeeze(0)

    return array
