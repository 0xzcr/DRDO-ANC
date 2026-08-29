from __future__ import annotations

from typing import Any

import numpy as np

from .interfaces import AudioInput, AudioOutput


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for desktop live audio. "
            "Install it with: pip install sounddevice"
        ) from exc

    return sd


def list_audio_devices() -> list[dict[str, Any]]:
    """
    Return host audio devices reported by PortAudio/sounddevice.

    Each entry contains ``index``, ``name``, ``max_input_channels``,
    ``max_output_channels``, and ``default_sample_rate``.
    """

    sd = _import_sounddevice()
    devices: list[dict[str, Any]] = []

    for index, info in enumerate(sd.query_devices()):
        devices.append(
            {
                "index": index,
                "name": info["name"],
                "max_input_channels": info["max_input_channels"],
                "max_output_channels": info["max_output_channels"],
                "default_sample_rate": info["default_samplerate"],
            }
        )

    return devices


def format_device_listing() -> str:
    """Return a human-readable device listing for CLI output."""

    lines = ["Available audio devices:"]

    for device in list_audio_devices():
        lines.append(
            f"  [{device['index']}] {device['name']} "
            f"(in={device['max_input_channels']}, "
            f"out={device['max_output_channels']}, "
            f"default_sr={device['default_sample_rate']:.0f} Hz)"
        )

    return "\n".join(lines)


class SoundDeviceAudioInput(AudioInput):
    """
    Desktop microphone capture via sounddevice/PortAudio.

    Audio is captured as mono float32 at the configured sample rate.
    ``read()`` blocks until up to ``max_samples`` frames are available.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        device: int | str | None = None,
        blocksize: int = 0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        sd = _import_sounddevice()

        self._sample_rate = sample_rate
        self._device = device
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            device=device,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
        )
        self._stream.start()

    def sample_rate(self) -> int:
        return self._sample_rate

    def read(self, max_samples: int) -> np.ndarray:
        if max_samples <= 0:
            return np.empty(0, dtype=np.float32)

        if self._stream is None:
            raise RuntimeError("AudioInput is closed.")

        frames, _overflowed = self._stream.read(max_samples)

        return (
            np.asarray(frames, dtype=np.float32)
            .reshape(-1)
        )

    def close(self) -> None:
        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None


class SoundDeviceAudioOutput(AudioOutput):
    """
    Desktop speaker playback via sounddevice/PortAudio.

    Audio is played as mono float32 at the configured sample rate.
    ``write()`` blocks until the host accepts the provided frames.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        device: int | str | None = None,
        blocksize: int = 0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        sd = _import_sounddevice()

        self._sample_rate = sample_rate
        self._device = device
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            device=device,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
        )
        self._stream.start()

    def sample_rate(self) -> int:
        return self._sample_rate

    def write(self, audio: np.ndarray) -> None:
        if self._stream is None:
            raise RuntimeError("AudioOutput is closed.")

        array = np.asarray(audio, dtype=np.float32).reshape(-1)

        if array.size == 0:
            return

        self._stream.write(array.reshape(-1, 1))

    def close(self) -> None:
        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None
