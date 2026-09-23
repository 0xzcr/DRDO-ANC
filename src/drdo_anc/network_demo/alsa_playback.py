"""ALSA ``aplay`` playback for the Pi network-audio receiver."""

from __future__ import annotations

import subprocess

import numpy as np

from .protocol import DEFAULT_SAMPLE_RATE


def is_alsa_hw_device(device: str) -> bool:
    text = device.strip().lower()
    return text.startswith("hw:") or text.startswith("plughw:")


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


class AlsaAplayOutput:
    """
    Stereo 48 kHz playback through ``aplay`` for ALSA ``hw:`` / ``plughw:``.

    Mono network audio is duplicated to both channels before conversion to
    little-endian S16 for the bcm2835 headphone DAC.
    """

    def __init__(
        self,
        device: str,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 2,
    ) -> None:
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError(
                f"Network receiver requires {DEFAULT_SAMPLE_RATE} Hz."
            )

        if channels != 2:
            raise ValueError("ALSA headphone output is stereo.")

        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._closed = False
        self._proc = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-D",
                device,
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                str(channels),
                "-r",
                str(sample_rate),
            ],
            stdin=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("aplay stdin was not opened.")

    def sample_rate(self) -> int:
        return self._sample_rate

    def write_stereo(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("ALSA output is closed.")

        frames = np.asarray(audio, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self._channels:
            raise ValueError(
                f"Expected stereo [T, {self._channels}], got {frames.shape}."
            )

        assert self._proc.stdin is not None
        self._proc.stdin.write(float_to_int16(frames).tobytes())
        self._proc.stdin.flush()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=2)
        except Exception:
            self._proc.kill()


def open_alsa_output(
    device: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AlsaAplayOutput:
    if not is_alsa_hw_device(device):
        raise ValueError(
            f"Expected ALSA hw:/plughw: device string, got {device!r}."
        )
    return AlsaAplayOutput(device, sample_rate=sample_rate)
