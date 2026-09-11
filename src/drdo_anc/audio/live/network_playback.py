"""Playback helpers for network-audio (Windows sounddevice + shared ALSA)."""

from __future__ import annotations

from drdo_anc.network_demo.alsa_playback import (
    AlsaAplayOutput,
    float_to_int16,
    is_alsa_hw_device,
    open_alsa_output,
)
from drdo_anc.network_demo.protocol import DEFAULT_SAMPLE_RATE
from .sounddevice_backend import open_sounddevice_output


def parse_output_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def open_receiver_output(
    sample_rate: int,
    output_device: str | int | None,
):
    """
    Open playback for development machines.

    ``hw:`` / ``plughw:`` strings use ALSA ``aplay`` (stereo S16).
    Anything else uses existing sounddevice playback (mono in, host upmix).

    Raspberry Pi production use should import ``drdo_anc.network_demo`` only
    (see ``scripts/run_network_audio_receiver.py``).
    """

    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"Network receiver requires {DEFAULT_SAMPLE_RATE} Hz."
        )

    if isinstance(output_device, str) and is_alsa_hw_device(output_device):
        return open_alsa_output(output_device, sample_rate=sample_rate)

    parsed = (
        parse_output_device(output_device)
        if isinstance(output_device, str)
        else output_device
    )
    return open_sounddevice_output(
        sample_rate,
        output_device=parsed,
        blocksize=480,
    )
