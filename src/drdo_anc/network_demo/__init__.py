"""
Minimal UDP network-audio demo for Raspberry Pi playback.

This package is intentionally independent of ``drdo_anc.audio`` so the Pi
receiver does not import soundfile, sounddevice, PyTorch, or DeepFilterNet.

AI inference runs on the Windows host only.
"""

from .alsa_playback import AlsaAplayOutput, is_alsa_hw_device, open_alsa_output
from .protocol import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLES_PER_PACKET,
    AudioPacket,
    PacketError,
    decode_packet,
    encode_packet,
    mono_to_stereo,
    parse_endpoint,
)
from .receiver import (
    DEFAULT_JITTER_CAPACITY,
    DEFAULT_JITTER_PREFILL,
    JitterBuffer,
    NetworkAudioReceiver,
    ReceiverStats,
)

__all__ = [
    "AlsaAplayOutput",
    "AudioPacket",
    "DEFAULT_CHANNELS",
    "DEFAULT_JITTER_CAPACITY",
    "DEFAULT_JITTER_PREFILL",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLES_PER_PACKET",
    "JitterBuffer",
    "NetworkAudioReceiver",
    "PacketError",
    "ReceiverStats",
    "decode_packet",
    "encode_packet",
    "is_alsa_hw_device",
    "mono_to_stereo",
    "open_alsa_output",
    "parse_endpoint",
]
