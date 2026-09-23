"""Demo UDP audio packet format (not a production protocol)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"DFAN"
VERSION = 1
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLES_PER_PACKET = 480
HEADER_STRUCT = struct.Struct("<4sB3xIIHHI")
HEADER_SIZE = HEADER_STRUCT.size

# < magic(4) version(1) pad(3) seq(u32) sample_rate(u32)
#   channels(u16) samples_per_packet(u16) payload_length(u32)


class PacketError(ValueError):
    """Raised when a datagram is not a valid demo audio packet."""


@dataclass(frozen=True)
class AudioPacket:
    version: int
    sequence: int
    sample_rate: int
    channels: int
    samples_per_packet: int
    samples: np.ndarray

    @property
    def payload_length(self) -> int:
        return int(self.samples.nbytes)


def parse_endpoint(value: str) -> tuple[str, int]:
    """Parse ``host:port`` into ``(host, port)``."""

    text = value.strip()
    if not text:
        raise ValueError("Endpoint must be host:port.")

    host, sep, port_text = text.rpartition(":")
    if not sep or not host or not port_text:
        raise ValueError(f"Invalid endpoint {value!r}. Expected host:port.")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid endpoint port in {value!r}.") from exc

    if not (0 < port < 65536):
        raise ValueError(f"Port must be in 1..65535, got {port}.")

    return host, port


def encode_packet(
    samples: np.ndarray,
    *,
    sequence: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    version: int = VERSION,
) -> bytes:
    """Encode one mono float32 frame as a UDP datagram."""

    if sequence < 0:
        raise ValueError("sequence must be non-negative.")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    if channels != DEFAULT_CHANNELS:
        raise ValueError("Demo packets are mono only.")

    audio = np.ascontiguousarray(samples, dtype="<f4").reshape(-1)

    if audio.size != DEFAULT_SAMPLES_PER_PACKET:
        raise ValueError(
            f"Expected {DEFAULT_SAMPLES_PER_PACKET} samples, got {audio.size}."
        )

    payload = audio.tobytes()
    header = HEADER_STRUCT.pack(
        MAGIC,
        version,
        sequence,
        sample_rate,
        channels,
        DEFAULT_SAMPLES_PER_PACKET,
        len(payload),
    )
    return header + payload


def decode_packet(datagram: bytes) -> AudioPacket:
    """Decode and validate a UDP datagram."""

    if not isinstance(datagram, (bytes, bytearray)):
        raise PacketError("Datagram must be bytes.")

    if len(datagram) < HEADER_SIZE:
        raise PacketError(
            f"Datagram shorter than header ({len(datagram)} < {HEADER_SIZE})."
        )

    magic, version, sequence, sample_rate, channels, n_samples, payload_length = (
        HEADER_STRUCT.unpack_from(datagram)
    )

    if magic != MAGIC:
        raise PacketError(f"Invalid magic {magic!r}.")

    if version != VERSION:
        raise PacketError(f"Unsupported version {version}.")

    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise PacketError(f"Unsupported sample rate {sample_rate}.")

    if channels != DEFAULT_CHANNELS:
        raise PacketError(f"Unsupported channel count {channels}.")

    if n_samples != DEFAULT_SAMPLES_PER_PACKET:
        raise PacketError(f"Unsupported samples_per_packet {n_samples}.")

    expected_payload = DEFAULT_SAMPLES_PER_PACKET * 4
    if payload_length != expected_payload:
        raise PacketError(
            f"Unexpected payload_length {payload_length}, "
            f"expected {expected_payload}."
        )

    payload = bytes(datagram[HEADER_SIZE:])
    if len(payload) != payload_length:
        raise PacketError(
            f"Payload size {len(payload)} does not match "
            f"payload_length {payload_length}."
        )

    samples = np.frombuffer(payload, dtype="<f4").copy()
    if samples.size != DEFAULT_SAMPLES_PER_PACKET:
        raise PacketError("Decoded sample count mismatch.")

    if not np.isfinite(samples).all():
        raise PacketError("Payload contains NaN or Inf.")

    return AudioPacket(
        version=int(version),
        sequence=int(sequence),
        sample_rate=int(sample_rate),
        channels=int(channels),
        samples_per_packet=int(n_samples),
        samples=samples,
    )


def mono_to_stereo(samples: np.ndarray) -> np.ndarray:
    """Duplicate mono ``[T]`` to interleaved stereo ``[T, 2]``."""

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    return np.column_stack((mono, mono))
