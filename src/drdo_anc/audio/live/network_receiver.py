"""Raspberry Pi UDP receiver: jitter buffer + playback helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network_protocol import (
    DEFAULT_SAMPLES_PER_PACKET,
    PacketError,
    decode_packet,
    mono_to_stereo,
)


DEFAULT_JITTER_CAPACITY = 8
DEFAULT_JITTER_PREFILL = 4


@dataclass
class ReceiverStats:
    packets_received: int = 0
    packets_lost: int = 0
    sequence_gaps: int = 0
    packets_malformed: int = 0
    packets_dropped_late: int = 0
    packets_dropped_overflow: int = 0
    underruns: int = 0
    output_samples: int = 0

    def as_dict(self, *, buffer_level: int, buffer_capacity: int) -> dict[str, int]:
        return {
            "packets_received": self.packets_received,
            "packets_lost": self.packets_lost,
            "sequence_gaps": self.sequence_gaps,
            "packets_malformed": self.packets_malformed,
            "packets_dropped_late": self.packets_dropped_late,
            "packets_dropped_overflow": self.packets_dropped_overflow,
            "buffer_level": buffer_level,
            "buffer_capacity": buffer_capacity,
            "underruns": self.underruns,
            "output_samples": self.output_samples,
        }

    def format_line(self, *, buffer_level: int, buffer_capacity: int) -> str:
        return (
            f"packets_rx={self.packets_received} "
            f"lost={self.packets_lost} "
            f"gaps={self.sequence_gaps} "
            f"malformed={self.packets_malformed} "
            f"buffer={buffer_level}/{buffer_capacity} "
            f"underruns={self.underruns} "
            f"out_samples={self.output_samples}"
        )


class JitterBuffer:
    """
    Small bounded packet buffer keyed by sequence number.

    Playback is deterministic: after prefill, each ``pop()`` consumes the
    next sequence. Missing packets become silence. Late packets are dropped.
    This is a demo jitter buffer, not a production PLC implementation.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_JITTER_CAPACITY,
        prefill: int = DEFAULT_JITTER_PREFILL,
        frame_samples: int = DEFAULT_SAMPLES_PER_PACKET,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        if prefill <= 0:
            raise ValueError("prefill must be positive.")

        if prefill > capacity:
            raise ValueError("prefill cannot exceed capacity.")

        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive.")

        self.capacity = capacity
        self.prefill = prefill
        self.frame_samples = frame_samples
        self.stats = ReceiverStats()
        self._packets: dict[int, np.ndarray] = {}
        self._next_seq: int | None = None
        self._started = False

    @property
    def level(self) -> int:
        return len(self._packets)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def next_sequence(self) -> int | None:
        return self._next_seq

    def push(self, sequence: int, samples: np.ndarray) -> None:
        frame = np.asarray(samples, dtype=np.float32).reshape(-1)
        if frame.size != self.frame_samples:
            raise ValueError(
                f"Expected {self.frame_samples} samples, got {frame.size}."
            )

        if self._next_seq is not None and sequence < self._next_seq:
            self.stats.packets_dropped_late += 1
            return

        if self._next_seq is not None and sequence >= self._next_seq + self.capacity:
            self.stats.packets_dropped_overflow += 1
            return

        if sequence in self._packets:
            return

        if len(self._packets) >= self.capacity:
            self.stats.packets_dropped_overflow += 1
            return

        self._packets[sequence] = frame.copy()

        if not self._started and len(self._packets) >= self.prefill:
            self._started = True
            self._next_seq = min(self._packets)

    def pop(self) -> np.ndarray | None:
        """
        Return the next mono frame, silence, or ``None`` during prefill.

        After playback starts, this always returns ``frame_samples`` samples.
        """

        if not self._started or self._next_seq is None:
            return None

        expected = self._next_seq
        frame = self._packets.pop(expected, None)

        if frame is None:
            has_later = any(seq > expected for seq in self._packets)
            silence = np.zeros(self.frame_samples, dtype=np.float32)
            if has_later:
                self.stats.packets_lost += 1
                self.stats.sequence_gaps += 1
            else:
                self.stats.underruns += 1
            self._next_seq = expected + 1
            self.stats.output_samples += self.frame_samples
            return silence

        self._next_seq = expected + 1
        self.stats.output_samples += int(frame.size)
        return frame


class NetworkAudioReceiver:
    """Decode datagrams into a jitter buffer and emit playback frames."""

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_JITTER_CAPACITY,
        prefill: int = DEFAULT_JITTER_PREFILL,
        frame_samples: int = DEFAULT_SAMPLES_PER_PACKET,
    ) -> None:
        self.buffer = JitterBuffer(
            capacity=capacity,
            prefill=prefill,
            frame_samples=frame_samples,
        )

    @property
    def stats(self) -> ReceiverStats:
        return self.buffer.stats

    def handle_datagram(self, datagram: bytes) -> None:
        try:
            packet = decode_packet(datagram)
        except PacketError:
            self.stats.packets_malformed += 1
            return

        self.stats.packets_received += 1
        self.buffer.push(packet.sequence, packet.samples)

    def pull_mono_frame(self) -> np.ndarray | None:
        return self.buffer.pop()

    def pull_stereo_frame(self) -> np.ndarray | None:
        mono = self.pull_mono_frame()
        if mono is None:
            return None
        return mono_to_stereo(mono)
