"""Windows-side UDP ``AudioOutput`` for the network-audio demo."""

from __future__ import annotations

import socket

import numpy as np

from .interfaces import AudioOutput
from .network_protocol import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLES_PER_PACKET,
    encode_packet,
    parse_endpoint,
)


class NetworkAudioOutput(AudioOutput):
    """
    Packetize mono float32 audio into 10 ms UDP frames.

    Implements ``AudioOutput`` so ``StreamingPipeline`` can send enhanced
    audio to a Raspberry Pi receiver without changing pipeline behavior.
    Arbitrary ``write()`` sizes are buffered into 480-sample packets.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        samples_per_packet: int = DEFAULT_SAMPLES_PER_PACKET,
        sock: socket.socket | None = None,
    ) -> None:
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError(
                f"Network audio demo requires {DEFAULT_SAMPLE_RATE} Hz, "
                f"got {sample_rate}."
            )

        if samples_per_packet != DEFAULT_SAMPLES_PER_PACKET:
            raise ValueError(
                f"Network audio demo requires {DEFAULT_SAMPLES_PER_PACKET} "
                f"samples per packet, got {samples_per_packet}."
            )

        self._host, self._port = parse_endpoint(endpoint)
        self._sample_rate = sample_rate
        self._frame_length = samples_per_packet
        self._owns_socket = sock is None
        self._socket = sock if sock is not None else socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self._buffer = np.empty(0, dtype=np.float32)
        self._sequence = 0
        self._packets_sent = 0
        self._samples_sent = 0
        self._closed = False

    @property
    def endpoint(self) -> tuple[str, int]:
        return self._host, self._port

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    @property
    def samples_sent(self) -> int:
        return self._samples_sent

    @property
    def next_sequence(self) -> int:
        return self._sequence

    def sample_rate(self) -> int:
        return self._sample_rate

    def write(self, audio: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("NetworkAudioOutput is closed.")

        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return

        self._buffer = np.concatenate((self._buffer, chunk))
        self._flush_complete_frames(pad=False)

    def close(self) -> None:
        if self._closed:
            return

        try:
            self._flush_complete_frames(pad=True)
        finally:
            self._closed = True
            if self._owns_socket:
                self._socket.close()

    def _flush_complete_frames(self, *, pad: bool) -> None:
        while len(self._buffer) >= self._frame_length:
            frame = self._buffer[: self._frame_length]
            self._buffer = self._buffer[self._frame_length :]
            self._send_frame(frame)

        if pad and len(self._buffer) > 0:
            padded = np.zeros(self._frame_length, dtype=np.float32)
            padded[: len(self._buffer)] = self._buffer
            self._buffer = np.empty(0, dtype=np.float32)
            self._send_frame(padded)

    def _send_frame(self, frame: np.ndarray) -> None:
        datagram = encode_packet(
            frame,
            sequence=self._sequence,
            sample_rate=self._sample_rate,
        )
        self._socket.sendto(datagram, (self._host, self._port))
        self._sequence += 1
        self._packets_sent += 1
        self._samples_sent += int(frame.size)
