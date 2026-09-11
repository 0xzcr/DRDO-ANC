"""Unit tests for the demo UDP audio transport (no hardware, no DF3)."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from drdo_anc.audio.live import (
    FakeAudioInput,
    NetworkAudioOutput,
    StreamingPipeline,
)
from drdo_anc.network_demo.alsa_playback import float_to_int16
from drdo_anc.audio.live.network_protocol import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLES_PER_PACKET,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    VERSION,
    PacketError,
    decode_packet,
    encode_packet,
    mono_to_stereo,
    parse_endpoint,
)
from drdo_anc.audio.live.network_receiver import JitterBuffer, NetworkAudioReceiver


class FakeUdpSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, data: bytes, address: tuple[str, int]) -> int:
        self.sent.append((bytes(data), address))
        return len(data)

    def close(self) -> None:
        self.closed = True


def _frame(value: float, sequence_tag: float | None = None) -> np.ndarray:
    audio = np.full(DEFAULT_SAMPLES_PER_PACKET, value, dtype=np.float32)
    if sequence_tag is not None:
        audio[0] = np.float32(sequence_tag)
    return audio


def test_parse_endpoint() -> None:
    assert parse_endpoint("10.144.163.155:5000") == ("10.144.163.155", 5000)
    assert parse_endpoint("0.0.0.0:5000") == ("0.0.0.0", 5000)
    try:
        parse_endpoint("no-port")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_encode_decode_roundtrip() -> None:
    samples = _frame(0.25, sequence_tag=0.5)
    datagram = encode_packet(samples, sequence=7)
    packet = decode_packet(datagram)

    assert packet.version == VERSION
    assert packet.sequence == 7
    assert packet.sample_rate == DEFAULT_SAMPLE_RATE
    assert packet.channels == 1
    assert packet.samples_per_packet == DEFAULT_SAMPLES_PER_PACKET
    assert packet.payload_length == DEFAULT_SAMPLES_PER_PACKET * 4
    assert len(datagram) == HEADER_SIZE + packet.payload_length
    assert np.allclose(packet.samples, samples)


def test_sequence_numbers_increment() -> None:
    sock = FakeUdpSocket()
    output = NetworkAudioOutput(
        "127.0.0.1:5000",
        sock=sock,
    )
    output.write(np.concatenate([_frame(0.1), _frame(0.2), _frame(0.3)]))
    output.close()

    assert len(sock.sent) == 3
    sequences = [decode_packet(data).sequence for data, _addr in sock.sent]
    assert sequences == [0, 1, 2]
    assert sock.sent[0][1] == ("127.0.0.1", 5000)
    assert sock.closed is False


def test_malformed_packets_are_rejected() -> None:
    good = encode_packet(_frame(0.1), sequence=0)

    try:
        decode_packet(good[:10])
        raise AssertionError("short datagram")
    except PacketError:
        pass

    bad_magic = b"XXXX" + good[4:]
    try:
        decode_packet(bad_magic)
        raise AssertionError("bad magic")
    except PacketError:
        pass

    header = bytearray(good[:HEADER_SIZE])
    magic, version, seq, sr, channels, n_samples, payload_len = HEADER_STRUCT.unpack(
        bytes(header)
    )
    header = HEADER_STRUCT.pack(
        magic,
        99,
        seq,
        sr,
        channels,
        n_samples,
        payload_len,
    )
    try:
        decode_packet(header + good[HEADER_SIZE:])
        raise AssertionError("bad version")
    except PacketError:
        pass

    truncated = good[:-4]
    try:
        decode_packet(truncated)
        raise AssertionError("truncated payload")
    except PacketError:
        pass

    receiver = NetworkAudioReceiver(prefill=1, capacity=4)
    receiver.handle_datagram(b"not-a-packet")
    receiver.handle_datagram(good[:8])
    assert receiver.stats.packets_malformed == 2
    assert receiver.stats.packets_received == 0


def test_packet_loss_outputs_silence() -> None:
    buffer = JitterBuffer(capacity=8, prefill=2)
    buffer.push(0, _frame(0.4, sequence_tag=0.0))
    buffer.push(1, _frame(0.5, sequence_tag=1.0))
    buffer.push(3, _frame(0.7, sequence_tag=3.0))

    first = buffer.pop()
    second = buffer.pop()
    lost = buffer.pop()
    third = buffer.pop()

    assert first is not None and first[0] == np.float32(0.0)
    assert second is not None and second[0] == np.float32(1.0)
    assert lost is not None and np.allclose(lost, 0.0)
    assert third is not None and third[0] == np.float32(3.0)
    assert buffer.stats.packets_lost == 1
    assert buffer.stats.sequence_gaps == 1
    assert buffer.stats.underruns == 0


def test_jitter_reordering() -> None:
    buffer = JitterBuffer(capacity=8, prefill=3)
    buffer.push(2, _frame(0.2, sequence_tag=2.0))
    buffer.push(0, _frame(0.0, sequence_tag=0.0))
    assert buffer.started is False
    buffer.push(1, _frame(0.1, sequence_tag=1.0))
    assert buffer.started is True

    frames = [buffer.pop() for _ in range(3)]
    tags = [float(frame[0]) for frame in frames if frame is not None]
    assert tags == [0.0, 1.0, 2.0]
    assert buffer.stats.packets_lost == 0


def test_late_and_overflow_packets_are_dropped() -> None:
    buffer = JitterBuffer(capacity=4, prefill=1)
    buffer.push(0, _frame(0.1))
    played = buffer.pop()
    assert played is not None

    buffer.push(0, _frame(0.9))
    assert buffer.stats.packets_dropped_late == 1

    buffer.push(10, _frame(0.2))
    assert buffer.stats.packets_dropped_overflow == 1


def test_underrun_outputs_silence() -> None:
    buffer = JitterBuffer(capacity=4, prefill=1)
    buffer.push(0, _frame(0.3))
    assert buffer.pop() is not None

    silence = buffer.pop()
    assert silence is not None
    assert np.allclose(silence, 0.0)
    assert buffer.stats.underruns == 1


def test_mono_to_stereo_duplicates_channels() -> None:
    mono = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    stereo = mono_to_stereo(mono)
    assert stereo.shape == (3, 2)
    assert np.allclose(stereo[:, 0], mono)
    assert np.allclose(stereo[:, 1], mono)

    pcm = float_to_int16(stereo)
    assert pcm.dtype == np.dtype("<i2")
    assert pcm.shape == (3, 2)
    assert pcm[0, 0] == pcm[0, 1]


def test_network_output_packetizes_arbitrary_chunks() -> None:
    sock = FakeUdpSocket()
    output = NetworkAudioOutput("192.0.2.10:5000", sock=sock)
    output.write(np.full(300, 0.1, dtype=np.float32))
    assert output.packets_sent == 0
    output.write(np.full(700, 0.2, dtype=np.float32))
    assert output.packets_sent == 2
    output.close()
    assert output.packets_sent == 3

    packets = [decode_packet(data) for data, _addr in sock.sent]
    assert [packet.sequence for packet in packets] == [0, 1, 2]
    recovered = np.concatenate([packet.samples for packet in packets])
    assert recovered.size == 3 * DEFAULT_SAMPLES_PER_PACKET
    assert np.allclose(recovered[:300], 0.1)
    assert np.allclose(recovered[300:1000], 0.2)
    assert np.allclose(recovered[1000:], 0.0)


def test_pipeline_writes_to_network_output() -> None:
    sock = FakeUdpSocket()
    chunks = [
        np.full(480, 0.11, dtype=np.float32),
        np.full(480, 0.22, dtype=np.float32),
    ]
    pipeline = StreamingPipeline(
        FakeAudioInput(chunks, sample_rate=DEFAULT_SAMPLE_RATE),
        NetworkAudioOutput("127.0.0.1:5000", sock=sock),
        enhancer=None,
        read_chunk_size=480,
        passthrough=True,
    )
    pipeline.run()

    assert len(sock.sent) == 2
    assert decode_packet(sock.sent[0][0]).sequence == 0
    assert np.allclose(decode_packet(sock.sent[1][0]).samples[0], 0.22)


def test_pi_receiver_import_boundary() -> None:
    """Pi receiver modules must not load drdo_anc.audio (and thus soundfile)."""

    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys
from drdo_anc.network_demo.protocol import decode_packet
from drdo_anc.network_demo.receiver import NetworkAudioReceiver
from drdo_anc.network_demo.alsa_playback import AlsaAplayOutput
assert "soundfile" not in sys.modules
assert "drdo_anc.audio.io" not in sys.modules
print("OK")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"import boundary failed:\\n{result.stdout}\\n{result.stderr}"
        )
    assert "OK" in result.stdout


def test_receiver_script_imports_network_demo_only() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_network_audio_receiver.py",
            "--help",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    assert "--output-device" in result.stdout


def test_header_layout() -> None:
    datagram = encode_packet(_frame(0.0), sequence=42)
    magic, version, seq, sr, channels, n_samples, payload_len = HEADER_STRUCT.unpack(
        datagram[:HEADER_SIZE]
    )
    assert magic == MAGIC
    assert version == VERSION
    assert seq == 42
    assert sr == 48_000
    assert channels == 1
    assert n_samples == 480
    assert payload_len == 1920
    assert struct.calcsize(HEADER_STRUCT.format) == 24


def main() -> None:
    print("=" * 70)
    print("DRDO-ANC | Network Audio Tests")
    print("=" * 70)

    tests = [
        test_parse_endpoint,
        test_encode_decode_roundtrip,
        test_sequence_numbers_increment,
        test_malformed_packets_are_rejected,
        test_packet_loss_outputs_silence,
        test_jitter_reordering,
        test_late_and_overflow_packets_are_dropped,
        test_underrun_outputs_silence,
        test_mono_to_stereo_duplicates_channels,
        test_network_output_packetizes_arbitrary_chunks,
        test_pipeline_writes_to_network_output,
        test_pi_receiver_import_boundary,
        test_receiver_script_imports_network_demo_only,
        test_header_layout,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
