"""Raspberry Pi UDP audio receiver (playback only; no AI inference)."""

from __future__ import annotations

import argparse
import socket
import sys
import time

from drdo_anc.audio.live.network_playback import (
    AlsaAplayOutput,
    is_alsa_hw_device,
    open_receiver_output,
)
from drdo_anc.audio.live.network_protocol import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLES_PER_PACKET,
    parse_endpoint,
)
from drdo_anc.audio.live.network_receiver import (
    DEFAULT_JITTER_CAPACITY,
    DEFAULT_JITTER_PREFILL,
    NetworkAudioReceiver,
)


FRAME_DURATION_S = DEFAULT_SAMPLES_PER_PACKET / DEFAULT_SAMPLE_RATE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive enhanced 48 kHz audio over UDP and play it on this "
            "machine. AI inference is performed on the Windows host; this "
            "process is a network audio endpoint only."
        ),
    )
    parser.add_argument(
        "--listen",
        default="0.0.0.0:5000",
        help="UDP bind address host:port (default: 0.0.0.0:5000).",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help=(
            "Playback device. ALSA hw:/plughw: strings use aplay; "
            "otherwise a sounddevice index or name."
        ),
    )
    parser.add_argument(
        "--jitter-capacity",
        type=int,
        default=DEFAULT_JITTER_CAPACITY,
        help="Max packets held in the jitter buffer.",
    )
    parser.add_argument(
        "--jitter-prefill",
        type=int,
        default=DEFAULT_JITTER_PREFILL,
        help="Packets to collect before starting playback.",
    )
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=1.0,
        help="Seconds between status lines.",
    )
    return parser


def _drain_socket(sock: socket.socket, receiver: NetworkAudioReceiver) -> None:
    sock.setblocking(False)
    try:
        while True:
            datagram, _addr = sock.recvfrom(4096)
            receiver.handle_datagram(datagram)
    except BlockingIOError:
        return
    finally:
        sock.setblocking(True)


def main() -> None:
    args = _build_parser().parse_args()
    host, port = parse_endpoint(args.listen)

    receiver = NetworkAudioReceiver(
        capacity=args.jitter_capacity,
        prefill=args.jitter_prefill,
    )
    output = open_receiver_output(DEFAULT_SAMPLE_RATE, args.output_device)
    alsa = isinstance(output, AlsaAplayOutput) or (
        isinstance(args.output_device, str)
        and is_alsa_hw_device(args.output_device)
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.002)

    print("=" * 70)
    print("DRDO-ANC | Network Audio Receiver")
    print("=" * 70)
    print("Role:         playback endpoint only (no AI inference)")
    print(f"Listen:       {host}:{port}")
    print(f"Sample rate:  {DEFAULT_SAMPLE_RATE} Hz")
    print(f"Packet:       {DEFAULT_SAMPLES_PER_PACKET} samples (10 ms, mono)")
    print(
        "Output:       "
        f"{args.output_device if args.output_device is not None else 'default'}"
    )
    print(
        f"Jitter:       prefill={args.jitter_prefill} "
        f"capacity={args.jitter_capacity}"
    )
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    last_status = time.perf_counter()
    playing = False

    try:
        while True:
            try:
                datagram, _addr = sock.recvfrom(4096)
                receiver.handle_datagram(datagram)
            except socket.timeout:
                pass

            if not playing and receiver.buffer.started:
                playing = True
                print("Playback started.", flush=True)

            if not playing:
                now = time.perf_counter()
                if now - last_status >= args.status_interval_s:
                    print(
                        receiver.stats.format_line(
                            buffer_level=receiver.buffer.level,
                            buffer_capacity=receiver.buffer.capacity,
                        ),
                        flush=True,
                    )
                    last_status = now
                continue

            _drain_socket(sock, receiver)
            frame_started = time.perf_counter()
            stereo = receiver.pull_stereo_frame()
            if stereo is None:
                continue

            if alsa:
                output.write_stereo(stereo)
            else:
                output.write(stereo[:, 0])

            remain = FRAME_DURATION_S - (time.perf_counter() - frame_started)
            if remain > 0:
                time.sleep(remain)

            now = time.perf_counter()
            if now - last_status >= args.status_interval_s:
                print(
                    receiver.stats.format_line(
                        buffer_level=receiver.buffer.level,
                        buffer_capacity=receiver.buffer.capacity,
                    ),
                    flush=True,
                )
                last_status = now
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        output.close()

    print(
        receiver.stats.format_line(
            buffer_level=receiver.buffer.level,
            buffer_capacity=receiver.buffer.capacity,
        )
    )
    print("Receiver stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
