import argparse
import sys

from drdo_anc.audio.live import (
    SoundDeviceAudioInput,
    SoundDeviceAudioOutput,
    StreamingPipeline,
    format_device_listing,
)
from drdo_anc.enhancement import create_enhancer, list_models


DEFAULT_MODEL_NAME = "DeepFilterNet3"
DEFAULT_READ_CHUNK_SIZE = 1024


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream live microphone audio through a registered enhancer "
            "to the speaker. Use --passthrough to measure hardware latency "
            "without model inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Audio semantics\n"
            "-----------------\n"
            "Sample rate:\n"
            "  Enhancement mode uses the selected model's sample rate "
            "(48 kHz for DeepFilterNet3). Input and output devices must "
            "be opened at that rate.\n"
            "  Pass-through mode defaults to 48 kHz unless --sample-rate "
            "is provided.\n"
            "\n"
            "Chunk sizes:\n"
            "  --chunk-size controls how many samples are requested per "
            "read() call. The host may return fewer samples. Arbitrary "
            "hardware chunks are forwarded directly to "
            "Enhancer.process_stream(); StreamingBuffer inside the "
            "enhancer converts them to model frames.\n"
            "\n"
            "Device selection:\n"
            "  Use --list-devices to show PortAudio device indices. Pass "
            "an integer index or host-specific device name to "
            "--input-device / --output-device. Omit both to use the host "
            "default devices.\n"
            "\n"
            "Shutdown:\n"
            "  Press Ctrl+C to stop. The pipeline calls enhancer.flush() "
            "exactly once, writes any remaining enhanced samples, then "
            "closes the audio streams."
        ),
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List host audio devices and exit.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=list_models(),
        help="Registered enhancer model name.",
    )
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help=(
            "Copy microphone input directly to the speaker without "
            "enhancement."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=(
            "Audio sample rate in Hz. Defaults to the model sample rate in "
            "enhancement mode, or 48000 in pass-through mode."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_READ_CHUNK_SIZE,
        help=(
            "Number of samples requested per AudioInput.read() call."
        ),
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Input device index or name (sounddevice/PortAudio).",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Output device index or name (sounddevice/PortAudio).",
    )

    return parser


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_devices:
        print(format_device_listing())
        return

    if args.passthrough:
        sample_rate = args.sample_rate or 48_000
        enhancer = None
        mode_label = "pass-through"
    else:
        enhancer = create_enhancer(args.model)
        sample_rate = args.sample_rate or enhancer.sample_rate()
        mode_label = args.model

    if sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive.")

    input_device = _parse_device(args.input_device)
    output_device = _parse_device(args.output_device)

    print("=" * 70)
    print("DRDO-ANC | Live Audio Streaming")
    print("=" * 70)
    print(f"Mode:        {mode_label}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk size:  {args.chunk_size} samples/read")
    print(f"Input dev:   {input_device if input_device is not None else 'default'}")
    print(
        f"Output dev:  {output_device if output_device is not None else 'default'}"
    )
    print("\nPress Ctrl+C to stop.")
    print("=" * 70)

    audio_input = SoundDeviceAudioInput(
        sample_rate,
        device=input_device,
    )
    audio_output = SoundDeviceAudioOutput(
        sample_rate,
        device=output_device,
    )

    pipeline = StreamingPipeline(
        audio_input,
        audio_output,
        enhancer,
        read_chunk_size=args.chunk_size,
    )

    pipeline.run()
    print("\nStream finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
