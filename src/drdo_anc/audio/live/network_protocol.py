"""Re-export canonical demo protocol (see ``drdo_anc.network_demo.protocol``)."""

from drdo_anc.network_demo.protocol import (  # noqa: F401
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLES_PER_PACKET,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    VERSION,
    AudioPacket,
    PacketError,
    decode_packet,
    encode_packet,
    mono_to_stereo,
    parse_endpoint,
)
