FROM python:3.11-slim-bookworm

ARG DEEPFILTERNET_REPO=https://github.com/Rikorose/DeepFilterNet.git

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cargo \
        git \
        libasound2 \
        libsndfile1 \
        portaudio19-dev \
        pkg-config \
        rustc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/drdo-anc

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY models ./models

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        -e ".[gui]" \
        soundfile \
        numpy \
        torch \
        'maturin>=1.3,<1.5'

RUN mkdir -p external \
    && git clone --depth 1 "${DEEPFILTERNET_REPO}" external/DeepFilterNet \
    && cd external/DeepFilterNet \
    && maturin develop --release -m pyDF/Cargo.toml \
    && test -f target/release/libdf.so \
    && test -f models/DeepFilterNet3_onnx.tar.gz

RUN python -c "import df, numpy, sounddevice, soundfile, torch, PySide6"

ENTRYPOINT ["python", "scripts/run_live_enhancement.py"]
CMD ["--model", "DeepFilterNet3", "--diagnose-audio"]
