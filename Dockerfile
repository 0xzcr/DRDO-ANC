FROM python:3.11-slim-bookworm

ARG DEEPFILTERNET_REPO=https://github.com/Rikorose/DeepFilterNet.git
ARG DEEPFILTERNET_REF=v0.5.6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/drdo-anc/external/DeepFilterNet/DeepFilterNet \
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cargo \
        git \
        libasound2 \
        libdbus-1-3 \
        libegl1 \
        libgl1 \
        libsndfile1 \
        libssl-dev \
        libx11-6 \
        libxcb-cursor0 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        portaudio19-dev \
        pkg-config \
        rustc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/drdo-anc

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY models ./models

RUN test -f models/dfn3_finetuned/live-finetuned/models/dfn3-epoch-130-onnx/_export_model/checkpoints/model_130.ckpt \
    && test -f models/dfn3_finetuned/live-finetuned/models/dfn3-epoch-130-onnx/_export_model/config.ini \
    && for member in enc.onnx erb_dec.onnx df_dec.onnx config.ini; do \
        test -f "models/dfn3_finetuned/live-finetuned/models/dfn3-epoch-130-onnx/onnx/${member}"; \
    done

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        -e ".[all]" \
        'maturin>=1.3,<1.5'

RUN mkdir -p external \
    && git clone --depth 1 --branch "${DEEPFILTERNET_REF}" \
        "${DEEPFILTERNET_REPO}" external/DeepFilterNet \
    && cd external/DeepFilterNet \
    && maturin develop --release -m pyDF/Cargo.toml \
    && test -f models/DeepFilterNet3_onnx.tar.gz \
    && cargo install --locked cargo-c \
    && cargo install --locked cbindgen \
    && cargo cinstall --profile=release-lto -p deep_filter \
        --destdir=/tmp/df-capi \
    && native_library="" \
    && for candidate in $(find /tmp/df-capi -type f \
        \( -name 'libdeep_filter*.so' -o -name 'deep_filter*.dll' \) -print); do \
        if nm -D "${candidate}" 2>/dev/null | grep -q 'df_create'; then \
            native_library="${candidate}"; \
            break; \
        fi; \
    done \
    && test -n "${native_library}" \
    && cp "${native_library}" target/release/libdf.so \
    && nm -D target/release/libdf.so | grep -q 'df_create' \
    && nm -D target/release/libdf.so | grep -q 'df_process_frame' \
    && nm -D target/release/libdf.so | grep -q 'df_free'

RUN python -m pip check \
    && python -c "import df, numpy, sounddevice, soundfile, torch, torchaudio; assert int(numpy.__version__.split('.')[0]) < 2; assert torch.__version__.split('+')[0] == torchaudio.__version__.split('+')[0]; print('DRDO-ANC and DeepFilterNet dependencies: OK')"

ENTRYPOINT ["python", "scripts/run_live_enhancement.py"]
CMD ["--model", "DeepFilterNet3", "--diagnose-audio"]
