#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly VENV_DIR="${PROJECT_ROOT}/.venv"
readonly DF_DIR="${PROJECT_ROOT}/external/DeepFilterNet"
readonly MODEL_ROOT="${PROJECT_ROOT}/models/dfn3_finetuned"
readonly DF_REPO="${DEEPFILTERNET_REPO:-https://github.com/Rikorose/DeepFilterNet.git}"
readonly INSTALL_GUI="${DRDO_ANC_INSTALL_GUI:-0}"
readonly FORCE_SETUP="${DRDO_ANC_FORCE_SETUP:-0}"
readonly BUILD_TMP="${PROJECT_ROOT}/.tmp"
readonly CARGO_HOME_DIR="${PROJECT_ROOT}/.cargo"
readonly PIP_BUILD_TRACKER_DIR="${PROJECT_ROOT}/.pip-build-tracker"
readonly DF_CAPI_STAGE="${BUILD_TMP}/df-capi"

mkdir -p \
  "${BUILD_TMP}" \
  "${CARGO_HOME_DIR}" \
  "${PIP_BUILD_TRACKER_DIR}" \
  "${DF_CAPI_STAGE}"

export TMPDIR="${BUILD_TMP}"
export CARGO_HOME="${CARGO_HOME_DIR}"
export PIP_BUILD_TRACKER="${PIP_BUILD_TRACKER_DIR}"
export PIP_NO_CACHE_DIR=1
trap 'printf "\nERROR: setup failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v python3 >/dev/null || die "python3 is required"
command -v git >/dev/null || die "git is required"
command -v cargo >/dev/null || die "cargo is required; install Rust/Cargo before running setup"
command -v rustc >/dev/null || die "rustc is required; install Rust before running setup"

case "${INSTALL_GUI}" in
  0|1) ;;
  *) die "DRDO_ANC_INSTALL_GUI must be 0 or 1" ;;
esac
case "${FORCE_SETUP}" in
  0|1) ;;
  *) die "DRDO_ANC_FORCE_SETUP must be 0 or 1" ;;
esac

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ required; found {sys.version.split()[0]}")
PY

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  die "64-bit ARM is required; found $(uname -m)"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}" || die "cannot create .venv; install the OS python3-venv package"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

readonly VENV_READY_MARKER="${VENV_DIR}/.drdo_anc_tools_ready"
if [[ "${FORCE_SETUP}" == "1" || ! -f "${VENV_READY_MARKER}" ]]; then
  python -m pip install \
    --no-cache-dir \
    --retries 20 \
    --timeout 120 \
    --upgrade pip setuptools wheel
  touch "${VENV_READY_MARKER}"
fi

if [[ "${FORCE_SETUP}" == "1" ]] || ! python -c \
  'import numpy, sounddevice, soundfile, torch' >/dev/null 2>&1 || \
  ! python -m pip show drdo-anc maturin >/dev/null 2>&1; then
  python -m pip install \
    --no-cache-dir \
    --retries 20 \
    --timeout 120 \
    -e "${PROJECT_ROOT}" \
    'maturin>=1.3,<1.5'
else
  printf 'Python dependencies already installed; skipping pip install.\n'
fi

if [[ "${INSTALL_GUI}" == "1" ]] && \
   { [[ "${FORCE_SETUP}" == "1" ]] || ! python -c 'import PySide6' >/dev/null 2>&1; }; then
  python -m pip install \
    --no-cache-dir \
    --retries 20 \
    --timeout 120 \
    'PySide6>=6.6'
fi

mkdir -p "${PROJECT_ROOT}/external"
if [[ ! -d "${DF_DIR}" ]]; then
  git clone --depth 1 "${DF_REPO}" "${DF_DIR}"
elif [[ ! -f "${DF_DIR}/pyDF/Cargo.toml" ]]; then
  die "${DF_DIR} exists but is not a DeepFilterNet source checkout"
fi

[[ -f "${MODEL_ROOT}/live-finetuned/models/dfn3-epoch-130-onnx/_export_model/config.ini" ]] \
  || die "fine-tuned export config.ini is missing"
[[ -f "${MODEL_ROOT}/live-finetuned/models/dfn3-epoch-130-onnx/_export_model/checkpoints/model_130.ckpt" ]] \
  || die "fine-tuned checkpoint is missing"
for member in enc.onnx erb_dec.onnx df_dec.onnx config.ini; do
  [[ -f "${MODEL_ROOT}/live-finetuned/models/dfn3-epoch-130-onnx/onnx/${member}" ]] \
    || die "fine-tuned ONNX member is missing: ${member}"
done

native_library="${DF_DIR}/target/release/libdf.so"
if [[ "${FORCE_SETUP}" == "1" || ! -f "${native_library}" ]] || \
   ! python -c 'import df' >/dev/null 2>&1; then
(
  cd "${DF_DIR}"
  PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    maturin develop --release -m pyDF/Cargo.toml
  cargo install --locked cargo-c
  cargo install --locked cbindgen
  cargo cinstall --profile=release-lto -p deep_filter \
    --destdir="${DF_CAPI_STAGE}"

  native_library=""
  while IFS= read -r candidate; do
    if nm -D "${candidate}" | grep -q 'df_create'; then
      native_library="${candidate}"
      break
    fi
  done < <(find "${DF_CAPI_STAGE}" -type f -name 'libdeep_filter*.so')

  [[ -n "${native_library}" ]] || die "DeepFilterNet C API library with df_create was not built"
  cp "${native_library}" "${DF_DIR}/target/release/libdf.so"
)
else
  printf 'DeepFilterNet native extension already built; skipping maturin build.\n'
fi

[[ -f "${native_library}" ]] || die "native streaming library is missing: ${native_library}"

python - <<'PY'
import df
import numpy
import sounddevice
import soundfile
import torch
print("DRDO-ANC dependencies: OK")
print(f"Python: {__import__('sys').version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
PY

if [[ "${INSTALL_GUI}" == "1" ]]; then
  python -c 'import PySide6; print(f"PySide6: {PySide6.__version__}")'
fi

printf '\nSetup complete. Activate with:\n  source %s/bin/activate\n' "${VENV_DIR}"
