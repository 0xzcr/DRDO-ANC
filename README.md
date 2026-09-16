# DRDO-ANC

## Raspberry Pi setup

On a 64-bit ARM Raspberry Pi with network access, run:

```bash
cd ~/DRDO-ANC
bash scripts/setup_raspberry_pi.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs the project and core audio
dependencies, builds the DeepFilterNet Python extension, validates the
checked-in fine-tuned artifact, and requires the Linux native library at
`external/DeepFilterNet/target/release/libdf.so`.

The package metadata provides optional groups for non-live workflows:

```bash
pip install -e ".[evaluation]"  # PESQ, STOI, and SciPy metrics
pip install -e ".[training]"    # scikit-learn and dataset helpers
```

The live stack intentionally follows DeepFilterNet's compatibility range:
NumPy stays below 2.0, and `torch`/`torchaudio` are installed together below
major version 3. The setup script also exposes the cloned DeepFilterNet
Python package beside its compiled `libdf` extension, so both come from the
same checkout.

The Raspberry Pi setup is headless by default because PySide6 wheels are not
available for every ARM64/Python/glibc combination. The live audio CLI does
not need PySide6. To opt into the GUI installation, use:

```bash
DRDO_ANC_INSTALL_GUI=1 bash scripts/setup_raspberry_pi.sh
```

Re-running the setup command is safe: it reuses `.venv`, existing Python
packages, the DeepFilterNet checkout, and an existing native build. To force
dependency checks and rebuild the native extension:

```bash
DRDO_ANC_FORCE_SETUP=1 bash scripts/setup_raspberry_pi.sh
```

If that fails with `no matching distribution for shiboken6`, use a newer
64-bit Raspberry Pi OS/Python combination with a compatible PySide6 wheel, or
run the GUI on a desktop machine. PySide6 and shiboken6 must be installed at
the same version.

If `external/DeepFilterNet` already exists, the script reuses it. Set
`DEEPFILTERNET_REPO` only when a compatible fork is required:

```bash
DEEPFILTERNET_REPO=https://github.com/your-org/DeepFilterNet.git \\
  bash scripts/setup_raspberry_pi.sh
```

The default DeepFilterNet checkout is the stable `v0.5.6` release. Override
it only when using a tested compatible revision:

```bash
DEEPFILTERNET_REF=v0.5.6 bash scripts/setup_raspberry_pi.sh
```

Run the fine-tuned model after setup:

```bash
source .venv/bin/activate
python scripts/run_live_enhancement.py --model DeepFilterNet3-Finetuned
```

## Docker

The Docker image targets Linux audio hosts and supports `linux/amd64` and
`linux/arm64` builds. It runs the existing local microphone-to-speaker CLI;
it does not yet provide Pi-to-PC network audio transport.

Build and run on a Linux audio host:

```bash
docker compose build
docker compose run --rm --device /dev/snd drdo-anc --list-devices
docker compose up
```

The DeepFilterNet checkout must provide the native `libdf.so` API expected by
this project. Use a compatible fork at build time when necessary:

```bash
DEEPFILTERNET_REPO=https://github.com/your-org/DeepFilterNet.git \\
DEEPFILTERNET_REF=your-compatible-branch docker compose build
```

The image runs the complete `[all]` project dependency set and fails during
build if the native library is missing `df_create`, `df_process_frame`, or
`df_free`.
