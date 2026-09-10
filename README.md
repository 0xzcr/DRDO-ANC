# DRDO-ANC

## Raspberry Pi setup

On a 64-bit ARM Raspberry Pi with network access, run:

```bash
cd ~/DRDO-ANC
bash scripts/setup_raspberry_pi.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs the project and GUI/audio
dependencies, builds the DeepFilterNet Python extension, validates the
checked-in fine-tuned artifact, and requires the Linux native library at
`external/DeepFilterNet/target/release/libdf.so`.

If `external/DeepFilterNet` already exists, the script reuses it. Set
`DEEPFILTERNET_REPO` only when a compatible fork is required:

```bash
DEEPFILTERNET_REPO=https://github.com/your-org/DeepFilterNet.git \\
  bash scripts/setup_raspberry_pi.sh
```

Run the fine-tuned model after setup:

```bash
source .venv/bin/activate
python scripts/run_live_enhancement.py --model DeepFilterNet3-Finetuned
```
