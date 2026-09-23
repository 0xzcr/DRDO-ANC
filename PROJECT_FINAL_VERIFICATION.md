# SIH ANC — Final Verification Report

Verification date: 2026-09-11. Code was inspected and executed; documentation was not treated as source of truth. No architecture redesign, no NLMS integration, no default-model switch. Device indexes below are **this machine, this session only**.

## 1. Environment

OS: Windows 10 (10.0.26200), `Windows-10-10.0.26200-SP0`

Python: 3.11.9 (`C:\Users\debar\AppData\Local\Programs\Python\Python311\python.exe`)

PyTorch: 2.8.0+cpu

DeepFilterNet version: `df` 0.5.6

Commit/hash if available: `1a85940b2932fa5b571c26f9c96a134ca7dfe78e`

Notes:

- Repo `.venv\Scripts\python.exe` is not present; commands used the system interpreter. Scripts add `src` to `sys.path`.
- `PySide6` was **not** installed at the start of this session (`ModuleNotFoundError`). It was installed for GUI launch (`PySide6 6.11.2`). That is an environment package, not a source change.
- Native streaming uses `external/DeepFilterNet/target/release/df.dll` (Windows). Fine-tuned weights live under `models/dfn3_finetuned/` (gitignored models tree).

## 2. Audio Devices

Listed with the project command:

```text
python scripts/run_live_enhancement.py --list-devices
```

PortAudio **host default** if `--input-device` / `--output-device` are omitted: `sd.default.device = [1, 4]` (MME Microphone Array + MME AB13X speakers, **44100 Hz**). That default is **not** the 48 kHz WASAPI pair used for DeepFilterNet3. Do not omit the device flags for live DF3 on this PC.

### Microphone/Input

Device index: **15**

Device name: Microphone Array (Realtek(R) Audio)

Host API: Windows WASAPI

Input channels: 2 (backend downmixes to mono)

Default sample rate: 48000 Hz

Actual test sample rate: 48000 Hz

Currently selected as microphone input: **yes**, via `--input-device 15`

Exact argument: `--input-device 15`

Indexes are **machine-dependent**. They were 9/8 on a previous day on the same hardware; they must be rediscovered with `--list-devices`.

### Output/Headphones

Device index: **13**

Device name: Speakers (AB13X USB Audio)

Host API: Windows WASAPI

Output channels: 2

Actual test sample rate: 48000 Hz

Currently selected as output: **yes**, via `--output-device 13`

Exact argument: `--output-device 13`

Alternate USB mic on this machine (not used for the DF3 soaks): WASAPI index **14**, Microphone (AB13X USB Audio).

## 3. Exact Live Command

Pretrained (default model name; do not omit devices):

```text
python scripts/run_live_enhancement.py --model DeepFilterNet3 --input-device 15 --output-device 13
```

Fine-tuned (**explicit only**; not the default):

```text
python scripts/run_live_enhancement.py --model DeepFilterNet3-Finetuned --input-device 15 --output-device 13
```

Timed soaks used the same devices:

```text
python scripts/run_live_soak.py --model DeepFilterNet3 --duration-s 30 --input-device 15 --output-device 13
python scripts/run_live_soak.py --model DeepFilterNet3-Finetuned --duration-s 30 --input-device 15 --output-device 13
```

GUI (CLI device flags only; no in-window device picker):

```text
python scripts/run_live_gui.py --input-device 15 --output-device 13
```

`INPUT DEVICE INDEX` and `OUTPUT DEVICE INDEX` must be re-listed on every machine and after Windows device-graph changes.

## 4. Microphone Test

**PASS**

Command: `python scripts/test_live_passthrough.py --mode capture --input-device 15 --duration 8`

Artifact: `data/live_soak_reports/capture_realtek_wasapi15.wav` (768044 bytes)

Duration: 8.0 s (384000 samples @ 48 kHz)

Samples: 384000 (mono WAV after downmix)

Input overflows: 0 (capture session)

Audio detected: yes (364880 nonzero samples)

RMS/level: peak 0.504, RMS 0.065; all finite; no NaN/Inf

Host: WASAPI index 15, 2 input channels, 48000 Hz

## 5. Pretrained DF3 Live Test

**PASS** (software duplex stream)

Report: `data/live_soak_reports/soak_2026-09-11_11-09-13.json`

Duration: wall elapsed 30.21 s (target 30 s)

Wall RTF (`pipeline_stats.realtime_ratio`): 0.988

Processing time: 5.77 s

Chunks: 1391 (host blocksize 1024)

Input overflows: 0

Samples in: 1,424,384

Samples out: 1,424,384

Errors: `null`

Model: DeepFilterNet3 @ 48000 Hz; eval-only streaming delay 1440 samples (30 ms). Native frames are 480 samples / 10 ms (`test_enhancer_streaming`: outputs in multiples of 480, e.g. 960).

## 6. Fine-Tuned DF3 Live Test

**PASS** (software duplex stream; model is **not** the live default)

Report: `data/live_soak_reports/soak_2026-09-11_11-10-01.json`

Duration: wall elapsed 30.26 s

Wall RTF: 0.986

Processing time: 5.39 s

Chunks: 1388

Input overflows: 0

Samples in: 1,421,312

Samples out: 1,421,312

Errors: `null`

Load: `DeepFilterNet3-Finetuned`, checkpoint epoch 130, DF rate 48000 Hz. Streaming pipeline smoke: `scripts/test_dfn3_finetuned_live.py` all PASS.

## 7. Physical Output

Raw: **NOT TESTED — HARDWARE REQUIRED** for human listening. Software passthrough 15→13 (8 s pipeline) wrote equal samples in/out, peak ≈ 0.516, 0 overflows (**software routing PASS**).

Pretrained: **NOT TESTED — HARDWARE REQUIRED** for listening. Soak wrote 1,424,384 samples to device 13, peak_output 0.521, 0 overflows (**software routing PASS**).

Fine-tuned: **NOT TESTED — HARDWARE REQUIRED** for listening. Soak wrote 1,421,312 samples to device 13, peak_output 0.535, 0 overflows (**software routing PASS**).

PHYSICAL LISTENING: **MANUAL VERIFICATION REQUIRED**

Do not treat software sample counts as headphone confirmation.

Diagnostic: `test_live_passthrough.py --mode sine --output-device 13` failed with `PortAudioError: Blocking API not supported yet [WDM-KS]`. Duplex WASAPI pipeline to index 13 succeeded. Sine-mode failure is **not** a DF3 live-pipeline failure.

## 8. GUI/Demo

Application launch: **PASS** after installing PySide6. `python scripts/run_live_gui.py --fake --output-device 13` stayed running with no traceback (~35 s). Initial attempt failed with `ModuleNotFoundError: No module named 'PySide6'`.

Demo loading: **PASS** (`scripts/test_gui_demo.py`, validated demo catalog).

Play: **PASS** in automated demo controller tests. Interactive Play click in the Qt window: **NOT TESTED — MANUAL**

Pause: **PASS** automated. Interactive: **NOT TESTED — MANUAL**

Stop: **PASS** automated. Interactive: **NOT TESTED — MANUAL**

A/B routing: **PASS** automated (`A Raw` / `B Enhanced` on `SelectableAudioOutput` / queued playback). Interactive: **NOT TESTED — MANUAL**

Audio output: CLI `--output-device` is wired; physical GUI listen: **MANUAL VERIFICATION REQUIRED**

Shutdown: process was stopped from the verification harness after idle launch. In-app clean quit: **NOT TESTED — MANUAL**

Usability gap (not redesigned): QML has **no microphone/device picker**. Input/output are CLI-only (`--input-device`, `--output-device`). Default GUI model remains `DeepFilterNet3`.

`scripts/test_gui_demo.py`: 21 tests PASS. `scripts/test_gui_waveform.py`: 4 tests PASS.

## 9. Offline Benchmark

Existing full protocol `sih26-eval-v1`, 60 cases × offline+streaming = 120 rows. Files: `data/benchmark_results/dfn3_finetuned_compare/pretrained_full.json` and `finetuned_full.json`. Not re-run in this session (deterministic prior run; 0 failures).

Pretrained (`DeepFilterNet3`):

SI-SDR: 12.71 (mean)

STOI: 0.667

PESQ: 1.850

SNR: 12.30

RTF: median 9.92, mean 12.13

Fine-tuned (`DeepFilterNet3-Finetuned`):

SI-SDR: 14.95

STOI: 0.708

PESQ: 2.119

SNR: 15.01

RTF: median 8.32, mean 9.97

Success/failure counts: **120 / 0** each model (240 successful rows total across both JSON files).

Fine-tuned is **not** switched to default despite higher SI-SDR; live default remains pretrained pending physical A/B listening.

## 10. Automated Tests

Primary log: `data/live_soak_reports/automated_tests_2026-09-11.txt` (13 scripts).

That log: **110 PASS**, **0 FAIL**, **2 SKIP**, 10× `ALL TESTS PASSED`.

Skips: Hugging Face / `SIH26_INTEGRATION=1` cases (e.g. zip integration, DF3 manifest integration).

Re-run this session (exit 0): `test_dfn3_finetuned_live` (5 PASS), `test_enhancer_streaming` (PASS; 480-sample frames), `test_noise_classifier` (10 PASS), `test_zip_manifest_dataset` (9 PASS + 1 SKIP).

Combined relevant suite: **no failures**. Approximate total **134+ PASS**, **3 SKIP**, **0 FAIL** (log encoding mixed NULs from the console capture; counts from `PASS:` after NUL strip plus the four re-runs).

List failures: **none**.

Not counted as test-suite failure: sine blocking-API WDM-KS error (separate diagnostic mode).

## 11. Raspberry Pi Readiness Risks

No Pi port was attempted.

### BLOCKING

- Native library is **Windows `df.dll`** (`src/drdo_anc/enhancement/deepfilternet.py`, `finetuned.py`). Linux/ARM needs a matching `.so` and build.
- Fine-tuned ONNX/checkpoint tree is **gitignored** (`models/dfn3_finetuned/models/`); must be copied onto the device.
- x86/x64 Windows ONNX + `df.dll` are **not** aarch64 artifacts.

### NON-BLOCKING

- WASAPI host API does not exist on Linux; PortAudio will use ALSA/Pulse/JACK. Device **indexes will differ**.
- Omitting device flags uses host defaults (here MME 44.1 kHz), which is wrong for a 48 kHz DF3 boundary.
- GUI extra `PySide6` is optional and was missing until installed; Pi GUI is heavy.
- DF logger `fatal: not a git repository` when cwd is not a git repo (noise, not a crash).
- Hard-coded **48000 Hz** model rate is correct for DF3, not a bug.

### MANUAL DEPLOYMENT TASKS

- Run `--list-devices` on the Pi; never hard-code 15/13.
- Install PortAudio, PyTorch (or CPU wheel), `df`, native lib, models.
- Confirm live RTF on ARM; this PC’s processing RTF was ~0.19 of wall (5.8 s / 30 s) on CPU — ARM is unknown.
- Physical headphone check on the target hardware.

## 12. Final Verdict

**READY WITH MANUAL CHECKS**

Live mic capture, pretrained DF3 duplex, and fine-tuned DF3 duplex all completed on WASAPI **15 → 13** at 48 kHz with 0 overflows and matched sample counts. Offline 60-case metrics exist with 0 failures. Automated relevant tests passed. Default model was **not** changed.

Remaining: listen on headphones (raw / pretrained / fine-tuned), click through Qt transport/A/B, confirm GUI extra is installed on other machines, and treat device indexes as per-machine. Raspberry Pi is **not** ready until native Linux/ARM `df` and models are supplied.
