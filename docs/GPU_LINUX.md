# Linux GPU acceleration for transcription (evidence, not a claim)

**Checked: 2026-10-30.** This note describes the documented upstream support and a reproducible investigation/benchmark procedure. It is not a claim that GPU acceleration works on the project laptop or that it is faster. No benchmark was run here.

## TL;DR

- Upstream faster-whisper/CTranslate2 documents NVIDIA CUDA on Linux. Current faster-whisper guidance calls for CUDA 12 cuBLAS and cuDNN 9; older CTranslate2 compatibility workarounds are version-pinned below.[1]
- CTranslate2 upstream release v4.8.4 publishes a separate Linux ROCm wheel archive; AMD support is a distinct wheel/runtime path, not the ordinary PyPI CUDA install.[2][5] AMD support depends on exact GPU, ROCm version, OS/kernel and package compatibility; check AMD’s current matrix before installing.[6][7]
- No Vulkan backend is documented by CTranslate2: its installation/build docs enumerate CUDA and HIP GPU backends, and its runtime device API reports `cpu` or `cuda`; do not infer that Vulkan is supported merely because other Whisper implementations support it.[2][4]
- The project’s GPU status remains **unverified** until a real target machine completes detection and a controlled benchmark. This research host is not the laptop and has no GPU tools/devices exposed; therefore speedup is **not measured**.

## Scope and provenance

- **Cited facts** below are supported by the upstream sources in [Sources](#sources).
- **Observed on the research host (2026-10-30):** `lspci`, `nvidia-smi`, and `rocminfo` were not installed; `lspci` yielded no device listing. The host reports Linux `x86_64` kernel `7.0.14-17-pve` on a PVE host. This does not establish the laptop’s GPU or lack thereof; the VM/container may not expose PCI devices. `ctranslate2` and `faster_whisper` were not checked/installed. No model, audio, transcription, GPU, timing, or memory benchmark was run.
- **Verdict:** hardware/model/backend compatibility and any performance difference on CachyOS remain unverified pending measurements on the actual laptop.

## What the backends mean

| Backend | Evidence-based support status | Version / qualification |
| --- | --- | --- |
| NVIDIA CUDA | Documented on Linux; official faster-whisper instructions require CUDA 12 cuBLAS + cuDNN 9 for current CTranslate2. Its note says CTranslate2 3.24.0 is the workaround for CUDA 11 + cuDNN 8; CTranslate2 4.4.0 for CUDA 12 + cuDNN 8.[1] | Prefer the current dependency-compatible stack and NVIDIA’s official driver/toolkit installation path. The project currently allows `faster-whisper>=1.1,<2`; do not edit that constraint or assume every allowed CTranslate2 resolution has the same GPU ABI/support. Confirm installed versions and runtime detection locally. |
| AMD ROCm/HIP | CTranslate2 has an AMD HIP backend and its v4.8.4 release includes a separate `rocm-python-wheels-Linux.zip` asset.[2][5] AMD’s ROCm guide describes ROCm 6.0+ and PyTorch 2.1+ for its dated CTranslate2 example, which builds a ROCm fork/container.[8] | The guide is dated 2024 and uses a fork/container; it is not current faster-whisper pip setup guidance. Check official ROCm OS/GPU matrices and the exact release wheel contents/compatibility first. v4.8.4 release archive exists but no ROCm install was attempted here. CachyOS is not listed in ROCm 7.0 supported Linux distributions.[6] |
| Vulkan | No supported Vulkan device backend appears in current CTranslate2 install/build docs or supported device API; no version/install recipe can be recommended for faster-whisper/CTranslate2.[2][4] | Vulkan support in another Whisper runtime is not evidence of CTranslate2 support. Treat Vulkan as unsupported for this stack unless upstream adds explicit documentation/backend. |

### Compute types

CTranslate2’s generic `compute_type` set is not a promise that every type works efficiently on every device. Upstream explicitly recommends querying the actual build/device using `ctranslate2.get_supported_compute_types(device, device_index)`; the documented device names are `cpu` and `cuda`.[3][4]

- **CPU baseline:** faster-whisper documents `device="cpu", compute_type="int8"`.[1] CTranslate2 CPU type support can vary with CPU architecture/backend; its docs state float16 may be converted to float32 on CPU, so specify and record the exact runtime-reported set rather than assuming.[3]
- **NVIDIA CUDA:** faster-whisper’s documented examples use `float16`, `int8_float16`, or `int8`.[1] CTranslate2 documents GPU-specific behavior by NVIDIA compute capability: FP16 optimized on capability >=7.0; BF16 >=8.0; INT8 support varies (>=7.0 or 6.1 in its documented matrix); some requested types may be converted to another type.[3] Start with `float16` for a supported NVIDIA GPU, and use `get_supported_compute_types("cuda")` to establish actual support. Do not assume BF16/INT8 support based only on the type name.
- **AMD ROCm:** the current public Python `get_supported_compute_types` reference names only `cpu` and `cuda`, and the official PyPI/CUDA install page is not an AMD install recipe.[2][4][5] Do not claim that the CUDA device string or CUDA type list proves ROCm functionality. The released ROCm wheel must be validated against its matching runtime and queried/benchmarked using the API supported by that build; use only values actually reported/accepted by that wheel. No reliable, current official per-type faster-whisper/ROCm matrix was found in the cited sources.

## `--compute-device`

`listn transcribe|live --compute-device`, `LISTN_COMPUTE_DEVICE` and
`compute_device` in `settings.json` choose where faster-whisper runs.
Precedence: CLI > environment > settings file; an invalid value never
falls back silently. The `--compute-device` flag is rejected by argparse
with a readable "invalid choice" message (exit 2, no traceback); an
invalid `LISTN_COMPUTE_DEVICE` environment value raises `SettingsError`
(a `ValueError`) from settings loading.

| Value | Behaviour |
|---|---|
| `auto` (default) | Main's shipped behaviour: GPU when this machine's CUDA (or ROCm) runtime is actually usable, otherwise CPU. The availability checks run in a child process (`listn/transcriber.py:resolve_device()`), because the native loader and the driver API can end a process without raising a Python exception; a child killed by a signal, a timeout or unparseable output all read as "not usable". What the user actually sees on `auto`: a machine with no GPU at all announces only `device: cpu`, with no note; the note `CUDA libraries unavailable — transcribing on CPU` appears only when a GPU is visible but unusable (or an availability failure happens mid-run). The probe's specific reason stays in `device_message`, which no user-facing surface displays — explicit `cuda`/`rocm` failures, by contrast, do show their reason. |
| `cpu` | Never touches the GPU; the probe is never run. |
| `cuda`, `rocm` | Require that GPU stack by name: "no device", "library unusable" or a CTranslate2 refusal raises `CudaUnavailable` with the missing piece and the exact remedy **before any model build** — the file job exits 1 with no traceback and no transcript, never a false success. |

Model load and inference for the default backend run in a separate process
(`listn/inference_worker.py`); the parent writes library files only after the
child answers with a complete result. A native SIGSEGV/SIGABRT kills only that
child: under `auto` the parent restarts it pinned to CPU exactly once and
appends `…; running on CPU` to the status line, under explicit
`cuda`/`rocm` it raises the actionable error with no CPU retry, and under `cpu`
a crash is an error because there is nothing to fall back to.
Requests to that child carry a silence-based deadline — a long one for the
model load (a first-run model download can take minutes) and a shorter one
for transcription: the clock starts before the request is written, so a
child that goes silent — whether it stops reading the request or stops
replying to it — is stopped and reported as one clean error rather than
wedging the caller, and closing never queues behind a stuck request. The
worker may confirm the device decision but never widen it: a child that
claims a GPU the parent did not choose has that claim dropped (device,
status line and device message alike) — under the default `auto` the
claim must also match this machine's own capability probe.
`tests/test_isolation.py` simulates each case, including those bounds and
claims.

`compute_type`: a value of `int8` — the settings default, indistinguishable
from an explicit set — resolves to `float16` on a GPU build and stays `int8`
on CPU; any other user-set value is passed through untouched on every device.

## Detect the actual machine first

Run these on the **CachyOS laptop itself**, in a terminal session with access to the host PCI devices (not this research container). Commands are observational; do not install drivers/packages as part of detection.

```sh
# PCI vendor/model and kernel driver in use; include all display/3D controllers
lspci -nnk | grep -A3 -Ei 'VGA|3D|Display'

# NVIDIA: present, driver loaded, visible GPU and driver-reported status
nvidia-smi

# AMD: ROCm runtime sees supported GPU(s) and agents (command may be absent)
rocminfo

# Useful supporting evidence
ls -l /dev/dri /dev/kfd 2>/dev/null
```

Interpretation:

- `lspci -nnk` lists PCI devices and the kernel driver in use, where visible; an NVIDIA/AMD entry proves only that the PCI device is visible, not that its user-space runtime, permissions, model, or CTranslate2 backend works.[9]
- A successful `nvidia-smi` lists NVIDIA GPUs and reports driver-provided device information. An absent command means only that the utility is unavailable; errors can also mean driver/device access problems. Match installed driver/toolkit to NVIDIA’s official CUDA compatibility/installation docs and faster-whisper’s cuBLAS/cuDNN requirements.[1][10]
- `rocminfo` enumerating a GPU/agent is evidence that ROCm runtime sees it, not proof the particular CTranslate2 wheel or model works. Check GPU and Linux OS/kernel compatibility in AMD’s current matrices; unsupported devices/OS versions can fail even if a PCI device is visible.[6][7][11]
- If commands are absent, device files are missing, or output is ambiguous, stop and resolve hardware/driver visibility first. Do not interpret tool absence as “no GPU.” An integrated GPU is not automatically a supported compute backend.

## Installation recipes (only after GPU identification and explicit approval)

These commands install software and can fetch packages from the internet. They are documentation, **not authorization to execute them**. Use an isolated project virtual environment; do not change Listn’s dependency files as part of this investigation. Model fetching is separate: use a local model directory already present and do not let benchmarking implicitly download a model.

### NVIDIA (upstream faster-whisper recipe, Linux)

Prerequisites: supported NVIDIA GPU/driver; CUDA 12 libraries; cuDNN 9 for CUDA 12. The faster-whisper README gives this pip path and says to set `LD_LIBRARY_PATH` before Python:[1]

```sh
# In a disposable venv, only after checking NVIDIA's distro/driver instructions:
python -m pip install 'faster-whisper>=1.1,<2' nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'

# Follow the project's exact documented path construction, with the venv's Python:
export LD_LIBRARY_PATH="$(python -c 'import os, nvidia.cublas, nvidia.cudnn; print(os.path.dirname(nvidia.cublas.__path__[0]) + "/cublas/lib:" + os.path.dirname(nvidia.cudnn.__path__[0]) + "/cudnn/lib")')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Install/configure the NVIDIA kernel driver and CUDA runtime using the official NVIDIA instructions for the actual distribution; do not install an arbitrary toolkit bundle just because it is newer. Check `nvidia-smi`, and then query CTranslate2’s runtime-supported types. Current upstream compatibility note: CUDA 11/cuDNN 8 requires CTranslate2 3.24.0 workaround; CUDA 12/cuDNN 8 CTranslate2 4.4.0; latest guidance uses CUDA 12/cuDNN 9.[1] These downgrades can conflict with project constraints and are not recommended as a Listn change.

### AMD (ROCm/HIP)

1. Identify exact GPU, OS, kernel and ROCm version. Verify all against the **current AMD** compatibility matrices first.[6][7]
2. Confirm the corresponding CTranslate2 release’s ROCm Linux wheel asset and its build/runtime requirements (for example, v4.8.4 publishes a Linux wheel archive).[5] Use AMD’s / CTranslate2’s official current ROCm wheel instructions for that exact release; the standard `pip install ctranslate2` route is not evidence of a ROCm build.[2]
3. Install only matched ROCm runtime packages for the supported OS/GPU, in a disposable venv. Confirm `rocminfo` and CTranslate2’s ROCm device/runtime detection, then check supported compute types on that build.
4. Do not use the AMD 2024 blog’s ROCm 6.0+/PyTorch 2.1+/Docker-fork steps as a current turnkey faster-whisper installation; that article is a separate dated example.[8]

**No verified one-line install command for the current official ROCm-wheel + faster-whisper combination is provided here:** the release API confirms the wheel archive exists but the retrieved official release notes do not give a detailed consumer install sequence/runtime matrix. Do not guess the wheel filename or claim a compatible ROCm ABI.[5]

### Vulkan

No installation steps: CTranslate2/faster-whisper’s documented backend/device API does not establish Vulkan support.[2][4]

## Benchmark plan (run only on the laptop)

Goal: compare the same faster-whisper/CTranslate2 build, model files, audio input, decoding options and output correctness on CPU int8 vs one verified GPU mode.[unverified] A benchmark result is not a general speed claim.[unverified]

### Controls

1. Record machine model, OS/kernel, CPU model/core count, exact GPU, RAM/VRAM, driver/runtime versions, Python, faster-whisper and CTranslate2 versions. Save `lspci -nnk`, `nvidia-smi` or `rocminfo` output as environment evidence (do not include audio, transcript, tokens, user paths or other private data in public logs).
2. Choose one **already-local** fixed Whisper model directory and one fixed audio file with known duration/sample properties. Record model identifier/hash and audio SHA-256 in the private benchmark record; do not publish the recording/hash if it can identify private input. No network/model download during runs.
3. Use exactly the same audio preprocessing, language, beam size (e.g. 5), VAD/word timestamps, batch size, chunking, and faster-whisper/CTranslate2/PyAV versions. For a controlled comparison use `batch_size=1`, `beam_size=5`, fixed language, VAD off unless the product uses it; state any production-specific settings. Compare generated text/segments for correctness, but keep transcript private.
4. CPU leg: `device="cpu", compute_type="int8"`. GPU leg: NVIDIA `device="cuda", compute_type="float16"` or another type the actual runtime reports; AMD only after that ROCm build’s API/device and accepted type are independently verified. Never compare types that silently fall back without recording effective runtime behavior.
5. Separate cold process/model load from transcription. Each run fully consumes the lazy `segments` generator. Warm up once, then run each leg at least five times in alternating order (CPU, GPU, GPU, CPU, …) to reduce order/thermal bias.[unverified] Keep system load and power mode stable; do not run concurrent jobs.

### Metrics and method

- **Wall time:** use `time.perf_counter()` around model initialization and, separately, the fully consumed transcription call; report both cold total and steady-state transcription. Repeat at least five times and report the median, range, audio duration, and real-time factor.[unverified]
- **Peak host RSS:** run each trial in a fresh child process and sample `/proc/<pid>/status` `VmHWM` (peak resident set in KiB) after process exit, or use `/usr/bin/time -v` and capture “Maximum resident set size”. Report units and method; RSS does not include GPU VRAM.[12]
- **Peak GPU memory:** record separately, not as process RSS. NVIDIA: sample `nvidia-smi` memory utilization/used memory during each isolated run; AMD: use matching ROCm monitoring tooling. Note sampling granularity and baseline/other processes. If precise peak is unavailable, report unavailable, do not substitute host RSS.
- **Outcome:** include errors/fallbacks, model-load time, each transcription time, RSS/VRAM readings, output segment count and a private correctness comparison. Redact transcript text/audio and avoid logging user/private data.
- **Speedup calculation:** only after successful complete matched runs, calculate `median_cpu_seconds / median_gpu_seconds`, label it as this machine/model/audio/settings, and include variability. A GPU run that fails, silently selects CPU, changes quality/options, or produces no evidence is not a successful result and must not be used to claim acceleration.

Illustrative inference configuration (not a ready-to-run benchmark script; load a local model path and use identical settings on both runs):

```python
from faster_whisper import WhisperModel

model = WhisperModel(LOCAL_MODEL_DIR, device=DEVICE, compute_type=TYPE)
segments, info = model.transcribe(LOCAL_AUDIO_PATH, language="en", beam_size=5, vad_filter=False)
segments = list(segments)  # transcription is lazy; consume all work
```

## Project status / follow-up

The architecture document correctly labels Linux GPU acceleration and live transcription performance **unverified**. Keep that status until GPU identity, runtime support, successful end-to-end transcription and controlled measurements are recorded on the actual CachyOS target. No measured speedup is asserted in this note.

## Sources

[1] https://github.com/SYSTRAN/faster-whisper — faster-whisper README
[2] https://opennmt.net/CTranslate2/installation.html — CTranslate2 Installation
[3] https://opennmt.net/CTranslate2/quantization.html — CTranslate2 Quantization
[4] https://opennmt.net/CTranslate2/python/ctranslate2.get_supported_compute_types.html — CTranslate2 get_supported_compute_types
[5] https://api.github.com/repos/OpenNMT/CTranslate2/releases/tags/v4.8.4 — CTranslate2 v4.8.4 release assets
[6] https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.0/reference/system-requirements.html — ROCm 7.0 Linux System Requirements
[7] https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibility.html — ROCm Radeon compatibility matrices
[8] https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html — AMD ROCm CTranslate2 guide
[9] https://man7.org/linux/man-pages/man8/lspci.8.html — lspci manual page
[10] https://docs.nvidia.com/deploy/nvidia-smi/index.html — NVIDIA System Management Interface
[11] https://github.com/ROCm/rocminfo — rocminfo project (moved)
[12] https://man7.org/linux/man-pages/man5/proc_pid_status.5.html — /proc pid status manual
