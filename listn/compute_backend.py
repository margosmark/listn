"""Vendor-neutral compute backends with honest capability reporting.

listn transcribes on three compute backends without assuming a vendor:

* ``cpu`` — always the safe fallback (int8 quantisation on CPU).
* ``cuda`` — NVIDIA: CTranslate2 must see a CUDA device *and* the CUDA 12
  runtime libraries it dlopens by soname (``libcublas.so.12``, cuDNN) must
  actually load. A visible GPU with missing libraries is NOT usable.
* ``rocm`` — AMD: CTranslate2 must be a ROCm/HIP build (its extension has
  the HIP runtime linked in) with a device visible through it. A CUDA-built
  CTranslate2 on a machine with an AMD GPU is NOT a ROCm backend.

What this machine can really run is answered by a child-process probe
(:func:`probe_capabilities`): loading a native GPU library can end a
process without anything Python can catch, so every native check runs in a
child that only ever reports one line of JSON back. Every unusable backend
is reported as the missing piece plus an exact remedy — never a bare error
string.

Selection policy (:func:`resolve`) is truthful in both directions: an
explicitly requested backend that cannot run fails loudly with its
remedy; ``auto`` uses a GPU only when the probe says it works and falls
back to CPU saying so in the output; CPU is always the answer on machines
with no usable GPU.

Local-first per AGENTS.md: the probe reads local device nodes, PCI vendor
ids, ``/proc/self/maps`` and library sonames. It never touches the
network, never downloads models, and never sees audio or transcripts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from listn.config import DEFAULT_COMPUTE_TYPE

# Device names this layer understands. "auto" means: GPU only when the
# probe says it works, CPU otherwise.
KNOWN_DEVICES = ("auto", "cpu", "cuda", "rocm")

# Shared libraries CTranslate2 dlopens by soname when a model is built for
# CUDA. On a machine with an NVIDIA driver but without the CUDA runtime
# libraries the load fails with "Library libcublas.so.12 is not found or
# cannot be loaded" — the failure seen on the CachyOS laptop. Versions are
# pinned in requirements-all.txt.
CUDA_RUNTIME_LIBS = ("libcublas.so.12", "libcudnn.so.9")

# HIP runtime sonames, newest first. Probing for them is capability
# detection, not a vendor assumption: a load failure simply means ROCm is
# not usable here.
HIP_RUNTIME_LIBS = ("libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so")


class DeviceUnavailable(RuntimeError):
    """An explicitly requested compute backend cannot run on this machine.

    Carries the missing piece and the exact remedy in the message, so the
    caller never has to translate a bare error string.
    """


class CudaUnavailable(DeviceUnavailable):
    """A GPU backend was asked for by name and this machine cannot run it.

    Raised before any model is built, so the user gets one readable line
    (with the remedy) instead of a native failure half-way through
    loading. Despite the historical name this covers ``cuda`` *and*
    ``rocm`` requests — callers treat it as "the named GPU backend is
    unavailable, never fall back silently".
    """


@dataclass(frozen=True)
class Capability:
    """One compute backend on this machine: usable, or missing + remedy."""

    name: str
    usable: bool
    missing: str | None = None
    remedy: str | None = None
    detail: str = ""

    def explain(self) -> str:
        """The human answer for an unusable backend (missing piece + fix)."""
        if self.usable:
            return self.detail or "available"
        parts = [p for p in (self.missing, self.remedy) if p]
        return "; ".join(parts) if parts else "unavailable (no detail reported)"


@dataclass(frozen=True)
class Plan:
    """What :func:`resolve` decided: the device to build with, and why."""

    device: str            # string handed to the backend (cpu / auto / cuda)
    note: str | None = None  # truthful deviation note, surfaced in output
    gpu_active: bool = False  # a GPU should (or will) run the model
    kind: str | None = None   # "cuda" | "rocm" | None — which GPU backend


# --- the probe -------------------------------------------------------------
# Runs in a child process: importing CTranslate2, dlopen'ing a CUDA/HIP
# library or querying the driver can terminate the process without ever
# raising a Python exception. The child takes that risk; the parent only
# reads one JSON line back. ``--full`` additionally proves the
# faster-whisper backend imports (report path only — the hot path skips it).

PROBE_SCRIPT = """
import ctypes, json, sys

argv = sys.argv[1:]
full = "--full" in argv
sonames = [a for a in argv if a != "--full"]
result = {
    "ct2": {"importable": False, "version": None, "error": None},
    "devices": 0,
    "device_error": None,
    "sonames": sonames,
    "missing": [],
    "hip": {"linked": [], "present": [], "error": None},
    "amd_gpu": False,
    "faster_whisper": None,
}
try:
    import ctranslate2
    result["ct2"]["importable"] = True
    result["ct2"]["version"] = getattr(ctranslate2, "__version__", None)
except Exception as exc:
    result["ct2"]["error"] = "%s: %s" % (type(exc).__name__, exc)
else:
    try:
        result["devices"] = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        result["device_error"] = str(exc)
    # What did importing CTranslate2 map into THIS process? A ROCm build
    # links the HIP runtime; a CUDA build never does. Reading
    # /proc/self/maps distinguishes the two builds without guessing from
    # install paths.
    try:
        maps = open("/proc/self/maps").read()
    except OSError:
        maps = ""
    for line in maps.splitlines():
        parts = line.split()
        path = parts[-1] if parts else ""
        if path.startswith("/") and any(
                key in path for key in ("amdhip", "rocblas", "hipblas", "hiprtc", "/rocm/")):
            if path not in result["hip"]["linked"]:
                result["hip"]["linked"].append(path)
for soname in sonames:
    try:
        ctypes.CDLL(soname)
    except OSError as exc:
        result["missing"].append("%s: %s" % (soname, exc))
# The HIP runtime may be installed even when this CTranslate2 is a CUDA
# build; the rocm capability needs to tell those two states apart.
for soname in ("libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"):
    try:
        ctypes.CDLL(soname)
        result["hip"]["present"].append(soname)
        break
    except OSError:
        pass
    except BaseException as exc:
        result["hip"]["error"] = "%s: %s" % (soname, exc)
        break
# Are the missing sonames present in the environment (the pip nvidia
# packages) even though the loader cannot see them? The loader does not
# search site-packages, so "installed" and "loadable" are different
# states with different remedies (LD_LIBRARY_PATH vs pip install).
result["on_disk"] = []
if result["missing"]:
    try:
        import glob as _g, os as _o, sys as _s
        names = [m.split(":", 1)[0] for m in result["missing"]]
        dirs = []
        for entry in _s.path:
            if entry.endswith("site-packages"):
                dirs.extend(_g.glob(_o.join(entry, "nvidia", "*", "lib")))
        result["on_disk"] = sorted(
            {n for d in dirs for n in names if _o.path.exists(_o.join(d, n))})
    except Exception:
        pass
# An AMD GPU: PCI vendor 0x1002, or the ROCm compute node device.
try:
    import glob, os
    for vendor in glob.glob("/sys/class/drm/card*/device/vendor"):
        try:
            if open(vendor).read().strip().lower() == "0x1002":
                result["amd_gpu"] = True
                break
        except OSError:
            pass
    if os.path.exists("/dev/kfd"):
        result["amd_gpu"] = True
except Exception:
    pass
if full:
    try:
        import faster_whisper  # noqa: F401
        result["faster_whisper"] = True
    except Exception as exc:
        result["faster_whisper"] = False
        result["faster_whisper_error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(result))
"""


def run_probe(script: str | None = None, sonames: Sequence[str] = (),
              timeout: float = 30.0, full: bool = False) -> tuple[dict | None, str | None]:
    """Run the native checks out-of-process; ``(None, reason)`` = failed.

    A child that dies by signal, times out, or prints nothing becomes a
    reason string — never a crash in this process.
    """
    if script is None:
        script = PROBE_SCRIPT
    argv = [sys.executable, "-c", script]
    if full:
        argv.append("--full")
    argv.extend(str(s) for s in sonames)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"the capability probe timed out after {timeout:g}s"
    if done.returncode != 0:
        how = (f"signal {-done.returncode}" if done.returncode < 0
               else f"exit code {done.returncode}")
        return None, f"the capability probe terminated with {how}"
    try:
        return json.loads(done.stdout.strip().splitlines()[-1]), None
    except (IndexError, ValueError):
        return None, "the capability probe returned no usable answer"


# --- interpretation: raw probe JSON -> per-backend capability ---------------

def _ct2_importable(raw: Mapping) -> bool:
    ct2 = raw.get("ct2") or {}
    return bool(ct2.get("importable"))


def _cpu_capability(raw: Mapping | None) -> Capability:
    if raw is None:
        return Capability("cpu", True, None, None, "the fallback backend")
    fw = raw.get("faster_whisper")
    if fw is False:
        return Capability(
            "cpu", False,
            "the faster-whisper backend is not installed",
            "pip install faster-whisper (listn's default transcription backend)",
            str(raw.get("faster_whisper_error") or ""))
    if fw is True:
        return Capability("cpu", True, None, None, "faster-whisper installed; int8 on CPU")
    return Capability("cpu", True, None, None, "always available as the fallback")


def _cuda_capability(raw: Mapping) -> Capability:
    if not _ct2_importable(raw):
        return Capability(
            "cuda", False,
            "CTranslate2 is not installed (the faster-whisper backend provides it)",
            "pip install faster-whisper",
            str((raw.get("ct2") or {}).get("error") or ""))
    hip = raw.get("hip") or {}
    if hip.get("linked"):
        return Capability(
            "cuda", False,
            "this CTranslate2 build is a ROCm/HIP build, so CUDA libraries cannot satisfy it",
            "use --compute-device rocm (or auto); CUDA needs a CUDA-built CTranslate2 "
            "(see docs/GPU_LINUX.md)",
            "linked: " + ", ".join(hip["linked"]))
    device_error = raw.get("device_error")
    if device_error:
        return Capability(
            "cuda", False,
            f"CTranslate2 cannot query the NVIDIA driver ({device_error})",
            "check that `nvidia-smi` runs for this user and the driver matches CUDA 12 "
            "(docs/GPU_LINUX.md)",
            "")
    devices = int(raw.get("devices") or 0)
    if devices <= 0:
        return Capability(
            "cuda", False,
            "CTranslate2 sees 0 CUDA devices (no NVIDIA driver, or none exposed to this process)",
            "check the driver with `nvidia-smi` and `lspci -nnk | grep -A3 -i nvidia`; "
            "on a machine without an NVIDIA GPU use --compute-device cpu or auto",
            "")
    missing = list(raw.get("missing") or [])
    if missing:
        names = {m.split(":", 1)[0] for m in missing}
        on_disk = set(raw.get("on_disk") or [])
        if names and names <= on_disk:
            return Capability(
                "cuda", False,
                "the CUDA runtime libraries are installed (pip) but not on the "
                "dynamic-loader path: " + "; ".join(missing),
                "set LD_LIBRARY_PATH to the nvidia pip packages' lib dirs — the "
                "exact command is documented in docs/GPU_LINUX.md "
                "(faster-whisper README)",
                f"{devices} CUDA device(s) visible")
        return Capability(
            "cuda", False,
            "; ".join(missing),
            "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
            "nvidia-cuda-runtime-cu12 (versions pinned in requirements-all.txt), "
            "then set LD_LIBRARY_PATH per docs/GPU_LINUX.md",
            f"{devices} CUDA device(s) visible")
    return Capability(
        "cuda", True, None, None,
        f"{devices} CUDA device(s) visible; the CUDA 12 runtime libraries load")


def _rocm_capability(raw: Mapping) -> Capability:
    if not _ct2_importable(raw):
        return Capability(
            "rocm", False,
            "CTranslate2 is not installed (the faster-whisper backend provides it)",
            "pip install faster-whisper",
            str((raw.get("ct2") or {}).get("error") or ""))
    hip = raw.get("hip") or {}
    amd_gpu = bool(raw.get("amd_gpu"))
    if hip.get("linked"):
        device_error = raw.get("device_error")
        devices = int(raw.get("devices") or 0)
        if device_error:
            return Capability(
                "rocm", False,
                f"the ROCm/HIP build cannot query a device ({device_error})",
                "check `rocminfo` and AMD's current GPU/ROCm compatibility matrix "
                "(docs/GPU_LINUX.md)",
                "linked: " + ", ".join(hip["linked"]))
        if devices <= 0:
            return Capability(
                "rocm", False,
                "the ROCm runtime loaded but CTranslate2 sees 0 devices",
                "check `rocminfo` enumerates a GPU and AMD's current GPU/ROCm "
                "compatibility matrix (docs/GPU_LINUX.md)",
                "linked: " + ", ".join(hip["linked"]))
        return Capability(
            "rocm", True, None, None,
            f"CTranslate2 is a ROCm/HIP build and sees {devices} device(s)")
    if hip.get("present"):
        return Capability(
            "rocm", False,
            "this CTranslate2 is a CUDA build, not a ROCm build (the HIP runtime is "
            "installed but CTranslate2 is not linked against it)",
            "install the ROCm-flavoured CTranslate2 wheel from the CTranslate2 releases "
            "page (rocm-python-wheels archive); see docs/GPU_LINUX.md",
            "HIP runtime present: " + ", ".join(hip["present"]))
    if amd_gpu:
        return Capability(
            "rocm", False,
            "an AMD GPU is visible but neither the ROCm runtime nor a ROCm-built "
            "CTranslate2 was found",
            "install ROCm per AMD's current system-requirements matrix, then the ROCm "
            "CTranslate2 wheel (docs/GPU_LINUX.md)",
            "")
    return Capability(
        "rocm", False,
        "no AMD GPU and no ROCm runtime are visible on this machine",
        "use --compute-device cpu or auto; ROCm needs an AMD GPU plus a ROCm-built "
        "CTranslate2 (docs/GPU_LINUX.md)",
        "")


def interpret(raw: Mapping) -> dict[str, Capability]:
    """Turn one probe's JSON into per-backend capabilities (pure, testable)."""
    return {
        "cpu": _cpu_capability(raw),
        "cuda": _cuda_capability(raw),
        "rocm": _rocm_capability(raw),
    }


def probe_capabilities(timeout: float = 30.0) -> dict[str, Capability]:
    """Probe this machine and report which compute backends really work.

    Returns ``{name: Capability}`` for cpu, cuda and rocm. Every unusable
    backend carries the missing piece and the exact remedy; a probe that
    fails outright is itself reported as a missing piece with a remedy,
    never as a bare error string.
    """
    raw, error = run_probe(sonames=CUDA_RUNTIME_LIBS, timeout=timeout, full=True)
    if raw is None:
        why = error or "the probe returned no usable answer"
        fail = Capability(
            "probe", False,
            f"the machine capability probe failed ({why})",
            "check GPU drivers and run `lspci -nnk` / `rocminfo`; see docs/GPU_LINUX.md",
            "")
        caps = {name: Capability(name, False, fail.missing, fail.remedy, "")
                for name in ("cuda", "rocm")}
        caps["cpu"] = _cpu_capability(None)
        return caps
    return interpret(raw)


def format_report(report: Mapping[str, Capability]) -> str:
    """Render a capability report as readable lines (missing + remedy included)."""
    lines = ["compute backends on this machine:"]
    for name in ("cpu", "cuda", "rocm"):
        cap = report.get(name)
        if cap is None:
            continue
        if cap.usable:
            lines.append(f"  {name:<5} usable — {cap.detail or 'available'}")
        else:
            lines.append(f"  {name:<5} NOT usable — missing: {cap.missing}")
            lines.append(f"        remedy: {cap.remedy}")
    return "\n".join(lines)


# --- ROCm status for selection (cached: hardware does not change) ----------

_rocm_cache: Capability | None = None


def clear_probe_cache() -> None:
    """Forget the cached ROCm status (tests, and after installing ROCm)."""
    global _rocm_cache
    _rocm_cache = None


def rocm_capability() -> Capability:
    """The ROCm capability of this machine, probed once per process."""
    global _rocm_cache
    if _rocm_cache is None:
        raw, error = run_probe(sonames=(), timeout=30.0)
        _rocm_cache = (_rocm_capability(raw) if raw is not None else
                       Capability(
                           "rocm", False,
                           f"the ROCm capability probe failed ({error or 'no usable answer'})",
                           "check GPU drivers and `rocminfo`; see docs/GPU_LINUX.md",
                           ""))
    return _rocm_cache


# --- selection policy -------------------------------------------------------

def resolve(requested: str | None, *, cuda_count, cuda_ready,
            rocm=None) -> Plan:
    """Turn a ``compute_device`` request into a truthful build plan.

    ``cuda_count``/``cuda_ready``/``rocm`` are the probe seams (injected so
    callers can pin them; production passes the real child-process probes).

    * ``cpu`` — never probes anything.
    * ``cuda`` / ``rocm`` by name — fail loudly with the missing piece and
      the remedy when the backend cannot run here; no silent fallback.
    * ``auto`` — GPU only when the probe says it works (CUDA first, then
      ROCm), CPU otherwise; a visible-but-broken GPU pins CPU with the
      reason attached as the note so the output says so.
    """
    want = (requested or "auto").strip().lower() or "auto"
    if rocm is None:
        rocm = rocm_capability  # late-bound: tests can pin this seam
    if want not in KNOWN_DEVICES:
        raise DeviceUnavailable(
            f"compute_device={want!r} is not a known compute backend. "
            "Use cpu, auto, cuda or rocm.")
    if want == "cpu":
        return Plan("cpu")
    if sys.platform == "darwin":
        # macOS has no CUDA or ROCm for this stack; never probe here.
        if want == "cuda":
            raise CudaUnavailable(
                "compute_device=cuda was requested, but macOS has no CUDA. "
                "Use compute_device=cpu, or leave it on auto.")
        if want == "rocm":
            raise CudaUnavailable(
                "compute_device=rocm was requested, but macOS has no ROCm runtime "
                "for this stack. Use compute_device=cpu, or leave it on auto.")
        return Plan("auto")
    if want == "cuda":
        if cuda_count() <= 0:
            rocm_cap = rocm()
            if rocm_cap.usable:
                raise CudaUnavailable(
                    "compute_device=cuda was requested, but "
                    f"{rocm_cap.detail}. Use --compute-device rocm (or auto), "
                    "or run on CPU with --compute-device cpu.")
            raise CudaUnavailable(
                "compute_device=cuda was requested, but CTranslate2 sees no CUDA "
                "device (no NVIDIA driver, or none exposed to this process). "
                "Check the driver with `nvidia-smi`; run on CPU with "
                "--compute-device cpu, or leave it on auto.")
        ready, detail = cuda_ready()
        if not ready:
            rocm_cap = rocm()
            if rocm_cap.usable:
                raise CudaUnavailable(
                    "compute_device=cuda was requested, but "
                    f"{rocm_cap.detail}. Use --compute-device rocm (or auto), "
                    "or run on CPU with --compute-device cpu.")
            raise CudaUnavailable(
                f"compute_device=cuda was requested, but CUDA is visible but the "
                f"CUDA runtime cannot be used ({detail}). Install the CUDA 12 "
                "libraries (pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
                "nvidia-cuda-runtime-cu12; docs/GPU_LINUX.md), or run on CPU with "
                "--compute-device cpu.")
        return Plan("cuda", None, True, "cuda")
    if want == "rocm":
        cap = rocm()
        if not cap.usable:
            raise CudaUnavailable(
                f"compute_device=rocm was requested, but {cap.missing}. {cap.remedy}")
        return Plan("cuda", "rocm (CTranslate2 HIP build)", True, "rocm")
    # auto: CUDA first, ROCm second, CPU otherwise.
    if cuda_count() > 0:
        ready, detail = cuda_ready()
        if ready:
            return Plan("auto", None, True, "cuda")
        rocm_cap = rocm()
        if rocm_cap.usable:
            # A ROCm build can count HIP devices as "cuda" while libcublas
            # can never load: the GPU that works here is the ROCm one.
            return Plan("auto", "rocm (CTranslate2 HIP build)", True, "rocm")
        reason = f"CUDA is visible but the CUDA runtime cannot be used ({detail})"
        return Plan("cpu", f"{reason}; see docs/GPU_LINUX.md", False, None)
    rocm_cap = rocm()
    if rocm_cap.usable:
        return Plan("auto", "rocm (CTranslate2 HIP build)", True, "rocm")
    return Plan("auto")


# --- per-backend compute type ----------------------------------------------

def compute_type_for(compute_type: str | None, device: str,
                     *, gpu_active: bool = False) -> str:
    """int8 on CPU, float16 on GPU — unless the user chose a type.

    ``compute_type`` equal to the settings default (``int8``) means "left
    alone", so the mapping applies: a GPU build gets float16 (faster-whisper's
    GPU default, output semantics preserved per the backend contract), a CPU
    build keeps int8. Any other value is an explicit user choice and is
    passed through untouched on every device.
    """
    chosen = compute_type or DEFAULT_COMPUTE_TYPE
    if chosen != DEFAULT_COMPUTE_TYPE:
        return chosen
    if device == "cuda" or (device == "auto" and gpu_active):
        return "float16"
    return chosen
