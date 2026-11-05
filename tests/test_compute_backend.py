"""Vendor-neutral compute backends: capability report, policy, compute types.

Two layers are covered here:

* ``listn.compute_backend`` — the capability probe's *interpretation* runs
  against simulated probe results for the environments that matter (no CUDA
  and no ROCm, the CachyOS laptop's visible-GPU/missing-libraries state, an
  AMD machine without ROCm, a ROCm-built CTranslate2), plus one real probe
  of this host and the selection/compute-type policy against injected
  seams. No GPU hardware is required and nothing touches the network.
* ``listn.transcriber`` — a real isolated worker child that crashes on
  purpose under ``compute_device=rocm`` (native-crash isolation inherited
  from t_7c87bd2c applies to every backend).

Simulated probe JSON is the evidence for "no CUDA / no ROCm" machines; the
real-host probe asserts honesty (missing piece + remedy) rather than a
host-specific verdict, so the suite stays green on any machine.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from listn import compute_backend as cb
from listn import transcriber as tr
from listn.transcriber import CudaUnavailable

FACTORIES = "tests._crash_backends"


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts and ends with empty caches (workers, probes)."""
    tr._close_cached()
    tr._cache.clear()
    cb.clear_probe_cache()
    yield
    tr._close_cached()
    tr._cache.clear()
    cb.clear_probe_cache()


# --- simulated probe results ----------------------------------------------

def _raw(**over):
    """A probe result for a plain CPU-only machine, with overrides."""
    base = {
        "ct2": {"importable": True, "version": "4.8.3", "error": None},
        "devices": 0,
        "device_error": None,
        "sonames": ["libcublas.so.12", "libcudnn.so.9"],
        "missing": [],
        "on_disk": [],
        "hip": {"linked": [], "present": [], "error": None},
        "amd_gpu": False,
        "faster_whisper": True,
    }
    base.update(over)
    return base


NO_GPU = _raw()
CUBLAS_MISSING = "libcublas.so.12: cannot open shared object file: No such file or directory"
CACHYOS = _raw(devices=1, missing=[CUBLAS_MISSING])
CACHYOS_INSTALLED = _raw(devices=1, missing=[CUBLAS_MISSING],
                         on_disk=["libcublas.so.12"])
DRIVER_BROKEN = _raw(device_error="libcuda.so.1: cannot open shared object file")
ROCM_BUILD = _raw(devices=2, missing=[CUBLAS_MISSING],
                  hip={"linked": ["/opt/rocm/lib/libamdhip64.so.6"],
                       "present": ["libamdhip64.so.6"], "error": None})
AMD_NO_ROCM = _raw(amd_gpu=True)
NO_CT2 = _raw(ct2={"importable": False, "version": None,
                   "error": "ModuleNotFoundError: No module named 'ctranslate2'"},
              faster_whisper=False,
              faster_whisper_error="ModuleNotFoundError: No module named 'faster_whisper'")

ALL_SIMULATED = {
    "no_gpu": NO_GPU,
    "cachyos_missing_lib": CACHYOS,
    "libs_installed_not_loadable": CACHYOS_INSTALLED,
    "driver_broken": DRIVER_BROKEN,
    "rocm_build": ROCM_BUILD,
    "amd_without_rocm": AMD_NO_ROCM,
    "no_ct2": NO_CT2,
}


def _assert_honest(caps: dict) -> None:
    """Every backend says either 'usable' or missing piece + exact remedy."""
    assert set(caps) == {"cpu", "cuda", "rocm"}
    for name, cap in caps.items():
        assert cap.name == name
        if cap.usable:
            assert cap.missing is None and cap.remedy is None, (name, cap)
        else:
            assert cap.missing, name
            assert cap.remedy, name


def _missing(caps: dict, name: str) -> str:
    value = caps[name].missing
    assert value, f"{name} must report its missing piece"
    return value


def _remedy(caps: dict, name: str) -> str:
    value = caps[name].remedy
    assert value, f"{name} must report an actionable remedy"
    return value


# --- capability interpretation (simulated environments) ---------------------

def test_no_cuda_no_rocm_machine_reports_cpu_and_actionable_gpu_remedies():
    caps = cb.interpret(NO_GPU)
    _assert_honest(caps)
    assert caps["cpu"].usable
    assert "0 CUDA devices" in _missing(caps, "cuda")
    assert "--compute-device cpu" in _remedy(caps, "cuda")
    assert "no AMD GPU" in _missing(caps, "rocm")
    assert "--compute-device cpu" in _remedy(caps, "rocm")


def test_cachyos_visible_gpu_missing_lib_names_the_package_and_file():
    """The observed failure: a real GPU, a missing libcublas.so.12."""
    caps = cb.interpret(CACHYOS)
    _assert_honest(caps)
    assert not caps["cuda"].usable
    assert "libcublas.so.12" in _missing(caps, "cuda")
    assert "nvidia-cublas-cu12" in _remedy(caps, "cuda")
    assert "requirements-all.txt" in _remedy(caps, "cuda")


def test_installed_but_not_loadable_names_the_loader_path_not_a_pip_install():
    """The pip package is present but the loader cannot see it: the missing
    piece is LD_LIBRARY_PATH, not another install."""
    caps = cb.interpret(CACHYOS_INSTALLED)
    assert not caps["cuda"].usable
    assert "not on the dynamic-loader path" in _missing(caps, "cuda")
    assert "LD_LIBRARY_PATH" in _remedy(caps, "cuda")
    assert "docs/GPU_LINUX.md" in _remedy(caps, "cuda")


def test_driver_query_failure_points_at_nvidia_smi():
    caps = cb.interpret(DRIVER_BROKEN)
    assert not caps["cuda"].usable
    assert "cannot query the NVIDIA driver" in _missing(caps, "cuda")
    assert "nvidia-smi" in _remedy(caps, "cuda")


def test_rocm_build_is_usable_and_cuda_is_honestly_not():
    caps = cb.interpret(ROCM_BUILD)
    _assert_honest(caps)
    assert caps["rocm"].usable
    assert "ROCm/HIP" in caps["rocm"].detail and "2" in caps["rocm"].detail
    # Same machine, same probe: CUDA cannot be satisfied by a HIP build.
    assert not caps["cuda"].usable
    assert "ROCm/HIP build" in _missing(caps, "cuda")
    assert "--compute-device rocm" in _remedy(caps, "cuda")


def test_amd_gpu_without_rocm_names_runtime_and_wheel_install():
    caps = cb.interpret(AMD_NO_ROCM)
    assert not caps["rocm"].usable
    assert "AMD GPU" in _missing(caps, "rocm")
    assert "ROCm" in _remedy(caps, "rocm")
    assert "docs/GPU_LINUX.md" in _remedy(caps, "rocm")


def test_missing_ctranslate2_names_faster_whisper_for_every_backend():
    caps = cb.interpret(NO_CT2)
    _assert_honest(caps)
    assert not caps["cpu"].usable
    assert "faster-whisper" in _remedy(caps, "cpu")
    assert "faster-whisper" in _remedy(caps, "cuda")
    assert "faster-whisper" in _remedy(caps, "rocm")


def test_every_simulated_environment_is_reported_honestly():
    for label, raw in ALL_SIMULATED.items():
        _assert_honest(cb.interpret(raw))


# --- the real probe on this host -------------------------------------------

def test_real_probe_on_this_host_reports_never_a_bare_error():
    caps = cb.probe_capabilities()
    _assert_honest(caps)
    # The dev environment ships faster-whisper, so CPU is really usable here.
    assert caps["cpu"].usable


def test_probe_failure_itself_becomes_missing_plus_remedy(monkeypatch):
    monkeypatch.setattr(cb, "run_probe",
                        lambda **kw: (None, "the capability probe timed out after 1s"))
    caps = cb.probe_capabilities()
    _assert_honest(caps)
    # A failed probe does not condemn the CPU: it never needed one.
    assert caps["cpu"].usable
    assert "timed out" in _missing(caps, "cuda")
    assert "docs/GPU_LINUX.md" in _remedy(caps, "cuda")
    assert "timed out" in _missing(caps, "rocm")


def test_report_format_includes_missing_and_remedy_for_every_failure():
    text = cb.format_report(cb.interpret(CACHYOS))
    assert "compute backends on this machine:" in text
    assert "cpu" in text and "usable" in text
    assert "cuda  NOT usable — missing: libcublas.so.12" in text
    assert "remedy: pip install nvidia-cublas-cu12" in text
    assert "rocm  NOT usable" in text


def test_real_rocm_capability_is_never_a_bare_error():
    cap = cb.rocm_capability()
    if not cap.usable:
        assert cap.missing and cap.remedy


# --- selection policy against injected seams -------------------------------

def _seams(*, count=0, ready=(True, ""), rocm_usable=False,
           rocm_missing="no AMD GPU and no ROCm runtime are visible on this machine",
           rocm_remedy="use --compute-device cpu or auto"):
    cap = cb.Capability(
        "rocm", rocm_usable,
        None if rocm_usable else rocm_missing,
        None if rocm_usable else rocm_remedy,
        "CTranslate2 is a ROCm/HIP build and sees 2 device(s)")
    return dict(cuda_count=lambda: count,
                cuda_ready=lambda *a, **k: ready,
                rocm=lambda: cap)


def _boom(*_a, **_kw):
    raise AssertionError("this mode must not probe anything")


def test_cpu_mode_never_probes_anything():
    plan = cb.resolve("cpu", cuda_count=_boom, cuda_ready=_boom, rocm=_boom)
    assert plan.device == "cpu" and not plan.gpu_active and plan.note is None


def test_unknown_backend_name_fails_loudly_with_the_valid_names():
    with pytest.raises(cb.DeviceUnavailable) as excinfo:
        cb.resolve("gpu", cuda_count=_boom, cuda_ready=_boom, rocm=_boom)
    assert "cpu, auto, cuda or rocm" in str(excinfo.value)


def test_explicit_cuda_without_a_device_is_actionable():
    with pytest.raises(CudaUnavailable) as excinfo:
        cb.resolve("cuda", **_seams(count=0))
    message = str(excinfo.value)
    assert "no CUDA device" in message and "--compute-device cpu" in message


def test_explicit_cuda_with_a_broken_runtime_names_the_packages():
    seams = _seams(count=1, ready=(False, CUBLAS_MISSING))
    with pytest.raises(CudaUnavailable) as excinfo:
        cb.resolve("cuda", **seams)
    message = str(excinfo.value)
    assert "libcublas.so.12" in message
    assert "nvidia-cublas-cu12" in message
    assert "docs/GPU_LINUX.md" in message and "--compute-device cpu" in message


def test_explicit_cuda_on_a_rocm_build_points_at_rocm_instead():
    with pytest.raises(CudaUnavailable) as excinfo:
        cb.resolve("cuda", **_seams(count=0, rocm_usable=True))
    assert "--compute-device rocm" in str(excinfo.value)


def test_explicit_rocm_unusable_fails_with_missing_piece_and_remedy():
    seams = _seams(rocm_usable=False,
                   rocm_missing="an AMD GPU is visible but neither the ROCm "
                                "runtime nor a ROCm-built CTranslate2 was found",
                   rocm_remedy="install ROCm per AMD's current "
                               "system-requirements matrix (docs/GPU_LINUX.md)")
    with pytest.raises(cb.DeviceUnavailable) as excinfo:
        cb.resolve("rocm", **seams)
    message = str(excinfo.value)
    assert "compute_device=rocm" in message
    assert "ROCm runtime" in message and "docs/GPU_LINUX.md" in message


def test_explicit_rocm_usable_resolves_to_the_gpu_with_a_truthful_note():
    plan = cb.resolve("rocm", **_seams(count=0, rocm_usable=True))
    assert plan.device == "cuda"          # CTranslate2's API device name
    assert plan.kind == "rocm" and plan.gpu_active
    assert plan.note == "rocm (CTranslate2 HIP build)"


def test_auto_without_any_gpu_stays_on_the_historical_cpu_path():
    plan = cb.resolve("auto", **_seams(count=0))
    assert (plan.device, plan.note, plan.gpu_active, plan.kind) == ("auto", None, False, None)


def test_auto_pins_cpu_with_the_reason_when_the_runtime_is_broken():
    plan = cb.resolve("auto", **_seams(count=1, ready=(False, CUBLAS_MISSING)))
    assert plan.device == "cpu" and not plan.gpu_active
    assert plan.note is not None
    assert "libcublas.so.12" in plan.note and "docs/GPU_LINUX.md" in plan.note


def test_auto_keeps_auto_when_cuda_is_usable():
    plan = cb.resolve("auto", **_seams(count=1, ready=(True, "")))
    assert plan.device == "auto" and plan.gpu_active and plan.kind == "cuda"
    assert plan.note is None


def test_auto_uses_rocm_when_only_the_rocm_build_works():
    plan = cb.resolve("auto", **_seams(count=0, rocm_usable=True))
    assert plan.device == "auto" and plan.gpu_active and plan.kind == "rocm"
    assert plan.note == "rocm (CTranslate2 HIP build)"


def test_auto_prefers_rocm_when_a_counted_device_has_no_cuda_runtime():
    """A HIP build can count devices while libcublas can never load."""
    plan = cb.resolve("auto", **_seams(count=2, ready=(False, CUBLAS_MISSING),
                                       rocm_usable=True))
    assert plan.device == "auto" and plan.kind == "rocm" and plan.gpu_active


def test_macos_never_probes_and_rejects_rocm_readably(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    plan = cb.resolve("auto", cuda_count=_boom, cuda_ready=_boom, rocm=_boom)
    assert plan.device == "auto" and plan.note is None
    with pytest.raises(CudaUnavailable, match="macOS"):
        cb.resolve("rocm", cuda_count=_boom, cuda_ready=_boom, rocm=_boom)
    with pytest.raises(CudaUnavailable, match="macOS"):
        cb.resolve("cuda", cuda_count=_boom, cuda_ready=_boom, rocm=_boom)


# --- per-backend compute type ----------------------------------------------

@pytest.mark.parametrize("compute_type,device,gpu_active,expected", [
    (None, "cpu", False, "int8"),
    ("int8", "cpu", False, "int8"),
    ("int8", "auto", False, "int8"),      # auto that stayed on CPU
    ("int8", "auto", True, "float16"),    # auto that resolved to a GPU
    ("int8", "cuda", True, "float16"),
    (None, "cuda", True, "float16"),      # unset means the default
    ("float32", "cuda", True, "float32"),  # an explicit choice always wins
    ("int8_float16", "cuda", True, "int8_float16"),
    ("float16", "cpu", False, "float16"),  # explicit choice, CPU included
])
def test_compute_type_mapping(compute_type, device, gpu_active, expected):
    assert cb.compute_type_for(compute_type, device, gpu_active=gpu_active) == expected


# --- integration: crash isolation for the rocm path ------------------------

def test_rocm_native_crash_fails_with_rocm_advice_and_never_retries(
        monkeypatch, data_dir, tmp_path):
    """Explicit rocm + a real SIGSEGV in the worker: loud, ROCm-flavoured
    advice, exactly one load (no silent CPU restart) — isolation inherited
    from t_7c87bd2c applies to this backend too."""
    log = tmp_path / "loads.txt"
    monkeypatch.setenv("LISTN_TEST_LOAD_LOG", str(log))
    monkeypatch.setattr(tr, "_FACTORY_PATH", f"{FACTORIES}:segfault_unless_cpu_factory")
    wrapper = tr.get_transcriber("tiny", "faster-whisper", compute_device="rocm")
    audio = np.full(16000, 0.1, dtype=np.float32)
    with pytest.raises(CudaUnavailable) as excinfo:
        wrapper.transcribe(audio)
    message = str(excinfo.value)
    assert "compute_device=rocm" in message and "SIGSEGV" in message
    assert "ROCm" in message and "--compute-device cpu" in message
    assert len(log.read_text().splitlines()) == 1  # no CPU retry
    # Sticky: a later call fails fast and never respawns a worker.
    with pytest.raises(CudaUnavailable, match="SIGSEGV"):
        wrapper.transcribe(audio)
    assert len(log.read_text().splitlines()) == 1
    wrapper.close()
