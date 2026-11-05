import sys
import types

import numpy as np
import pytest

from listn import compute_backend as cb
from listn import transcriber as tr
from listn.config import COMPUTE_TYPES, Settings
from listn.transcriber import RawSegment, _WindowedBackend, is_hallucination

from .conftest import FakeTranscriber


@pytest.mark.parametrize("text", ["Thanks for watching!", "  thank you for watching. ",
                                  "Subtitles by the Amara.org community", "[BLANK_AUDIO]", "(music)", "...", ""])
def test_known_hallucinations_dropped(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", ["OK.", "No.", "Yes", "Thank you.", "So the budget is fine."])
def test_real_short_answers_kept(text):
    assert not is_hallucination(text, no_speech_prob=0.1)


def test_fillers_dropped_only_when_no_speech_likely():
    assert is_hallucination("Thank you.", no_speech_prob=0.7)
    assert is_hallucination("you", quiet=True)
    assert not is_hallucination("you", no_speech_prob=0.05)


def test_transcribe_applies_offset_and_filters():
    t = FakeTranscriber()
    audio = (0.2 * np.sin(np.arange(32000) / 5)).astype(np.float32)
    segs = t.transcribe(audio, offset=12.0)
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(12.0)
    assert segs[0].end == pytest.approx(13.0)
    assert segs[0].text == "chunk 1 first half"


def test_transcribe_reports_progress_to_completion():
    seen = []
    FakeTranscriber().transcribe(np.full(16000, 0.1, dtype=np.float32), progress=seen.append)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_empty_audio_returns_nothing():
    assert FakeTranscriber().transcribe(np.zeros(0, dtype=np.float32)) == []


class Windowed(_WindowedBackend):
    window_seconds = 2

    def __init__(self):
        super().__init__("tiny")
        self.prompts = []

    def _run(self, audio, language, prompt):
        self.prompts.append(prompt)
        return [{"start": 0.5, "end": 1.0, "text": "hello"}]


def test_windowed_backend_offsets_windows_and_skips_silence():
    audio = np.concatenate([np.full(32000, 0.1), np.zeros(32000), np.full(32000, 0.1)]).astype(np.float32)
    w = Windowed()
    segs = list(w.segments(audio))
    assert [s.start for s in segs] == [0.5, 4.5]
    assert w.prompts == [None, "hello"]


def test_resolve_backend_on_linux_defaults_to_faster_whisper(monkeypatch):
    monkeypatch.setattr(tr.sys, "platform", "linux")
    assert tr.resolve_backend("auto") == "faster-whisper"
    assert tr.resolve_backend("mlx") == "mlx"


def test_get_transcriber_caches_and_reports_missing_backend():
    made = []

    def factory(model, download=None):
        made.append(model)
        return FakeTranscriber(model)

    tr.register_backend("fake", factory)
    a = tr.get_transcriber("tiny", "fake")
    assert tr.get_transcriber("tiny", "fake") is a
    tr.get_transcriber("base", "fake")
    assert made == ["tiny", "base"]

    def missing(model, download=None):
        raise ImportError("nope")

    tr.register_backend("mlx", missing)
    with pytest.raises(RuntimeError, match="listn\\[mlx\\]"):
        tr.get_transcriber("tiny", "mlx")
    tr.register_backend("mlx", tr.MlxWhisper)


def test_raw_segment_defaults():
    assert RawSegment(0, 1, "x").no_speech_prob == 0.0


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel and records how it was built."""

    instances = []
    # Devices whose construction raises the real-world CUDA failure.
    fail_devices: set = set()
    # What the inner ctranslate2 model reports as its device; None derives it
    # from the requested one (auto means CPU on a machine without a GPU).
    device_report = None

    def __init__(self, model, **kwargs):
        self.model_name = model
        self.init_kwargs = kwargs
        self.transcribe_kwargs = None
        FakeWhisperModel.instances.append(self)
        if kwargs.get("device") in FakeWhisperModel.fail_devices:
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        report = FakeWhisperModel.device_report
        if report is None:
            requested = kwargs.get("device")
            report = "cpu" if requested in (None, "auto", "cpu") else requested
        # Like the real faster_whisper.WhisperModel, ``model`` is the inner
        # ctranslate2 model, which reports the device it is actually on.
        self.model = types.SimpleNamespace(device=report)

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs = kwargs
        return iter([]), types.SimpleNamespace(language="en")


@pytest.fixture
def fake_whisper_module(monkeypatch):
    """Replace faster_whisper so no model is loaded and no network is touched."""
    FakeWhisperModel.instances = []
    FakeWhisperModel.fail_devices = set()
    FakeWhisperModel.device_report = None
    tr._cache.clear()
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    yield FakeWhisperModel
    tr._cache.clear()


@pytest.fixture(autouse=True)
def no_cuda_by_default(monkeypatch):
    """Pin device resolution to a machine without a GPU or ROCm.

    The knob tests below assert the exact ``WhisperModel`` kwargs, and a GPU
    test host resolves ``auto`` differently — that is the new behaviour. Tests
    that want a visible GPU (or a usable ROCm build) opt in by re-patching
    ``cuda_device_count`` / ``compute_backend.rocm_capability``.
    """
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(cb, "rocm_capability", lambda: cb.Capability(
        "rocm", False,
        "no AMD GPU and no ROCm runtime are visible on this machine",
        "use --compute-device cpu or auto"))


def _build(model="small", **kwargs):
    # These tests inspect the constructed model in-process (device knobs,
    # caching, status lines). The process boundary itself — including
    # native crashes — is covered in tests/test_isolation.py.
    kwargs.setdefault("isolate", False)
    return tr.get_transcriber(model, "faster-whisper", **kwargs)


def _knobs(instance):
    """The kwargs WhisperModel got, minus the cache-path marker.

    ``load_local_or_download`` adds ``local_files_only=True`` on the cached
    attempt; the knob identity we care about is everything else.
    """
    return {k: v for k, v in instance.init_kwargs.items() if k != "local_files_only"}


def test_default_knobs_are_byte_identical_to_old_hardcoded_call(fake_whisper_module, data_dir):
    """Nothing set -> safe CPU default, int8, no cpu_threads, beam 5."""
    t = _build()
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "cpu", "compute_type": "int8"}
    assert "cpu_threads" not in made.init_kwargs
    list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert made.transcribe_kwargs["beam_size"] == 5
    # Settings defaults agree with what the backend received.
    s = Settings.load()
    assert (s.compute_type, s.cpu_threads, s.beam_size, s.compute_device) == ("int8", None, 5, "cpu")


def test_settings_overrides_reach_the_backend(fake_whisper_module, data_dir):
    """Values written to settings.json flow all the way into WhisperModel."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text(
        '{"compute_type": "int8_float16", "cpu_threads": 3, "beam_size": 9}')
    t = _build()
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "cpu", "compute_type": "int8_float16",
                            "cpu_threads": 3}
    list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert made.transcribe_kwargs["beam_size"] == 9


def test_env_overrides_win_over_settings_file(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setenv("LISTN_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("LISTN_CPU_THREADS", "2")
    monkeypatch.setenv("LISTN_BEAM_SIZE", "7")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"compute_type": "int8", "beam_size": 5}')
    t = _build()
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "cpu", "compute_type": "float16",
                            "cpu_threads": 2}
    list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert made.transcribe_kwargs["beam_size"] == 7


def test_explicit_arguments_override_settings(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setenv("LISTN_COMPUTE_TYPE", "float16")
    _build(compute_type="int8", cpu_threads=None, beam_size=1)
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "cpu", "compute_type": "int8"}
    assert "cpu_threads" not in made.init_kwargs


def test_knob_change_evicts_the_cached_model(fake_whisper_module, data_dir):
    """A different knob set must not reuse a model built with the old one."""
    a = _build()
    b = _build()
    assert a is b and len(fake_whisper_module.instances) == 1
    c = _build(compute_type="float16")
    assert c is not a and len(fake_whisper_module.instances) == 2


def test_non_faster_whisper_backends_ignore_the_knobs(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setenv("LISTN_COMPUTE_TYPE", "float16")
    tr.register_backend("fake-knobs", lambda model, download=None: FakeTranscriber(model))
    t = tr.get_transcriber("tiny", "fake-knobs")
    assert isinstance(t, FakeTranscriber)


@pytest.mark.parametrize("env,field,bad", [
    ("LISTN_COMPUTE_TYPE=banana", "compute_type", "banana"),
    ("LISTN_COMPUTE_TYPE=Int8 Float16", "compute_type", "int8 float16"),
    ("LISTN_CPU_THREADS=0", "cpu_threads", "0"),
    ("LISTN_CPU_THREADS=-2", "cpu_threads", "-2"),
    ("LISTN_CPU_THREADS=many", "cpu_threads", "many"),
    ("LISTN_CPU_THREADS=2.5", "cpu_threads", "2.5"),
    ("LISTN_BEAM_SIZE=0", "beam_size", "0"),
    ("LISTN_BEAM_SIZE=huge", "beam_size", "huge"),
    ("LISTN_BEAM_SIZE=101", "beam_size", "101"),
    ("LISTN_COMPUTE_DEVICE=gpu", "compute_device", "gpu"),
    ("LISTN_COMPUTE_DEVICE=CUDA!", "compute_device", "CUDA!"),
])
def test_invalid_env_values_fail_loudly_with_a_readable_message(data_dir, monkeypatch, env, field, bad):
    var, value = env.split("=", 1)
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError) as excinfo:
        Settings.load()
    message = str(excinfo.value)
    assert field in message, message
    assert bad in message, message


def test_invalid_settings_file_values_fail_loudly(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"beam_size": 0}')
    with pytest.raises(ValueError, match="beam_size"):
        Settings.load()
    (data_dir / "settings.json").write_text('{"compute_type": "tpu"}')
    with pytest.raises(ValueError, match="compute_type"):
        Settings.load()
    (data_dir / "settings.json").write_text('{"cpu_threads": "none"}')
    with pytest.raises(ValueError, match="cpu_threads"):
        Settings.load()


def test_all_documented_compute_types_are_accepted(data_dir):
    for kind in COMPUTE_TYPES:
        assert Settings(compute_type=kind).validated().compute_type == kind


def test_cpu_threads_and_beam_size_round_trip_through_strings(data_dir):
    """Env vars are strings; they must parse to the right ints, not be rejected."""
    s = Settings(**{"cpu_threads": "4", "beam_size": "5"}).validated()  # pyright: ignore[reportArgumentType]
    assert (s.cpu_threads, s.beam_size) == (4, 5)
    assert Settings(**{"cpu_threads": ""}).validated().cpu_threads is None  # pyright: ignore[reportArgumentType]


def test_transcribe_help_shows_the_effective_knob_values(data_dir, capsys, monkeypatch):
    """`listn transcribe --help` reports what this run would actually use."""
    from listn import cli

    monkeypatch.setenv("LISTN_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("LISTN_CPU_THREADS", "4")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["transcribe", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "compute type: float16" in out
    assert "cpu threads:  4" in out
    assert "beam size:    5" in out
    assert "LISTN_COMPUTE_TYPE" in out


def test_transcribe_help_says_threads_are_auto_when_unset(data_dir, capsys):
    from listn import cli

    with pytest.raises(SystemExit):
        cli.main(["transcribe", "--help"])
    out = capsys.readouterr().out
    assert "cpu threads:  auto (backend chooses)" in out


def test_invalid_knob_exits_readably_not_with_a_traceback(data_dir, capsys, monkeypatch):
    from listn import cli

    monkeypatch.setenv("LISTN_BEAM_SIZE", "banana")
    assert cli.main(["transcribe", "whatever.wav"]) == 2
    err = capsys.readouterr().err
    assert "beam_size" in err and "banana" in err
    assert "Traceback" not in err


# --- compute device: CPU mode, CUDA pre-flight, safe fallback ---------------

def _gpu_visible_but_libs_broken(monkeypatch):
    """The CachyOS laptop: a CUDA device CTranslate2 can see, no libcublas."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        tr, "cuda_runtime_ready",
        lambda *a, **k: (False, "libcublas.so.12: cannot open shared object file: No such file or directory"))


def _never_probe(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the GPU must not be probed in this mode")

    monkeypatch.setattr(tr, "cuda_device_count", boom)
    monkeypatch.setattr(tr, "cuda_runtime_ready", boom)


def test_resolve_device_cpu_never_touches_the_gpu(monkeypatch):
    _never_probe(monkeypatch)
    assert tr.resolve_device("cpu") == ("cpu", None)
    assert tr.resolve_device("CPU") == ("cpu", None)


def test_resolve_device_auto_without_a_gpu_is_unchanged(monkeypatch):
    assert tr.resolve_device("auto") == ("auto", None)
    assert tr.resolve_device("") == ("auto", None)


def test_resolve_device_auto_pins_cpu_when_the_runtime_is_broken(monkeypatch):
    _gpu_visible_but_libs_broken(monkeypatch)
    device, note = tr.resolve_device("auto")
    assert device == "cpu"
    assert "libcublas.so.12" in note and "docs/GPU_LINUX.md" in note


def test_resolve_device_auto_keeps_auto_when_cuda_is_usable(monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 2)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    assert tr.resolve_device("auto") == ("auto", None)
    assert tr.resolve_device("cuda") == ("cuda", None)


def test_resolve_device_cuda_without_a_gpu_is_actionable(monkeypatch):
    with pytest.raises(tr.CudaUnavailable) as excinfo:
        tr.resolve_device("cuda")
    message = str(excinfo.value)
    assert "no CUDA device" in message and "--compute-device cpu" in message


def test_resolve_device_cuda_with_a_broken_runtime_is_actionable(monkeypatch):
    _gpu_visible_but_libs_broken(monkeypatch)
    with pytest.raises(tr.CudaUnavailable) as excinfo:
        tr.resolve_device("cuda")
    message = str(excinfo.value)
    assert "libcublas.so.12" in message
    assert "--compute-device cpu" in message and "docs/GPU_LINUX.md" in message


def test_resolve_device_on_macos_never_probes(monkeypatch):
    monkeypatch.setattr(tr.sys, "platform", "darwin")
    _never_probe(monkeypatch)
    assert tr.resolve_device("auto") == ("auto", None)
    with pytest.raises(tr.CudaUnavailable, match="macOS"):
        tr.resolve_device("cuda")


# The probe itself: a real child process, because loading a CUDA library can
# end a process without raising anything Python can catch.

_REAL_LIB = "libSystem.B.dylib" if sys.platform == "darwin" else "libc.so.6"


def test_cuda_runtime_probe_reports_a_loadable_library():
    assert tr.cuda_runtime_ready((_REAL_LIB,)) == (True, "")


def test_cuda_runtime_probe_reports_a_missing_library():
    ready, detail = tr.cuda_runtime_ready(("liblistn-not-a-library.so.1",))
    assert not ready
    assert "liblistn-not-a-library.so.1" in detail


def test_cuda_runtime_probe_survives_a_child_killed_by_a_signal(monkeypatch):
    monkeypatch.setattr(tr, "_PROBE_SCRIPT",
                        "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)")
    ready, detail = tr.cuda_runtime_ready((_REAL_LIB,))
    assert not ready
    assert "signal 11" in detail


def test_cuda_runtime_probe_reports_a_timeout(monkeypatch):
    monkeypatch.setattr(tr, "_PROBE_SCRIPT", "import time; time.sleep(30)")
    ready, detail = tr.cuda_runtime_ready((_REAL_LIB,), timeout=1)
    assert not ready
    assert "timed out" in detail


# Building with the fake faster_whisper: fallback, clean failure, no false success.

def test_auto_falls_back_to_cpu_when_the_gpu_load_fails(fake_whisper_module, data_dir, monkeypatch):
    """Probe passes, CTranslate2 still refuses: transcribe on CPU, say so."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    fake_whisper_module.fail_devices = {"auto"}
    seen = []
    t = _build(compute_device="auto", status=seen.append)
    assert [m.init_kwargs["device"] for m in fake_whisper_module.instances] == ["auto", "cpu"]
    assert t.device == "cpu"
    assert seen[-1].startswith("device: cpu")
    assert "libcublas.so.12" in seen[-1] and "running on CPU" in seen[-1]
    assert list(t.segments(np.zeros(16000, dtype=np.float32))) == []


def test_auto_uses_the_gpu_when_the_probe_passes(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    fake_whisper_module.device_report = "cuda"
    seen = []
    t = _build(compute_device="auto", status=seen.append)
    assert fake_whisper_module.instances[0].init_kwargs["device"] == "auto"
    assert t.device == "cuda"
    assert seen == ["device: cuda"]


def test_explicit_cuda_failure_is_readable_and_never_retries_on_cpu(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    fake_whisper_module.fail_devices = {"cuda"}
    with pytest.raises(tr.CudaUnavailable) as excinfo:
        _build(compute_device="cuda")
    message = str(excinfo.value)
    assert "libcublas.so.12" in message and "--compute-device cpu" in message
    # No silent CPU retry when the user asked for the GPU by name.
    assert [m.init_kwargs["device"] for m in fake_whisper_module.instances] == ["cuda"]


def test_a_device_failure_never_claims_a_download(fake_whisper_module, data_dir, monkeypatch):
    """A broken GPU is not a missing model: no download message, no network path."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    fake_whisper_module.fail_devices = {"auto"}
    seen = []
    _build(status=seen.append)
    # Every attempt was the cached one; the download path was never entered.
    assert all(m.init_kwargs.get("local_files_only") for m in fake_whisper_module.instances)
    assert not [m for m in seen if "Downloading" in m]


def test_cpu_mode_builds_and_runs_without_probing_anything(fake_whisper_module, data_dir, monkeypatch):
    _never_probe(monkeypatch)
    seen = []
    t = _build(compute_device="cpu", status=seen.append)
    assert fake_whisper_module.instances[0].init_kwargs["device"] == "cpu"
    assert t.device == "cpu"
    assert seen == ["device: cpu"]
    assert list(t.segments(np.zeros(16000, dtype=np.float32))) == []


def test_device_is_reported_on_cache_hits_too(fake_whisper_module, data_dir):
    first, second = [], []
    a = _build(status=first.append)
    b = _build(status=second.append)
    assert a is b and len(fake_whisper_module.instances) == 1
    assert first == ["device: cpu"] and second == ["device: cpu"]


def test_compute_device_is_part_of_the_cache_key(fake_whisper_module, data_dir):
    a = _build(compute_device="auto")
    b = _build(compute_device="cpu")
    assert a is not b
    assert len(fake_whisper_module.instances) == 2


def test_settings_compute_device_reaches_the_backend(fake_whisper_module, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"compute_device": "cpu"}')
    _build()
    assert fake_whisper_module.instances[0].init_kwargs["device"] == "cpu"


def test_env_compute_device_beats_the_settings_file(fake_whisper_module, data_dir, monkeypatch):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"compute_device": "cpu"}')
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "cuda")
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    _build()
    assert fake_whisper_module.instances[0].init_kwargs["device"] == "cuda"


def test_explicit_argument_beats_env_and_file(fake_whisper_module, data_dir, monkeypatch):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"compute_device": "cuda"}')
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "cuda")
    _never_probe(monkeypatch)  # cpu mode must not even look at the GPU
    _build(compute_device="cpu")
    assert fake_whisper_module.instances[0].init_kwargs["device"] == "cpu"


def test_invalid_compute_device_fails_loudly(data_dir):
    with pytest.raises(ValueError, match="compute_device"):
        Settings(**{"compute_device": "gpu"}).validated()  # pyright: ignore[reportArgumentType]
    assert Settings(**{"compute_device": " CPU "}).validated().compute_device == "cpu"  # pyright: ignore[reportArgumentType]


def test_invalid_compute_device_env_exits_readably_not_with_a_traceback(data_dir, capsys, monkeypatch):
    """Kept beside the other CLI knob test: this file runs no engine jobs."""
    from listn import cli

    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "gpu")
    assert cli.main(["transcribe", "whatever.wav"]) == 2
    err = capsys.readouterr().err
    assert "compute_device" in err and "gpu" in err
    assert "Traceback" not in err


# --- vendor-neutral backends: ROCm selection and per-device compute type ----

def _pin_rocm(monkeypatch, usable: bool, *, missing: str = "", remedy: str = ""):
    monkeypatch.setattr(cb, "rocm_capability", lambda: cb.Capability(
        "rocm", usable, None if usable else missing, None if usable else remedy,
        "CTranslate2 is a ROCm/HIP build and sees 2 device(s)"))


def test_auto_uses_rocm_when_only_the_rocm_build_works(fake_whisper_module, data_dir, monkeypatch):
    """AMD machine: no CUDA device, ROCm build usable -> GPU via auto, said out loud."""
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    _pin_rocm(monkeypatch, True)
    fake_whisper_module.device_report = "cuda"
    seen = []
    t = _build(compute_device="auto", status=seen.append)
    made = fake_whisper_module.instances[0]
    assert made.init_kwargs["device"] == "auto"
    # Left at the default compute type: float16 on the GPU, int8 on CPU.
    assert made.init_kwargs["compute_type"] == "float16"
    assert t.device == "cuda"
    assert "rocm" in seen[-1] and "HIP" in seen[-1]


def test_explicit_rocm_builds_on_the_gpu_device_with_float16(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    _pin_rocm(monkeypatch, True)
    fake_whisper_module.device_report = "cuda"
    seen = []
    t = _build(compute_device="rocm", status=seen.append)
    made = fake_whisper_module.instances[0]
    assert made.init_kwargs["device"] == "cuda"       # CTranslate2's API device
    assert made.init_kwargs["compute_type"] == "float16"
    assert t.device == "cuda"
    assert seen == ["device: cuda — rocm (CTranslate2 HIP build)"]


def test_explicit_rocm_unusable_fails_before_any_model_build(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    _pin_rocm(monkeypatch, False,
              missing="an AMD GPU is visible but neither the ROCm runtime nor "
                      "a ROCm-built CTranslate2 was found",
              remedy="install ROCm per AMD's current system-requirements "
                     "matrix (docs/GPU_LINUX.md)")
    with pytest.raises(tr.CudaUnavailable) as excinfo:
        _build(compute_device="rocm")
    message = str(excinfo.value)
    assert "ROCm runtime" in message and "docs/GPU_LINUX.md" in message
    assert fake_whisper_module.instances == []  # failed before any build


def test_explicit_rocm_failure_never_retries_on_cpu(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    _pin_rocm(monkeypatch, True)
    fake_whisper_module.fail_devices = {"cuda"}
    with pytest.raises(tr.CudaUnavailable) as excinfo:
        _build(compute_device="rocm")
    message = str(excinfo.value)
    assert "compute_device=rocm" in message and "--compute-device cpu" in message
    # No silent CPU retry when the GPU was asked for by name.
    assert [m.init_kwargs["device"] for m in fake_whisper_module.instances] == ["cuda"]


def test_auto_rocm_failure_falls_back_to_cpu_with_int8(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    _pin_rocm(monkeypatch, True)
    fake_whisper_module.fail_devices = {"auto"}
    seen = []
    t = _build(compute_device="auto", status=seen.append)
    attempts = fake_whisper_module.instances
    assert [m.init_kwargs["device"] for m in attempts] == ["auto", "cpu"]
    assert attempts[0].init_kwargs["compute_type"] == "float16"  # the GPU attempt
    assert attempts[1].init_kwargs["compute_type"] == "int8"      # the CPU retry
    assert t.device == "cpu"
    assert "ROCm unusable" in seen[-1] and "running on CPU" in seen[-1]


def test_explicit_compute_type_wins_on_a_gpu(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(tr, "cuda_runtime_ready", lambda *a, **k: (True, ""))
    _build(compute_device="auto", compute_type="float32")
    assert fake_whisper_module.instances[0].init_kwargs["compute_type"] == "float32"
