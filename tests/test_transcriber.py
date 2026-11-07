import sys
import types

import numpy as np
import pytest

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

    def __init__(self, model, **kwargs):
        self.model = model
        self.init_kwargs = kwargs
        self.transcribe_kwargs = None
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs = kwargs
        return iter([]), types.SimpleNamespace(language="en")


@pytest.fixture
def fake_whisper_module(monkeypatch):
    """Replace faster_whisper so no model is loaded and no network is touched."""
    FakeWhisperModel.instances = []
    tr._cache.clear()
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    yield FakeWhisperModel
    tr._cache.clear()


def _build(model="small", **kwargs):
    return tr.get_transcriber(model, "faster-whisper", **kwargs)


def _knobs(instance):
    """The kwargs WhisperModel got, minus the cache-path marker.

    ``load_local_or_download`` adds ``local_files_only=True`` on the cached
    attempt; the knob identity we care about is everything else.
    """
    return {k: v for k, v in instance.init_kwargs.items() if k != "local_files_only"}


def test_default_knobs_are_byte_identical_to_old_hardcoded_call(fake_whisper_module, data_dir):
    """Nothing set -> exactly the old call: device auto, int8, no cpu_threads, beam 5."""
    t = _build()
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "auto", "compute_type": "int8"}
    assert "cpu_threads" not in made.init_kwargs
    list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert made.transcribe_kwargs["beam_size"] == 5
    # Settings defaults agree with what the backend received.
    s = Settings.load()
    assert (s.compute_type, s.cpu_threads, s.beam_size) == ("int8", None, 5)


def test_settings_overrides_reach_the_backend(fake_whisper_module, data_dir):
    """Values written to settings.json flow all the way into WhisperModel."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text(
        '{"compute_type": "int8_float16", "cpu_threads": 3, "beam_size": 9}')
    t = _build()
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "auto", "compute_type": "int8_float16",
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
    assert _knobs(made) == {"device": "auto", "compute_type": "float16",
                            "cpu_threads": 2}
    list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert made.transcribe_kwargs["beam_size"] == 7


def test_explicit_arguments_override_settings(fake_whisper_module, data_dir, monkeypatch):
    monkeypatch.setenv("LISTN_COMPUTE_TYPE", "float16")
    _build(compute_type="int8", cpu_threads=None, beam_size=1)
    made = fake_whisper_module.instances[0]
    assert _knobs(made) == {"device": "auto", "compute_type": "int8"}
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


# --- CPU fallback when the CUDA user-space libraries are missing ------------
# Mark's CachyOS laptop: CTranslate2 counts a GPU (device "auto" resolves to
# CUDA) but libcublas.so.12 is not installed, so every op on the CUDA build
# dies with a RuntimeError naming the library.

_CUBLAS_ERROR = "Library libcublas.so.12 is not found or cannot be loaded"


class CudaBrokenWhisper:
    """faster_whisper.WhisperModel stand-in for that laptop.

    Builds and transcribes on ``device="auto"`` raise the way CTranslate2
    does; ``device="cpu"`` works — unless a test widens ``fail_devices``.
    """

    instances: list = []
    fail_devices = ("auto",)
    fail_at_init = False

    def __init__(self, model, **kwargs):
        self.model = model
        self.init_kwargs = kwargs
        self.transcribe_kwargs = None
        CudaBrokenWhisper.instances.append(self)
        if CudaBrokenWhisper.fail_at_init and kwargs.get("device") in CudaBrokenWhisper.fail_devices:
            raise RuntimeError(_CUBLAS_ERROR)

    def transcribe(self, audio, **kwargs):
        self.transcribe_kwargs = kwargs
        if self.init_kwargs.get("device") in CudaBrokenWhisper.fail_devices:
            raise RuntimeError(_CUBLAS_ERROR)
        segment = types.SimpleNamespace(start=0.0, end=1.0, text="hello", no_speech_prob=0.0)
        return iter([segment]), types.SimpleNamespace(language="en")


@pytest.fixture
def cuda_broken_whisper(monkeypatch, data_dir):
    """A GPU-visible box with no CUDA libraries, faked end to end."""
    CudaBrokenWhisper.instances = []
    CudaBrokenWhisper.fail_devices = ("auto",)
    CudaBrokenWhisper.fail_at_init = False
    tr._cache.clear()
    # The probe reports a device, like the broken laptop; the libraries are
    # still absent, which only the op-time error can reveal.
    monkeypatch.setattr(tr, "_cuda_device_count", lambda: 1)
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=CudaBrokenWhisper))
    yield CudaBrokenWhisper
    tr._cache.clear()


def test_transcribe_cuda_error_falls_back_to_cpu_once_and_reports_it(cuda_broken_whisper):
    statuses = []
    t = _build(status=statuses.append)
    assert t.device == "auto" and t.status_note is None

    segs = list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert [s.text for s in segs] == ["hello"]
    # Exactly one rebuild: the auto build raised, the CPU build ran and served
    # the segments — no retry loop.
    assert [m.init_kwargs["device"] for m in cuda_broken_whisper.instances] == ["auto", "cpu"]
    assert t.device == "cpu"
    assert t.status_note == "CUDA libraries unavailable — transcribing on CPU"
    assert statuses == [tr.CPU_FALLBACK_MESSAGE]

    # Later calls reuse the CPU model: no rebuild, no repeated announcement.
    list(t.segments(np.zeros(16000, dtype=np.float32)))
    assert len(cuda_broken_whisper.instances) == 2
    assert statuses == [tr.CPU_FALLBACK_MESSAGE]


def test_model_init_cuda_error_falls_back_to_cpu_and_reports_it(cuda_broken_whisper):
    cuda_broken_whisper.fail_at_init = True
    statuses = []
    t = _build(status=statuses.append)
    assert t.device == "cpu"
    assert t.status_note == tr.CPU_FALLBACK_MESSAGE
    assert statuses == [tr.CPU_FALLBACK_MESSAGE]
    assert [m.init_kwargs["device"] for m in cuda_broken_whisper.instances] == ["auto", "cpu"]
    # The job still completes on the rebuilt model.
    assert [s.text for s in t.segments(np.zeros(16000, dtype=np.float32))] == ["hello"]


def test_cuda_error_on_the_cpu_rebuild_raises_after_one_attempt(cuda_broken_whisper):
    """Even a hopeless box must fail once, honestly — never loop."""
    cuda_broken_whisper.fail_devices = ("auto", "cpu")
    statuses = []
    t = _build(status=statuses.append)
    with pytest.raises(RuntimeError, match="libcublas"):
        list(t.segments(np.zeros(16000, dtype=np.float32)))
    # One auto build + one CPU rebuild, then the error propagates.
    assert len(cuda_broken_whisper.instances) == 2
    assert statuses == [tr.CPU_FALLBACK_MESSAGE]


def test_cache_key_distinguishes_the_cpu_fallback_from_the_configured_build(cuda_broken_whisper):
    t = _build()
    # A healthy build is keyed as configured (device "auto").
    assert [key[-1] for key in tr._cache] == ["auto"]

    list(t.segments(np.zeros(16000, dtype=np.float32)))   # falls back to CPU
    assert t.device == "cpu"

    hits = []
    again = _build(status=hits.append)
    assert again is t                       # served from the cpu key, not rebuilt
    assert [key[-1] for key in tr._cache] == ["cpu"]   # the auto key never survives
    assert hits == [tr.CPU_FALLBACK_MESSAGE]

    # A different configuration must not inherit the fallback model.
    other = _build(compute_type="float16")
    assert other is not t
    assert len(cuda_broken_whisper.instances) == 3   # auto, cpu rebuild, float16 auto
