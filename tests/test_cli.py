import os
import subprocess
import sys

import pytest

from listn import cli, engine as engine_mod, transcriber as tr
from listn.config import Settings
from listn.library import Library, Segment

from .test_engine import wait_for, write_wav


def test_parser_defaults_to_ui():
    args = cli.build_parser().parse_args(["live", "--device", "BlackHole", "--model", "base"])
    assert args.device == "BlackHole" and args.model == "base"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["live", "--model", "huge"])


def test_list_and_export(data_dir, capsys, tmp_path):
    lib = Library()
    m = lib.create("Planning", "live")
    lib.save_segments(m.id, [Segment(0, 2, "Ship it Friday")])
    assert cli.main(["list", "friday"]) == 0
    assert "Planning" in capsys.readouterr().out
    out = tmp_path / "p.srt"
    assert cli.main(["export", m.id, "--format", "srt", "--output", str(out)]) == 0
    assert "Ship it Friday" in out.read_text()
    assert cli.main(["export", "20990101-000000-000000"]) == 1


def test_transcribe_command(data_dir, tmp_path, capsys, monkeypatch, fake_factory):
    monkeypatch.setattr(engine_mod, "get_transcriber", fake_factory)
    wav = write_wav(tmp_path / "memo.wav")
    assert cli.main(["transcribe", str(wav), "--model", "tiny"]) == 0
    assert "chunk 1 first half" in capsys.readouterr().out
    assert len(Library().list()) == 1
    assert cli.main(["transcribe", str(tmp_path / "missing.wav")]) == 1


def test_path_command(data_dir, capsys):
    assert cli.main(["path"]) == 0
    assert str(data_dir) in capsys.readouterr().out


def test_settings_error_is_a_value_error_subclass():
    """Callers that already catch ValueError (the web API, older tests) keep firing."""
    from listn.config import SettingsError

    assert issubclass(SettingsError, ValueError)


def test_unexpected_value_error_from_a_subcommand_propagates(data_dir, capsys, monkeypatch):
    """A bug's ValueError must not be dressed up as ``listn: <message>`` / exit 2.

    cli.main() used to catch every ValueError raised by every subcommand, so a
    bad unpack or off-by-one printed a one-line message and exited 2 — a real
    defect wearing a user-input costume.
    """
    def boom():
        raise ValueError("off-by-one in cmd_path")

    monkeypatch.setattr(cli, "data_dir", boom)
    with pytest.raises(ValueError, match="off-by-one in cmd_path"):
        cli.main(["path"])
    assert "listn:" not in capsys.readouterr().err


def test_unexpected_value_error_exits_1_with_a_traceback():
    """Same contract end to end: a real process dies with a traceback, not exit 2."""
    script = (
        "from listn import cli\n"
        "def boom():\n"
        "    raise ValueError('off-by-one in cmd_path')\n"
        "cli.data_dir = boom\n"
        "cli.main(['path'])\n"
    )
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" in proc.stderr
    assert "off-by-one in cmd_path" in proc.stderr
    assert "listn:" not in proc.stderr


def test_settings_validation_still_exits_2_with_a_clear_message(data_dir, capsys):
    """The intended user-facing error keeps exit 2 and its one-line message,
    whatever subcommand loads the settings (this one is not transcribe)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"beam_size": 0}')
    assert cli.main(["diarize", "20990101-000000-000000"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("listn: "), err
    assert "beam_size" in err and "0" in err
    assert "Traceback" not in err


def test_compute_device_flag_beats_env_and_settings_file(data_dir, tmp_path, capsys, monkeypatch, fake_factory):
    """CLI > LISTN_COMPUTE_DEVICE > settings.json, and the engine sees the choice."""
    monkeypatch.setattr(engine_mod, "get_transcriber", fake_factory)
    scheduled = []
    monkeypatch.setattr("listn.summary.schedule_notes",
                        lambda *a, **k: scheduled.append(a))
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text('{"compute_device": "cuda"}')
    monkeypatch.setenv("LISTN_COMPUTE_DEVICE", "cuda")
    wav = write_wav(tmp_path / "memo.wav")
    assert cli.main(["transcribe", str(wav), "--model", "tiny", "--compute-device", "cpu"]) == 0
    # The engine's worker thread re-reads Settings after publishing job.done;
    # wait until it is past that point so a later test's environment cannot
    # surface underneath it (and so we know that read saw this run's choice).
    wait_for(lambda: scheduled, timeout=10.0)
    assert Settings.load().compute_device == "cpu"
    assert os.environ["LISTN_COMPUTE_DEVICE"] == "cpu"
    assert "chunk 1 first half" in capsys.readouterr().out


def test_compute_device_flag_rejects_unknown_values(data_dir):
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["transcribe", "x.wav", "--compute-device", "gpu"])
    assert excinfo.value.code == 2


def test_transcribe_help_shows_the_compute_device(data_dir, capsys):
    with pytest.raises(SystemExit):
        cli.main(["transcribe", "--help"])
    out = capsys.readouterr().out
    # Contract (a): auto is the shipped default, not the branch's cpu.
    assert "compute device: auto" in out
    assert "--compute-device" in out
    assert "LISTN_COMPUTE_DEVICE" in out


def test_gpu_failure_fails_the_job_with_no_false_success(data_dir, tmp_path, capsys, monkeypatch):
    """A CUDA failure must be exit 1 with the real reason — never an empty success."""
    def broken_factory(model, backend, status=None):
        raise tr.CudaUnavailable(
            "compute_device=cuda was requested, but CUDA is visible but the CUDA runtime "
            "cannot be used (libcublas.so.12: cannot open shared object file). Install the "
            "CUDA 12 libraries (docs/GPU_LINUX.md), or run on CPU with --compute-device cpu.")

    monkeypatch.setattr(engine_mod, "get_transcriber", broken_factory)
    wav = write_wav(tmp_path / "memo.wav")
    assert cli.main(["transcribe", str(wav)]) == 1
    captured = capsys.readouterr()
    assert "failed" in captured.err
    assert "libcublas.so.12" in captured.err and "--compute-device cpu" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out.strip() == ""  # no transcript was written
    sessions = Library().list()
    assert [s.status for s in sessions] == ["error"]
    assert "libcublas.so.12" in sessions[0].error
