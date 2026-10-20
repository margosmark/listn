import pytest

from listn import cli, engine as engine_mod
from listn.library import Library, Segment

from .test_engine import write_wav


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
