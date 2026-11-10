import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from listn.bundle import assemble

from .test_hardening import _uv_layout

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "macos" / "scripts" / "build-app.sh"
MACHO = b"\xcf\xfa\xed\xfe" + b"\0" * 60


def test_the_diarize_extra_is_pyannote_4_with_a_matching_torch():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extra = project["optional-dependencies"]["diarize"]
    pyannote = next(d for d in extra if d.startswith("pyannote.audio"))
    assert ">=4" in pyannote and "<5" in pyannote
    torch = next(d for d in extra if re.match(r"torch[<>=]", d))
    # F1: >=2.13 clears every fixable advisory on torch 2.8.0 (checkpoint RCE
    # CVE-2026-24747 etc.); CVE-2026-4538 has no upstream fix.
    assert ">=2.13" in torch and "<2.9" not in torch
    assert any(d.startswith("torchaudio>=2.11") for d in extra)
    assert any(d.startswith("torchcodec>=0.12") for d in extra)
    assert project["version"] == "0.5.0"


def test_the_lock_resolves_pyannote_4():
    lock = (ROOT / "uv.lock").read_text()
    version = re.search(r'name = "pyannote-audio"\nversion = "([^"]+)"', lock).group(1)
    assert version.startswith("4.")
    assert re.search(r'name = "torch"\nversion = "2\.(8|9|1\d)', lock)


def test_the_build_script_bundles_the_extra_and_checks_it_imports():
    text = SCRIPT.read_text()
    assert 'SYNC_HINT="uv sync --extra mlx --extra llm --extra diarize"' in text
    check = text.index("find_spec('pyannote.audio')")
    bundle = text.index("-m listn.bundle")
    imports = text.index("speakers --check")
    signing = text.index("codesign --force --sign \"$IDENTITY\" --timestamp=none")
    assert check < bundle < imports < signing
    assert "LISTN_WITHOUT_SPEAKERS" in text


def test_the_bundle_carries_pyannote_and_signs_torch_libraries(tmp_path):
    prefix, venv = _uv_layout(tmp_path)
    site = venv / "lib" / "python3.12" / "site-packages"
    (site / "pyannote" / "audio" / "pipelines").mkdir(parents=True)
    (site / "pyannote" / "audio" / "__init__.py").write_text("__version__ = '4.0.7'\n")
    (site / "pyannote" / "audio" / "pipelines" / "speaker_diarization.py").write_text("x = 1\n")
    (site / "torch" / "lib").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("")
    (site / "torch" / "lib" / "libtorch_cpu.dylib").write_bytes(MACHO)
    (site / "torch" / "_C.cpython-312-darwin.so").write_bytes(MACHO)
    project = tmp_path / "project"
    (project / "listn" / "web").mkdir(parents=True)
    (project / "listn" / "__init__.py").write_text('__version__ = "0.5.0"\n')

    dest = tmp_path / "python"
    assemble(venv, project, dest)
    site_dest = dest / "prefix" / "lib" / "python3.12" / "site-packages"
    assert (site_dest / "pyannote" / "audio" / "pipelines" / "speaker_diarization.py").exists()
    machos = (dest / "macho.txt").read_text().split()
    prefix_site = "prefix/lib/python3.12/site-packages"
    assert f"{prefix_site}/torch/lib/libtorch_cpu.dylib" in machos
    assert f"{prefix_site}/torch/_C.cpython-312-darwin.so" in machos
    size = (dest / "SIZE.txt").read_text()
    assert "pyannote speaker models" in size and "not included" in size


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def build_tree(tmp_path):
    """A copy of macos/ with stand-ins for uname, swift, codesign and the venv's Python."""
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    root = tmp_path / "repo"
    (root / "macos" / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "macos" / "scripts" / "build-app.sh")
    shutil.copy(ROOT / "macos" / "Info.plist", root / "macos" / "Info.plist")
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "log.txt"
    _stub(tools / "uname", 'echo Darwin\n')
    _stub(tools / "swift", f'mkdir -p .build/release && printf x > .build/release/ListnMenu && echo swift >> "{log}"\n')
    _stub(tools / "codesign", f'echo "codesign $*" >> "{log}"\n')
    launcher = (
        f'if [[ "$1" == "speakers" ]]; then echo "speakers-check" >> "{log}"; exit "${{FAKE_IMPORT:-0}}"; fi\n'
        'echo 0.5.0\n'
    )
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _stub(venv_bin / "python", f'''
if [[ "$1" == "-c" ]]; then echo "find-spec" >> "{log}"; exit "${{FAKE_SPEC:-0}}"; fi
if [[ "$2" == "listn.bundle" ]]; then
  dest="${{@: -1}}"
  mkdir -p "$dest/bin"
  cat > "$dest/bin/listn" <<'EOF'
#!/bin/bash
{launcher}EOF
  chmod +x "$dest/bin/listn"
  echo "prefix/bin/python3" > "$dest/macho.txt"
  echo "bundle" >> "{log}"
fi
''')

    def run(**env):
        full = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "TMPDIR": str(tmp_path), **env}
        log.write_text("")
        done = subprocess.run(["bash", str(root / "macos" / "scripts" / "build-app.sh")],
                              env=full, capture_output=True, text=True, timeout=60)
        return done, log.read_text().split("\n"), root / "macos" / "Listn.app"

    return run


def _steps(lines):
    return [line.split(" ")[0] if not line.startswith("codesign --force --sign - --timestamp") else "sign-python"
            for line in lines if line]


def test_build_fails_early_without_the_extra(build_tree):
    done, lines, app = build_tree(FAKE_SPEC="1")
    assert done.returncode == 1
    assert "uv sync --extra mlx --extra llm --extra diarize" in done.stderr
    assert "LISTN_WITHOUT_SPEAKERS=1" in done.stderr
    assert "bundle" not in lines and not app.exists()


def test_build_checks_the_bundled_import_before_signing(build_tree):
    done, lines, app = build_tree()
    assert done.returncode == 0, done.stderr
    steps = _steps(lines)
    assert steps.index("find-spec") < steps.index("bundle") < steps.index("speakers-check") < steps.index("sign-python")
    assert "Speaker labels: pyannote.audio and torch are bundled" in done.stdout
    assert (app / "Contents" / "Resources" / "engine.path").read_text().strip() == "python/bin/listn"


def test_build_stops_when_pyannote_does_not_import_from_the_bundle(build_tree):
    done, lines, app = build_tree(FAKE_IMPORT="1")
    assert done.returncode == 1 and "does not import from the bundled Python" in done.stderr
    assert "sign-python" not in _steps(lines) and not app.exists()


def test_build_without_speakers_is_still_possible(build_tree):
    done, lines, app = build_tree(FAKE_SPEC="1", LISTN_WITHOUT_SPEAKERS="1")
    assert done.returncode == 0, done.stderr
    assert "speakers-check" not in lines and "find-spec" not in lines
    assert app.exists()
