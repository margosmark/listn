"""The capture rig script (``scripts/linux_capture_rig.sh``).

The full rig needs a real PipeWire daemon and takes ~10 s, so the default
run only checks the script's syntax and interface. Set ``LISTN_RIG_IT=1``
to run the whole rig end to end — that is the one-command gate
``docs/LINUX_CAPTURE.md`` quotes:

    LISTN_RIG_IT=1 uv run pytest tests/test_capture_rig.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RIG = REPO / "scripts" / "linux_capture_rig.sh"
bash = shutil.which("bash") or ""  # tests are skipped when bash is absent

pytestmark = pytest.mark.skipif(not bash, reason="bash not available")


def test_rig_script_exists_and_is_executable():
    assert RIG.is_file()
    assert os.access(RIG, os.X_OK), "scripts/linux_capture_rig.sh must be executable"


def test_rig_script_syntax():
    subprocess.run([bash, "-n", str(RIG)], check=True)


def test_rig_script_help_documents_exit_codes():
    proc = subprocess.run([bash, str(RIG), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Exit codes" in proc.stdout
    assert "--duration" in proc.stdout


def test_rig_script_rejects_unknown_arguments():
    proc = subprocess.run([bash, str(RIG), "--bogus"], capture_output=True, text=True)
    assert proc.returncode == 3
    assert "unknown argument" in proc.stderr


@pytest.mark.skipif(
    not os.getenv("LISTN_RIG_IT"),
    reason="set LISTN_RIG_IT=1 to run the full PipeWire daemon rig (~10 s)",
)
def test_rig_end_to_end_both_tracks_ok():
    """The whole rig: daemon, sink, tunnel mic, tone, recorder, teardown.

    Asserts the honesty contract the script encodes: exit 0 only when both
    tracks report verdict=ok with a 440 Hz tone present in each WAV.
    """
    proc = subprocess.run([bash, str(RIG)], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert out.count("verdict=ok") == 2, out
    assert "dominant=440.0 Hz" in out, out
    assert "rig result: OK" in out, out
