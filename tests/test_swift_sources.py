"""Guards on the Swift capture code. There is no Swift toolchain in CI."""

import re
from pathlib import Path

SOURCES = Path(__file__).resolve().parent.parent / "macos" / "Sources" / "ListnMenu"


def read(name: str) -> str:
    return (SOURCES / name).read_text(encoding="utf-8")


def test_exclude_list_taps_keep_isexclusive():
    # 0.4.0 and 0.4.1 set isExclusive = false after the exclude-list
    # initializer, which taps only listn itself: every sample was zero.
    tap = read("SystemTap.swift")
    assert tap.count("isExclusive = false") == 1
    legacy = tap.index("case .legacyIncludeOnly:", tap.index("func makeDescription"))
    following = tap.index("case ", legacy + 1)
    assert legacy < tap.index("isExclusive = false") < following
    for name in SOURCES.glob("*.swift"):
        if name.name != "SystemTap.swift":
            assert "isExclusive = false" not in name.read_text(encoding="utf-8"), name.name


def test_the_old_configuration_is_never_used_for_recordings():
    tap = read("SystemTap.swift")
    start = tap.index("static func from(id:")
    body = tap[start:tap.index("\n    }\n", start)]
    assert "include-only-self" not in body
    assert "legacyIncludeOnly" not in read("CaptureEngine.swift")


def test_system_silence_warning_is_gated_on_playing_apps():
    engine = read("CaptureEngine.swift")
    assert re.search(r"if track == 0, let playing = playingNow, playing\.isEmpty", engine)
    menu = read("MenuBarController.swift")
    assert "system <= -119 && somethingPlaying" in menu
