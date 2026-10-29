import pytest

from listn import devices as devmod

from .conftest import MAC_DEVICES, FakeSD, dev


def build(raw=MAC_DEVICES, default_input=0):
    return devmod.build_devices([dict(d, index=i) for i, d in enumerate(raw)], ["Core Audio"], default_input)


def test_classify():
    assert devmod.classify("BlackHole 2ch") == "blackhole"
    assert devmod.classify("BlackHole 16ch") == "blackhole"
    assert devmod.classify("Aggregate Device") == "aggregate"
    assert devmod.classify("Loopback Audio") == "loopback"
    assert devmod.classify("MacBook Pro Microphone") == "input"


def test_auto_select_prefers_blackhole():
    assert devmod.resolve(None, build()).name == "BlackHole 2ch"


def test_auto_select_falls_back_to_aggregate_then_default():
    no_bh = [d for d in MAC_DEVICES if "BlackHole" not in d["name"]]
    assert devmod.resolve(None, build(no_bh)).name == "Aggregate Device"
    mic_only = [dev("USB Mic", inputs=1), dev("Studio Mic", inputs=1)]
    assert devmod.resolve("", build(mic_only, default_input=1)).name == "Studio Mic"


def test_resolve_by_name_substring_and_index():
    devices = build()
    assert devmod.resolve("BlackHole 2ch", devices).index == 2
    assert devmod.resolve("blackhole", devices).index == 2
    assert devmod.resolve("4", devices).name == "Aggregate Device"


def test_resolve_errors_are_actionable():
    devices = build()
    with pytest.raises(LookupError, match="output device"):
        devmod.resolve(1, devices)
    with pytest.raises(LookupError, match="No device with index 42"):
        devmod.resolve(42, devices)
    with pytest.raises(LookupError, match="several"):
        devmod.resolve("Mac", build([dev("MacBook Mic A", 1), dev("MacBook Mic B", 1)]))
    with pytest.raises(LookupError, match="No input device matching"):
        devmod.resolve("Zoom", devices)
    with pytest.raises(LookupError, match="No audio input"):
        devmod.resolve(None, build([dev("Speakers", inputs=0, outputs=2)]))


def test_name_survives_index_shift():
    shifted = [dev("AirPods", inputs=1)] + MAC_DEVICES
    assert devmod.resolve("BlackHole 2ch", build(shifted)).index == 3


def test_candidate_configs_try_native_first():
    agg = build()[4]
    configs = devmod.candidate_stream_configs(agg)
    assert configs[0] == (3, 48000.0)
    assert (2, 48000.0) in configs and (1, 16000.0) in configs
    mono = build()[0]
    assert {c for c, _ in devmod.candidate_stream_configs(mono)} == {1}


def _titles(notes):
    return [n["title"] for n in notes]


def test_guidance_missing_blackhole(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    raw = [dev("MacBook Pro Microphone", 1), dev("MacBook Pro Speakers", 0, 2)]
    notes = devmod.setup_guidance(build(raw), "MacBook Pro Speakers")
    assert "No BlackHole needed" in _titles(notes)
    assert notes[0]["level"] == "ok"


def test_guidance_missing_multi_output(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    raw = [dev("MacBook Pro Microphone", 1), dev("BlackHole 2ch", 2, 2)]
    notes = devmod.setup_guidance(build(raw), "BlackHole 2ch")
    assert "No Multi-Output Device found" in _titles(notes)


def test_guidance_output_not_routed(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    notes = devmod.setup_guidance(build(), "MacBook Pro Speakers")
    assert any("System output is" in t for t in _titles(notes))


def test_guidance_ready(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    notes = devmod.setup_guidance(build(), "Multi-Output Device")
    assert [n["level"] for n in notes] == ["ok"]


def test_list_inputs_from_fake_sounddevice():
    sd = FakeSD(MAC_DEVICES, default=(0, 3))
    inputs = devmod.list_inputs(sd)
    assert [d.name for d in inputs] == ["MacBook Pro Microphone", "BlackHole 2ch", "Aggregate Device"]
    assert inputs[0].is_default_input
    assert devmod.default_output_name(sd) == "Multi-Output Device"
