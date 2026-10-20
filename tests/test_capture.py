import pytest

from listn.capture import DeviceError, measure_levels, open_input

from .conftest import MAC_DEVICES, FakeSD


def test_aggregate_opens_with_all_channels():
    sd = FakeSD(MAC_DEVICES)
    opened = open_input("Aggregate", lambda *a: None, sd=sd)
    opened.stream.stop()
    assert (opened.channels, opened.samplerate) == (3, 48000.0)


def test_falls_back_when_driver_rejects_channel_count():
    sd = FakeSD(MAC_DEVICES, reject=lambda ch, rate: ch == 3)
    opened = open_input("Aggregate", lambda *a: None, sd=sd)
    opened.stream.stop()
    assert opened.channels == 2


def test_falls_back_to_other_rate():
    sd = FakeSD(MAC_DEVICES, reject=lambda ch, rate: rate == 48000.0)
    opened = open_input("BlackHole", lambda *a: None, sd=sd)
    opened.stream.stop()
    assert opened.samplerate == 44100.0


def test_unopenable_device_gives_guidance():
    sd = FakeSD(MAC_DEVICES, reject=lambda ch, rate: True)
    with pytest.raises(DeviceError, match="Audio MIDI Setup"):
        open_input("Aggregate", lambda *a: None, sd=sd)


def test_unknown_device_is_device_error():
    with pytest.raises(DeviceError, match="No input device matching"):
        open_input("Zoom Audio", lambda *a: None, sd=FakeSD(MAC_DEVICES))


def test_measure_levels_ok_and_silent():
    sd = FakeSD(MAC_DEVICES)
    report = measure_levels("BlackHole", seconds=0.3, sd=sd)
    assert report["verdict"] == "ok"
    assert report["per_channel"][0]["peak_db"] > -15
    assert report["per_channel"][1]["peak_db"] == -120.0

    sd.amplitude = 0.0
    silent = measure_levels("BlackHole", seconds=0.3, sd=sd)
    assert silent["verdict"] == "silent" and "Multi-Output" in silent["advice"]
