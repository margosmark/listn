import sys
from fastapi.testclient import TestClient
from listn import devices
from listn.server import create_app


def test_linux_guidance_is_honest_about_single_track(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    inputs = devices.build_devices([{'name': 'Microphone', 'max_input_channels': 1}])
    text = ' '.join(n['detail'] for n in devices.setup_guidance(inputs, None))
    assert 'one input' in text
    assert 'separate' in text
    assert 'BlackHole' not in text


def test_status_exposes_platform(data_dir, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    with TestClient(create_app(), base_url='http://localhost') as client:
        assert client.get('/api/status').json()['platform'] == 'linux'


def test_linux_silent_level_advice_avoids_macos(fake_sd, monkeypatch):
    from listn.capture import measure_levels
    monkeypatch.setattr(sys, 'platform', 'linux')
    fake_sd.amplitude = 0
    report = measure_levels(None, seconds=0.01, sd=fake_sd)
    assert report['verdict'] == 'silent'
    assert 'BlackHole' not in report['advice']
    assert 'macOS' not in report['advice']
