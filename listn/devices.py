"""Audio device discovery, selection, and setup guidance.

Device indices shift whenever macOS adds or removes a device (plugging in
headphones, creating an aggregate device), so listn identifies devices by
name and resolves them to an index at the moment a stream is opened.

Functions here take plain device dicts (the shape sounddevice returns) so the
logic can be tested without audio hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower", "vb-cable", "aggregate")
MULTI_OUTPUT_HINTS = ("multi-output", "multi output", "blackhole")


@dataclass
class Device:
    index: int
    name: str
    inputs: int
    outputs: int
    default_samplerate: float
    hostapi: str
    kind: str
    is_default_input: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def classify(name: str) -> str:
    lower = name.lower()
    if "blackhole" in lower:
        return "blackhole"
    if "aggregate" in lower:
        return "aggregate"
    if any(h in lower for h in LOOPBACK_HINTS):
        return "loopback"
    return "input"


def _hostapi_names(sd) -> list[str]:
    try:
        return [api["name"] for api in sd.query_hostapis()]
    except Exception:
        return []


def _default_indices(sd) -> tuple[int, int]:
    try:
        default = sd.default.device
        return int(default[0]), int(default[1])
    except Exception:
        return -1, -1


def build_devices(raw: list[dict], hostapis: list[str] | None = None, default_input: int = -1) -> list[Device]:
    hostapis = hostapis or []
    devices = []
    for idx, dev in enumerate(raw):
        api_idx = dev.get("hostapi", 0)
        devices.append(
            Device(
                index=dev.get("index", idx),
                name=str(dev.get("name", f"Device {idx}")),
                inputs=int(dev.get("max_input_channels", 0)),
                outputs=int(dev.get("max_output_channels", 0)),
                default_samplerate=float(dev.get("default_samplerate") or 48000.0),
                hostapi=hostapis[api_idx] if 0 <= api_idx < len(hostapis) else "",
                kind=classify(str(dev.get("name", ""))),
                is_default_input=dev.get("index", idx) == default_input,
            )
        )
    return devices


def _sd():
    import sounddevice as sd

    return sd


def list_all(sd=None) -> list[Device]:
    sd = sd or _sd()
    default_in, _ = _default_indices(sd)
    return build_devices(list(sd.query_devices()), _hostapi_names(sd), default_in)


def list_inputs(sd=None) -> list[Device]:
    return [d for d in list_all(sd) if d.inputs > 0]


def default_output_name(sd=None) -> str | None:
    sd = sd or _sd()
    _, out_idx = _default_indices(sd)
    if out_idx < 0:
        return None
    try:
        return str(sd.query_devices(out_idx)["name"])
    except Exception:
        return None


def resolve(selector: str | int | None, devices: list[Device]) -> Device:
    """Find an input device by index, exact name, or case-insensitive substring.

    With no selector, prefer a BlackHole device, then an aggregate/loopback
    device, then the system default input.
    """
    inputs = [d for d in devices if d.inputs > 0]
    if not inputs:
        raise LookupError("No audio input devices found.")

    if selector is None or str(selector).strip() == "":
        for kind in ("blackhole", "aggregate", "loopback"):
            for d in inputs:
                if d.kind == kind:
                    return d
        for d in inputs:
            if d.is_default_input:
                return d
        return inputs[0]

    text = str(selector).strip()
    if text.lstrip("-").isdigit():
        idx = int(text)
        for d in devices:
            if d.index == idx:
                if d.inputs <= 0:
                    raise LookupError(f"Device {idx} ({d.name}) has no input channels; it is an output device.")
                return d
        raise LookupError(f"No device with index {idx}. Run `listn devices` to see what's available.")

    for d in inputs:
        if d.name == text:
            return d
    matches = [d for d in inputs if text.lower() in d.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(f"[{d.index}] {d.name}" for d in matches)
        raise LookupError(f"'{text}' matches several input devices: {names}. Use the full name or index.")
    raise LookupError(f"No input device matching '{text}'. Run `listn devices` to see what's available.")


def candidate_stream_configs(device: Device) -> list[tuple[int, float]]:
    """(channels, samplerate) pairs to try, most faithful first.

    Aggregate devices expose the sum of their members' channels (for example
    BlackHole 2ch + a mono mic = 3). Opening all of them keeps every source;
    we then mix down ourselves. Fallbacks cover drivers that reject the
    reported channel count or rate.
    """
    channel_opts = [device.inputs]
    for c in (2, 1):
        if c < device.inputs and c not in channel_opts:
            channel_opts.append(c)
    rate_opts = []
    for r in (device.default_samplerate, 48000.0, 44100.0, 16000.0):
        if r and r not in rate_opts:
            rate_opts.append(float(r))
    return [(c, r) for c in channel_opts for r in rate_opts]


def setup_guidance(devices: list[Device], output_name: str | None) -> list[dict]:
    """Human-readable checks about whether system-audio capture will work."""
    notes: list[dict] = []
    inputs = [d for d in devices if d.inputs > 0]
    has_blackhole = any(d.kind == "blackhole" for d in devices)
    has_capture = any(d.kind in ("blackhole", "aggregate", "loopback") for d in inputs)
    multi_output = any("multi-output" in d.name.lower() or "multi output" in d.name.lower() for d in devices)

    if not inputs:
        notes.append({
            "level": "error",
            "title": "No input devices found",
            "detail": "macOS did not report any audio inputs. Check System Settings > Privacy & Security > Microphone and allow your terminal app.",
        })
        return notes

    if not has_blackhole:
        notes.append({
            "level": "warn",
            "title": "BlackHole is not installed",
            "detail": "To capture system audio (Zoom, Meet, browser), install BlackHole with `brew install blackhole-2ch`, then reboot or restart Core Audio. Until then you can only transcribe a microphone.",
        })
    elif not multi_output:
        notes.append({
            "level": "warn",
            "title": "No Multi-Output Device found",
            "detail": "BlackHole is installed, but without a Multi-Output Device you can't hear audio while it's captured. Open Audio MIDI Setup, click +, choose Create Multi-Output Device, and tick your speakers/headphones and BlackHole 2ch.",
        })

    if has_blackhole and output_name is not None:
        lower = output_name.lower()
        if not any(h in lower for h in MULTI_OUTPUT_HINTS):
            notes.append({
                "level": "warn",
                "title": f"System output is '{output_name}'",
                "detail": "Audio is not being routed to BlackHole, so listn will hear silence. Set System Settings > Sound > Output (or your meeting app's speaker) to your Multi-Output Device.",
            })

    if has_capture and not notes:
        notes.append({
            "level": "ok",
            "title": "System audio capture looks ready",
            "detail": "Pick the BlackHole (or aggregate) input and use Test input to confirm you see signal.",
        })
    return notes
