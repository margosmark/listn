"""L3a: Linux meeting detection sources, polling watcher, tray prompt UX.

Covers the acceptance list from the parity spec:

* supported meeting apps detected from a real PipeWire fixture;
* browser meetings and Slack huddles only with a title hint (the macOS
  rule, reused from ``listn.meetings``);
* false positives (playback-only streams, ordinary browser titles);
* polling dedup — one offer per occurrence, however many polls;
* no recording start without consent (auto-record off by default);
* dismissal / snooze / end offer;
* prompt metadata carries no window titles.
"""

from __future__ import annotations

import json

from listn.config import Settings
from listn.meeting_sources import Hit, detect, normalize_identity, x11_window_titles
from listn.meeting_watch import MeetingPrompt, MeetingWatcher
from listn.tray import (
    ITEM_MEETING_DISMISS, ITEM_MEETING_END, ITEM_MEETING_RECORD,
    ITEM_MEETING_SNOOZE, ITEM_START, TrayController, build_menu,
)


# ---------------------------------------------------------------------------
# PipeWire fixtures (shape of real `pw-dump` output)
# ---------------------------------------------------------------------------

def node(media_class, *, app_id=None, binary=None, app_name=None,
         media_name=None, node_name="stream"):
    props = {"media.class": media_class, "node.name": node_name}
    if app_id:
        props["application.id"] = app_id
    if binary:
        props["application.process.binary"] = binary
    if app_name:
        props["application.name"] = app_name
    if media_name:
        props["media.name"] = media_name
    return {"id": 7, "type": "PipeWire:Interface:Node",
            "info": {"props": props}}


def zoom_call():
    return [node("Stream/Input/Audio", app_id="us.zoom.xos",
                 binary="zoom", app_name="zoom")]


# ---------------------------------------------------------------------------
# Detection: supported apps, browser meetings, false positives
# ---------------------------------------------------------------------------

def test_supported_meeting_apps_detected_from_mic_capture():
    payload = [
        node("Stream/Input/Audio", app_id="us.zoom.xos", binary="zoom"),
        node("Stream/Input/Audio", app_id="com.microsoft.teams2",
             binary="teams"),
        node("Stream/Input/Audio", app_id="com.webex.meetingmanager",
             binary="webex"),
    ]
    hits = detect(payload)
    assert [h.label for h in hits] == ["Zoom", "Teams", "Webex"]
    assert all(h.source == "pipewire" for h in hits)
    assert hits[0].meeting_id.startswith("pipewire:")


def test_playback_only_stream_is_not_a_meeting():
    """False positive: an app merely *playing* audio is not in a call."""
    payload = [node("Stream/Output/Audio", app_id="us.zoom.xos",
                    binary="zoom")]
    assert detect(payload) == []


def test_browser_meeting_needs_a_window_title_hint():
    payload = [node("Stream/Input/Audio", binary="firefox",
                    app_name="Firefox")]
    meeting = detect(payload, titles={"firefox": ["Weekly — Google Meet"]})
    assert [h.label for h in meeting] == ["Browser meeting"]

    ordinary = detect(payload, titles={"firefox": ["A song on YouTube"]})
    assert ordinary == []

    # No title source (Wayland, xprop missing): no browser claim at all.
    assert detect(payload) == []


def test_slack_huddle_needs_a_huddle_title():
    payload = [node("Stream/Input/Audio", app_id="com.slack.Slack",
                    binary="slack", app_name="Slack")]
    huddle = detect(payload, titles={"slack": ["Design huddle"]})
    assert [h.label for h in huddle] == ["Slack huddle"]
    assert detect(payload, titles={"slack": ["general"]}) == []


def test_identity_normalization_is_stable():
    assert normalize_identity("/usr/lib/firefox/firefox") == "firefox"
    assert normalize_identity("Google-chrome.desktop") == "google-chrome"
    assert normalize_identity(None) == ""


def test_x11_title_source_is_absent_without_a_display():
    # Wayland session, no XWayland display: the source reports unavailable
    # instead of guessing or scraping an unsupported window API.
    assert x11_window_titles(env={"XDG_SESSION_TYPE": "wayland"}) == {}
    assert x11_window_titles(env={"DISPLAY": ""}) == {}


# ---------------------------------------------------------------------------
# Watcher: dedup, consent, dismiss/snooze, end offer
# ---------------------------------------------------------------------------

def hit(label="Zoom", identity="zoom", source="pipewire"):
    return Hit(identity=identity, label=label, source=source,
               app_name="zoom")


def test_poll_offers_once_per_occurrence():
    watcher = MeetingWatcher()
    first = watcher.poll([hit()], now=0.0)
    assert [e.kind for e in first] == ["offer"]
    # The same meeting on every later poll: no repeat prompts.
    assert watcher.poll([hit()], now=1.0) == []
    assert watcher.poll([hit()], now=2.0) == []
    # It goes away and comes back: a new occurrence, so a new offer.
    assert watcher.poll([], now=3.0) == []
    assert [e.kind for e in watcher.poll([hit()], now=4.0)] == ["offer"]


def test_auto_record_is_off_by_default():
    assert MeetingWatcher().auto_record is False
    assert Settings().auto_record_meetings is False


def test_auto_record_starts_only_when_enabled():
    watcher = MeetingWatcher(auto_record=True)
    events = watcher.poll([hit()], now=0.0)
    assert [e.kind for e in events] == ["auto_start"]
    assert watcher.poll([hit()], now=1.0) == []
    # and the tracked meeting ending auto-stops
    assert [e.kind for e in watcher.poll([], now=2.0)] == ["auto_end"]


def test_manual_offer_never_starts_and_accept_tracks_the_meeting():
    watcher = MeetingWatcher()
    watcher.poll([hit()], now=0.0)
    assert watcher.tracked_id is None
    watcher.accept("pipewire:zoom")
    assert watcher.tracked_id == "pipewire:zoom"
    # accepted: no repeated offers for this occurrence
    assert watcher.poll([hit()], now=1.0) == []
    # and the tracked meeting ending offers a stop instead of auto-stopping
    assert [e.kind for e in watcher.poll([], now=2.0)] == ["end"]


def test_dismiss_suppresses_until_the_meeting_ends():
    watcher = MeetingWatcher()
    watcher.poll([hit()], now=0.0)
    watcher.dismiss("pipewire:zoom")
    assert watcher.poll([hit()], now=1.0) == []
    assert watcher.poll([hit()], now=2.0) == []
    assert watcher.poll([], now=3.0) == []
    # a new occurrence of the same app asks again
    assert [e.kind for e in watcher.poll([hit()], now=4.0)] == ["offer"]


def test_snooze_reoffers_after_the_snooze_expires():
    watcher = MeetingWatcher(snooze_seconds=300.0)
    watcher.poll([hit()], now=0.0)
    watcher.snooze("pipewire:zoom", now=0.0)
    assert watcher.poll([hit()], now=299.0) == []
    assert [e.kind for e in watcher.poll([hit()], now=301.0)] == ["offer"]


def test_prompt_metadata_never_carries_window_titles():
    watcher = MeetingWatcher()
    prompt = watcher.poll([hit()], now=0.0)[0]
    payload = prompt.to_dict()
    assert set(payload) == {"kind", "meeting_id", "label", "source", "title"}
    assert payload["kind"] == "offer"
    assert payload["label"] == "Zoom"
    # the contract is a small stable dict for L2 (captions/notifications)
    assert isinstance(MeetingPrompt.from_dict(payload), MeetingPrompt)


# ---------------------------------------------------------------------------
# Tray: prompt menu, consent-gated start, end offer
# ---------------------------------------------------------------------------

class FakeClient:
    """Loopback client stand-in: records actions, serves real-shaped JSON."""

    def __init__(self):
        self.starts = []
        self.stops = 0
        self.settings_data = {"offer_meetings": True,
                              "auto_record_meetings": False}

    def live(self):
        return {"live": None, "pipewire": None}

    def devices(self):
        return {"sinks": [{"name": "sink", "description": "Speakers"}],
                "sources": [{"name": "mic", "description": "Mic"}],
                "defaults": {"sink": "sink", "source": "mic"}}

    def settings(self):
        return dict(self.settings_data)

    def start(self, system, mic, title=None):
        self.starts.append((system, mic, title))
        return {}

    def pause(self):
        return {}

    def resume(self):
        return {}

    def stop(self):
        self.stops += 1
        return {}


def controller_for(hits, tmp_path, watcher=None, client=None):
    client = client or FakeClient()
    watcher = watcher if watcher is not None else MeetingWatcher()
    ctl = TrayController(client, meeting_watcher=watcher,
                         meeting_source=lambda: list(hits),
                         targets_file=tmp_path / "targets.json")
    return ctl, client, watcher


def test_detection_offers_a_tray_prompt_but_never_starts(tmp_path):
    ctl, client, _ = controller_for([hit()], tmp_path)
    ctl.refresh()
    prompt = ctl.meeting_prompt
    assert prompt is not None and prompt.kind == "offer"
    assert client.starts == []              # no start without consent

    labels = {i.id: i for i in ctl.menu()}
    assert ITEM_MEETING_RECORD in labels
    assert labels[ITEM_MEETING_RECORD].label == "Record Zoom?"
    assert labels[ITEM_MEETING_RECORD].enabled
    assert ITEM_MEETING_DISMISS in labels
    assert ITEM_MEETING_SNOOZE in labels

    # a second poll keeps the prompt but does not re-offer or start
    ctl.refresh()
    assert ctl.meeting_prompt == prompt
    assert client.starts == []


def test_record_click_starts_with_the_meeting_title(tmp_path):
    ctl, client, watcher = controller_for([hit()], tmp_path)
    ctl.refresh()
    ctl.action(ITEM_MEETING_RECORD)
    assert client.starts == [("sink", "mic", "Zoom")]
    assert watcher.tracked_id == "pipewire:zoom"
    assert ctl.meeting_prompt is None       # prompt consumed by consent


def test_dismiss_and_snooze_clear_the_prompt_without_starting(tmp_path):
    ctl, client, _ = controller_for([hit()], tmp_path)
    ctl.refresh()
    ctl.action(ITEM_MEETING_DISMISS)
    assert ctl.meeting_prompt is None and client.starts == []
    ctl.refresh()
    assert ctl.meeting_prompt is None       # dismissed: stays quiet

    ctl.action(ITEM_MEETING_SNOOZE)
    assert ctl.meeting_prompt is None and client.starts == []


def test_end_offer_appears_when_the_meeting_goes_away(tmp_path):
    hits = [hit()]
    ctl, client, watcher = controller_for(hits, tmp_path)
    ctl.refresh()
    ctl.action(ITEM_MEETING_RECORD)         # consent -> recording
    assert client.starts

    hits.clear()                            # meeting ended
    ctl.refresh()
    prompt = ctl.meeting_prompt
    assert prompt is not None and prompt.kind == "end"
    items = {i.id: i for i in ctl.menu()}
    assert ITEM_MEETING_END in items
    assert items[ITEM_MEETING_END].label == "Zoom ended — stop recording"

    ctl.action(ITEM_MEETING_END)
    assert client.stops == 1
    assert ctl.meeting_prompt is None


def test_auto_record_setting_starts_without_a_prompt(tmp_path):
    watcher = MeetingWatcher(auto_record=True)
    client = FakeClient()
    client.settings_data["auto_record_meetings"] = True
    ctl, _, _ = controller_for([hit()], tmp_path, watcher=watcher,
                               client=client)
    ctl.refresh()
    assert client.starts == [("sink", "mic", "Zoom")]
    assert ctl.meeting_prompt is None       # auto-record is not an offer


def test_menu_without_a_prompt_has_no_meeting_controls(tmp_path):
    ctl, _, _ = controller_for([], tmp_path)
    ctl.refresh()
    labels = " ".join(i.label for i in ctl.menu()).lower()
    for forbidden in ("record zoom", "meeting", "auto-record", "snooze",
                      "remind me"):
        assert forbidden not in labels, labels
    assert ITEM_MEETING_RECORD not in {i.id for i in ctl.menu()}


def test_prompt_items_render_through_the_menu_builder():
    prompt = MeetingPrompt(kind="offer", meeting_id="pipewire:zoom",
                           label="Zoom", source="pipewire", title="Zoom")
    items = build_menu(TrayController(FakeClient(),
                                      meeting_watcher=MeetingWatcher(),
                                      meeting_source=lambda: []).status,
                       [], [], None, None, meeting_prompt=prompt)
    ids = {i.id for i in items if not i.separator}
    assert ITEM_MEETING_RECORD in ids and ITEM_MEETING_DISMISS in ids
    assert ITEM_MEETING_SNOOZE in ids


def test_pw_dump_fixture_shape_round_trips():
    """The fixture is real pw-dump shape: props nested under info."""
    payload = json.loads(json.dumps([
        node("Stream/Input/Audio", app_id="us.zoom.xos", binary="zoom")]))
    hits = detect(payload)
    assert len(hits) == 1 and isinstance(hits[0], Hit)
