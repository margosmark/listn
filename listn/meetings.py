"""Which apps count as a meeting, and which calendar event they overlap.

The menu bar app uses the same rules. Bundle ids are matched exactly.
Browsers count only when a window title looks like Meet, Zoom, Teams, or
Webex. Slack counts only for a huddle. The calendar pick is the shortest
event that contains the current time, which is usually the meeting rather
than an all-day hold.
"""

from __future__ import annotations

from datetime import datetime

MEETING_APPS = {
    "us.zoom.xos": "Zoom",
    "com.microsoft.teams2": "Teams",
    "com.microsoft.teams": "Teams",
    "com.cisco.webexmeetings": "Webex",
    "com.webex.meetingmanager": "Webex",
    "Cisco-Systems.Spark": "Webex",
    "com.apple.FaceTime": "FaceTime",
    "com.apple.facetime": "FaceTime",
}

BROWSERS = {
    "com.google.Chrome",
    "com.apple.Safari",
    "company.thebrowser.Browser",
    "com.brave.Browser",
    "com.microsoft.edgemac",
    "org.mozilla.firefox",
}

BROWSER_HINTS = ("google meet", "meet.google", "zoom meeting", "microsoft teams", "webex")
SLACK = "com.tinyspeck.slackmacgap"


def meeting_label(bundle_id: str, app_name: str = "", window_titles: list[str] | None = None) -> str | None:
    titles = " ".join(window_titles or []).lower()
    name = (app_name or "").lower()
    if bundle_id in MEETING_APPS:
        return MEETING_APPS[bundle_id]
    if bundle_id == SLACK and ("huddle" in titles or "huddle" in name):
        return "Slack huddle"
    if bundle_id in BROWSERS:
        blob = f"{titles} {name}"
        if any(hint in blob for hint in BROWSER_HINTS):
            return "Browser meeting"
    return None


def _when(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def pick_event(events: list[dict], moment: datetime) -> dict | None:
    """Shortest event that contains ``moment``. All-day holds lose to the meeting."""
    best = None
    best_length = None
    for event in events:
        try:
            start = _when(event["start"])
            end = _when(event["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= moment < end:
            length = (end - start).total_seconds()
            if best is None or length < best_length:
                best = event
                best_length = length
    return best


def title_and_attendees(typed_title: str | None, events: list[dict], moment: datetime) -> tuple[str | None, list[str]]:
    event = pick_event(events, moment)
    title = (typed_title or "").strip()
    attendees: list[str] = []
    if event is not None:
        if not title:
            title = str(event.get("title") or "").strip()
        for name in event.get("attendees") or []:
            text = str(name).strip()
            if text and text not in attendees:
                attendees.append(text[:80])
    return (title[:200] or None), attendees[:40]
