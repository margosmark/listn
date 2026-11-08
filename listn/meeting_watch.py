"""Linux meeting watcher: polling state machine + prompt contract (L3a).

The watcher is pure state: classified hits in, prompt events out. It never
starts anything by itself — ``auto_record`` is off unless the user turned
it on in settings (default ``False``), and even then the start is an
explicit *event* the tray controller executes against the same loopback
start route a human click would use. With auto-record off, ``poll()``
emits only offers; recording begins solely from an accepted click.

Occurrence semantics (mirrors the macOS ``MeetingWatch``): one offer per
occurrence of an app identity, deduped across polls; the offer can be
dismissed (quiet until the meeting ends) or snoozed (asks again later);
when an accepted/tracked meeting disappears the watcher emits an end
offer — or an auto-stop when auto-record is on.

The notification/session metadata contract shared with L2
(captions/hotkeys, task t_7c59bc5e) is ``MeetingPrompt.to_dict()``: a
fixed five-key dict, no window titles, no attendee/calendar content, no
PII beyond the app label. L2 should consume this shape rather than adding
meeting routes; L3a adds no server routes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

SNOOZE_SECONDS = 300.0

PROMPT_KINDS = ("offer", "auto_start", "end", "auto_end")


@dataclass(frozen=True)
class MeetingPrompt:
    """Stable prompt/session metadata. Fixed keys — no titles, ever."""

    kind: str             # offer | auto_start | end | auto_end
    meeting_id: str       # "<source>:<identity>", e.g. "pipewire:zoom"
    label: str            # "Zoom" | "Browser meeting" | "Slack huddle"
    source: str           # "pipewire" | "x11"
    title: str            # suggested recording title (the label)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "meeting_id": self.meeting_id,
                "label": self.label, "source": self.source,
                "title": self.title}

    @classmethod
    def from_dict(cls, data: dict) -> "MeetingPrompt":
        return cls(kind=str(data["kind"]), meeting_id=str(data["meeting_id"]),
                   label=str(data["label"]), source=str(data["source"]),
                   title=str(data["title"]))


@dataclass
class _Occurrence:
    state: str = "new"            # new | offered | dismissed | snoozed | consumed
    snoozed_until: float | None = None


class MeetingWatcher:
    """Poll dedup + consent state for one running tray."""

    def __init__(self, *, offer: bool = True, auto_record: bool = False,
                 snooze_seconds: float = SNOOZE_SECONDS, clock=None):
        self.offer = offer
        self.auto_record = auto_record
        self.snooze_seconds = float(snooze_seconds)
        self._clock = clock or time.monotonic
        self._present: dict[str, _Occurrence] = {}
        self._hits: dict[str, object] = {}
        self._tracked: str | None = None

    # -- observations -----------------------------------------------------

    @property
    def tracked_id(self) -> str | None:
        return self._tracked

    def present_ids(self) -> set[str]:
        return set(self._present)

    def poll(self, hits, now: float | None = None) -> list[MeetingPrompt]:
        """Advance with one poll's worth of classified hits.

        Emits at most one event per identity per occurrence; a hit that
        stays present never re-offers until it disappears (dedup), is
        dismissed, or its snooze expires.
        """
        now = self._clock() if now is None else float(now)
        events: list[MeetingPrompt] = []
        now_ids: dict[str, object] = {}
        for hit in hits:
            now_ids[hit.meeting_id] = hit

        for meeting_id, hit in now_ids.items():
            occ = self._present.get(meeting_id)
            if occ is None:
                occ = _Occurrence()
                self._present[meeting_id] = occ
            self._hits[meeting_id] = hit
            if meeting_id == self._tracked or not self.offer:
                continue
            if occ.state == "new":
                events.append(self._offer_or_start(hit, occ))
            elif occ.state == "snoozed" and occ.snoozed_until is not None \
                    and now >= occ.snoozed_until:
                events.append(self._offer_or_start(hit, occ))

        for meeting_id in [m for m in self._present if m not in now_ids]:
            hit = self._hits.pop(meeting_id, None)
            self._present.pop(meeting_id, None)
            if hit is None:
                continue
            if meeting_id == self._tracked:
                self._tracked = None
                kind = "auto_end" if self.auto_record else "end"
                events.append(self._prompt(kind, hit))
        return events

    def _offer_or_start(self, hit, occ: _Occurrence) -> MeetingPrompt:
        if self.auto_record:
            occ.state = "consumed"
            self._tracked = hit.meeting_id
            return self._prompt("auto_start", hit)
        occ.state = "offered"
        occ.snoozed_until = None
        return self._prompt("offer", hit)

    @staticmethod
    def _prompt(kind: str, hit) -> MeetingPrompt:
        return MeetingPrompt(kind=kind, meeting_id=hit.meeting_id,
                             label=hit.label, source=hit.source,
                             title=hit.label)

    # -- user actions (consent) ------------------------------------------

    def accept(self, meeting_id: str) -> bool:
        """Consent: record this meeting; no more offers, end offer on exit."""
        occ = self._present.get(meeting_id)
        if occ is None:
            return False
        occ.state = "consumed"
        occ.snoozed_until = None
        self._tracked = meeting_id
        return True

    def dismiss(self, meeting_id: str) -> bool:
        """Not now: quiet for the rest of this occurrence."""
        occ = self._present.get(meeting_id)
        if occ is None:
            return False
        occ.state = "dismissed"
        occ.snoozed_until = None
        return True

    def snooze(self, meeting_id: str, now: float | None = None) -> bool:
        """Ask again after the snooze window, while the meeting continues."""
        occ = self._present.get(meeting_id)
        if occ is None:
            return False
        now = self._clock() if now is None else float(now)
        occ.state = "snoozed"
        occ.snoozed_until = now + self.snooze_seconds
        return True
