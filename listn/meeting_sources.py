"""Linux meeting-detection sources: PipeWire app identity, X11 titles.

Which sources are reliable on Linux, and why (this is the L3a contract,
documented in ``docs/LINUX_MEETINGS.md``):

* **PipeWire capture streams** (``pw-dump``) work on X11 *and* Wayland and
  need no window permission: an app with a ``Stream/Input/Audio`` node is
  using the microphone, which is the same "audio is active" signal the
  macOS watcher gets from its process tap. Identity comes from
  ``application.id`` / ``application.process.binary`` / ``application.name``
  — the desktop-id style names PipeWire already exposes. Playback-only
  nodes (``Stream/Output/Audio``) are deliberately *not* evidence: an app
  merely playing audio is not in a call.
* **X11 window titles** (``xprop``, argv-only, no shell) are the one
  supported title API, and only on X11/XWayland. Native Wayland windows
  are not exposed to clients, so on Wayland browsers and Slack huddles
  simply are not detected — listn reports no hit rather than reaching for
  an unsupported/private window API (KWin scripting, foreign-toplevel
  protocols) or requiring a permission it cannot justify.

Privacy: titles and stream names are *classification inputs only*. They
are never logged, never persisted, and never appear in prompt metadata
(``MeetingPrompt.to_dict()`` carries label/source/id, never a title).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from listn.meetings import meeting_label


@dataclass(frozen=True)
class Hit:
    """One classified meeting occurrence source. No window titles on it."""

    identity: str
    label: str
    source: str          # "pipewire" | "x11"
    app_name: str

    @property
    def meeting_id(self) -> str:
        return f"{self.source}:{self.identity}"


def normalize_identity(raw) -> str:
    """Normalize an app identity to a stable, lowercase desktop-style id."""
    if not raw:
        return ""
    text = str(raw).strip().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    if text.lower().endswith(".desktop"):
        text = text[: -len(".desktop")]
    text = text.lower()
    return re.sub(r"[^a-z0-9._-]", "-", text)


def capture_evidence(payload) -> list[dict]:
    """App-identity evidence from ``pw-dump`` JSON: mic-capturing nodes only.

    Returns ``{identities, app_name, hints}`` dicts; ``identities`` are
    normalized candidates tried in order (application.id, binary, name).
    """
    out: list[dict] = []
    if not isinstance(payload, list):
        return out
    for obj in payload:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = obj.get("props") or (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") != "Stream/Input/Audio":
            continue
        candidates: list[str] = []
        for key in ("application.id", "application.process.binary",
                    "application.name"):
            ident = normalize_identity(props.get(key))
            if ident and ident not in candidates:
                candidates.append(ident)
        if not candidates:
            continue
        app_name = str(props.get("application.name")
                       or props.get("application.process.binary")
                       or candidates[0])
        hints = tuple(str(props[key]) for key in
                      ("media.name", "node.description", "media.role")
                      if props.get(key))
        out.append({"identities": candidates, "app_name": app_name,
                    "hints": hints})
    return out


def x11_window_titles(runner=subprocess.run, env=None) -> dict[str, list[str]]:
    """Window titles per app identity from X11 (``xprop``), or ``{}``.

    Empty whenever the source is unavailable: no display, no ``xprop``,
    a failed query. Argv-only — no shell ever sees a window id or title.
    """
    env = os.environ if env is None else env
    if not (env.get("DISPLAY") or "").strip():
        return {}
    if shutil.which("xprop") is None:
        return {}
    try:
        root = runner(["xprop", "-root", "_NET_CLIENT_LIST"],
                      capture_output=True, text=True, timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if getattr(root, "returncode", 1) != 0:
        return {}
    titles: dict[str, list[str]] = {}
    for wid in re.findall(r"0x[0-9a-fA-F]+", root.stdout or ""):
        try:
            out = runner(["xprop", "-id", wid, "_NET_WM_NAME", "WM_NAME",
                          "WM_CLASS"],
                         capture_output=True, text=True, timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if getattr(out, "returncode", 1) != 0:
            continue
        text = out.stdout or ""
        cls = re.search(r'WM_CLASS\([^)]*\)\s*=\s*"[^"]*",\s*"([^"]*)"', text)
        if not cls:
            continue
        identity = normalize_identity(cls.group(1))
        if not identity:
            continue
        title = (re.search(r'_NET_WM_NAME\([^)]*\)\s*=\s*"(.*)"', text)
                 or re.search(r'WM_NAME\([^)]*\)\s*=\s*"(.*)"', text))
        if title:
            titles.setdefault(identity, []).append(title.group(1))
    return titles


def detect(payload, titles: dict[str, list[str]] | None = None) -> list[Hit]:
    """Classify mic-capturing apps with the shared ``listn.meetings`` rules.

    Reuses the macOS rules verbatim: dedicated meeting apps match on
    identity alone (they only capture the mic during a call), browsers and
    Slack only when a title hint says meeting/huddle.
    """
    titles = titles or {}
    hits: list[Hit] = []
    seen: set[str] = set()
    for evidence in capture_evidence(payload):
        hints = list(evidence["hints"])
        for identity in evidence["identities"]:
            hints.extend(titles.get(identity, []))
        label = None
        matched = None
        for identity in evidence["identities"]:
            label = meeting_label(identity, evidence["app_name"], hints)
            if label:
                matched = identity
                break
        if not label or matched is None:
            continue
        hit = Hit(identity=matched, label=label, source="pipewire",
                  app_name=evidence["app_name"])
        if hit.meeting_id in seen:
            continue
        seen.add(hit.meeting_id)
        hits.append(hit)
    return hits


def read_pw_dump(runner=subprocess.run, timeout: float = 5.0):
    """Raw ``pw-dump`` JSON, or ``None`` when unavailable (never raises)."""
    if shutil.which("pw-dump") is None:
        return None
    try:
        proc = runner(["pw-dump"], capture_output=True, text=True,
                      timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    try:
        return json.loads(proc.stdout or "")
    except json.JSONDecodeError:
        return None


_last_collect = (0.0, [])


def collect_hits(min_interval: float = 2.0, clock=None) -> list[Hit]:
    """Default tray source: PipeWire + X11, throttled, never raises.

    Unavailable sources yield an empty list — detection degrades to "no
    meeting seen", never to a guess.
    """
    import time
    global _last_collect
    now = (clock or time.monotonic)()
    if now - _last_collect[0] < min_interval:
        return list(_last_collect[1])
    payload = read_pw_dump()
    hits = detect(payload if payload is not None else [],
                  titles=x11_window_titles())
    _last_collect = (now, hits)
    return list(hits)
