"""Session notes: a summary, decisions, and action items.

Notes are stored in ``sessions/<id>/summary.json`` and can be edited.
Generation prefers the local MLX model and falls back to a deterministic
extractive pass so a machine without MLX still gets notes from the transcript.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable

from listn.config import Settings
from listn.library import Library, Segment
from listn.llm import DEFAULT_LLM, generate, mlx_lm_available

log = logging.getLogger("listn")

_DECISION = re.compile(
    r"\b(decid(?:e|ed|ing)|agreed|agree to|we(?:'ll| will) go|the plan is|decision is)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(i(?:'ll| will)|we(?:'ll| will)|can you|could you|please|action item|to-?do|follow up|"
    r"send|email|schedule|i(?:'m| am) going to)\b",
    re.IGNORECASE,
)


def speaker_name(segment: Segment, speakers: dict | None) -> str:
    if not segment.speaker:
        return "Others"
    return (speakers or {}).get(segment.speaker) or segment.speaker


def transcript_lines(segments: list[Segment], speakers: dict | None) -> list[str]:
    lines = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text or text == "[paused]":
            continue
        lines.append(f"[{speaker_name(segment, speakers)}] {text}")
    return lines


def extractive_notes(segments: list[Segment], speakers: dict | None) -> dict:
    """Pull a short recap, decisions, and commitments without a model."""
    lines = transcript_lines(segments, speakers)
    decisions: list[str] = []
    actions: list[dict] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        who = speaker_name(segment, speakers)
        if _DECISION.search(text) and text not in decisions:
            decisions.append(text)
        if _ACTION.search(text):
            actions.append({"owner": who, "text": text})
    if not lines:
        summary = "No speech was transcribed."
    elif len(lines) <= 4:
        summary = " ".join(lines)
    else:
        summary = " ".join(lines[:3]) + " … " + lines[-1]
    return {
        "summary": summary[:2000],
        "decisions": decisions[:12],
        "actions": actions[:24],
        "model": None,
        "backend": "extractive",
        "edited": False,
    }


def _as_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict) and str(item.get("text") or "").strip():
            out.append(str(item["text"]).strip())
    return out[:12]


def _as_actions(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append({"owner": "Others", "text": item.strip()})
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            owner = str(item.get("owner") or "Others").strip() or "Others"
            out.append({"owner": owner[:80], "text": text[:500]})
    return out[:24]


def parse_model_notes(raw: str) -> dict | None:
    """Pull the JSON object out of a model reply, including fenced blocks."""
    text = raw or ""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fenced.group(1) if fenced else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            blob = text[start:end + 1]
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return None
    return {
        "summary": summary[:4000],
        "decisions": _as_str_list(data.get("decisions")),
        "actions": _as_actions(data.get("actions")),
    }


def prompt_for(segments: list[Segment], speakers: dict | None) -> str:
    body = "\n".join(transcript_lines(segments, speakers))
    return (
        "You are taking notes on a meeting transcript. Speakers are labelled in brackets. "
        "Me is the person who owns this computer; everyone else is Others or a named speaker. "
        "Attribute each action item to the person who committed to it, or who was asked to do it.\n"
        "Reply with JSON only, no markdown, in this shape:\n"
        '{"summary":"2-4 sentences","decisions":["..."],'
        '"actions":[{"owner":"Me","text":"..."}]}\n\n'
        f"Transcript:\n{body}"
    )


def build_notes(
    segments: list[Segment],
    speakers: dict | None,
    model_id: str = DEFAULT_LLM,
    complete: Callable[[str], str] | None = None,
) -> dict:
    """Notes for one transcript. ``complete`` overrides the MLX call (tests)."""
    fallback = extractive_notes(segments, speakers)
    if not transcript_lines(segments, speakers):
        return fallback
    runner = complete
    if runner is None and mlx_lm_available():
        runner = lambda prompt: generate(prompt, model_id)  # noqa: E731
    if runner is None:
        return fallback
    try:
        parsed = parse_model_notes(runner(prompt_for(segments, speakers)))
    except Exception:
        log.exception("Local notes model failed; using the extractive fallback")
        return fallback
    if not parsed:
        fallback["backend"] = "extractive"
        return fallback
    parsed["model"] = model_id
    parsed["backend"] = "mlx" if complete is None else "model"
    parsed["edited"] = False
    return parsed


def write_notes(library: Library, session_id: str, settings: Settings,
                complete: Callable[[str], str] | None = None) -> dict:
    meta = library.get(session_id)
    notes = build_notes(library.segments(session_id), meta.speakers, settings.llm_model, complete)
    library.save_notes(session_id, notes)
    return notes


def schedule_notes(library: Library, session_id: str, settings: Settings, bus=None) -> None:
    if not settings.summarize_after:
        return

    def run() -> None:
        try:
            notes = write_notes(library, session_id, settings)
        except Exception:
            log.exception("Could not write notes for %s", session_id)
            return
        if bus is not None:
            bus.publish({"type": "summary.ready", "session": session_id, "notes": notes})

    threading.Thread(target=run, name="listn-notes", daemon=True).start()


def edited_notes(summary: str, decisions: list[str], actions: list[dict], previous: dict | None) -> dict:
    base = dict(previous or {})
    base["summary"] = (summary or "").strip()[:4000]
    base["decisions"] = _as_str_list(decisions)
    base["actions"] = _as_actions(actions)
    base["edited"] = True
    base.setdefault("backend", "edited")
    return base
