"""Session notes: a summary, key points, decisions, and action items.

Notes are stored in ``sessions/<id>/summary.json`` and can be edited.
Generation prefers the local MLX model and falls back to a deterministic
extractive pass so a machine without MLX still gets notes from the transcript.

How the model pass works:

* The transcript is folded into numbered speaker turns (``L12 [03:06] Alex: ...``)
  so the model can point at the line where each item was said. Those line
  numbers become ``start`` times that the UI links to the audio.
* Short meetings go to the model in one prompt. Long ones are split into
  overlapping parts (map), and the part notes are merged (reduce): the model
  writes the overall summary and key points from the part notes, and
  decisions and actions are merged and de-duplicated in code so none of their
  source times are lost.
* Each reply must be one JSON object. A reply that does not parse gets one
  retry before the extractive fallback is used.
* Owners are resolved against the speaker names: "Me" is the person who owns
  this Mac; SPEAKER_xx labels become their display names.

``summary.json`` stays backward compatible: ``summary`` (str), ``decisions``
(list of str) and ``actions`` (list of {owner, text}) are unchanged, and new
fields are added beside them (``key_points``, ``decision_items``,
``open_questions``, and ``start``/``due`` on each action).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from listn.config import Settings
from listn.library import Library, Segment
from listn.llm import DEFAULT_LLM, generate, mlx_lm_available, notes_backend

log = logging.getLogger("listn")

NOTES_VERSION = 2
SINGLE_PASS_CHARS = 24000   # about 6k tokens of transcript in one prompt
CHUNK_CHARS = 14000         # about 3.5k tokens per part for long meetings
CHUNK_OVERLAP_TURNS = 2
TURN_CHARS = 700            # merge consecutive lines of one speaker up to this
NOTES_MAX_TOKENS = 2000
MAX_KEY_POINTS = 8
MAX_DECISIONS = 12
MAX_ACTIONS = 20
MAX_QUESTIONS = 8

_DECISION = re.compile(
    r"\b(decid(?:e|ed|ing)|agreed|agree to|we(?:'ll| will) go with|the plan is|decision is|let's go with|"
    r"we(?:'re| are) going (?:to go )?with)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(i(?:'ll| will)|we(?:'ll| will)|can you|could you|please|action item|to-?do|follow up|"
    r"send|email|schedule|i(?:'m| am) going to)\b",
    re.IGNORECASE,
)
# Phrases that match _ACTION but are almost never a task.
_NOT_ACTION = re.compile(
    r"\b(i'll be honest|i'll tell you|i'll be the first|i'll be out|i'll catch you|catch you later|"
    r"step away|grab (?:some )?water|i'm going to say|i was going to say|be right back|"
    r"talk (?:to you|soon)|see you|thank)\b",
    re.IGNORECASE,
)
_FILLER = re.compile(r"^(thank you|thanks|yeah|yes|no|okay|ok|right|cool|bye|hmm|mm-?hmm|uh|um|so)[.!?, ]*$",
                     re.IGNORECASE)
_COMMITMENT = re.compile(
    r"\b(i(?:'ll| will| would| can)|we(?:'ll| will| can)|i(?:'m| am) (?:going to|gonna)|we(?:'re| are) (?:going to|gonna)|"
    r"gonna|let me|let's|can you|could you|would you|will you|please|need(?:s)? to|have to|has to|got to|"
    r"make sure|action item|to-?do|follow up|next step|by (?:monday|tuesday|wednesday|thursday|friday|tomorrow|"
    r"tonight|end of|next week|eod)|shall|should|must)\b",
    re.IGNORECASE,
)
_PLEASANTRY = re.compile(r"^(thank|appreciate|say (good)?bye|wave|greet|introduce (yourself|himself|herself))\b", re.IGNORECASE)
_FILLER_WORDS = set("thank you thanks yeah yes no okay ok right cool bye hmm mm hmm uh um so oh all good".split())
_ME_ALIASES = {"me", "i", "myself", "the user", "user", "owner", "me (the user)", "you"}
_VAGUE_DECISION = re.compile(r"\b(agreed|decided) to (discuss|talk|revisit|think|look into|continue)\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You write accurate, specific meeting notes from automatic speech-to-text transcripts. "
    "You only state what the transcript supports. You reply with one JSON object and nothing else."
)


@dataclass
class Turn:
    number: int
    start: float
    end: float
    speaker: str
    text: str


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


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_turns(segments: list[Segment], speakers: dict | None, max_chars: int = TURN_CHARS) -> list[Turn]:
    """Fold consecutive segments of one speaker into numbered turns."""
    turns: list[Turn] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text or text == "[paused]":
            continue
        who = speaker_name(segment, speakers)
        last = turns[-1] if turns else None
        if last and last.speaker == who and len(last.text) + len(text) < max_chars:
            last.text = f"{last.text} {text}"
            last.end = segment.end
        else:
            turns.append(Turn(len(turns) + 1, float(segment.start), float(segment.end), who, text))
    return turns


def participants(turns: list[Turn], speakers: dict | None) -> list[str]:
    seen: list[str] = []
    for turn in turns:
        if turn.speaker not in seen:
            seen.append(turn.speaker)
    return seen


# --------------------------------------------------------------------------- extractive


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def extractive_notes(segments: list[Segment], speakers: dict | None) -> dict:
    """Pull a short recap, decisions, and commitments without a model."""
    lines = transcript_lines(segments, speakers)
    decisions: list[dict] = []
    actions: list[dict] = []
    seen: set[str] = set()
    for segment in segments:
        text = (segment.text or "").strip()
        if not text or text == "[paused]":
            continue
        who = speaker_name(segment, speakers)
        words = _words(text)
        if len(words) < 4 or _NOT_ACTION.search(text):
            continue
        key = " ".join(words)
        if key in seen:
            continue
        if _DECISION.search(text) and len(decisions) < MAX_DECISIONS:
            decisions.append({"text": text, "start": round(float(segment.start), 2)})
            seen.add(key)
        if _ACTION.search(text) and len(words) >= 5 and len(actions) < MAX_ACTIONS:
            actions.append({"owner": who, "text": text, "due": None, "start": round(float(segment.start), 2)})
            seen.add(key)
    if not lines:
        summary = "No speech was transcribed."
    elif len(lines) <= 4:
        summary = " ".join(lines)
    else:
        turns = [t for t in build_turns(segments, speakers)
                 if len([w for w in _words(t.text) if w not in _FILLER_WORDS]) >= 3]
        best = sorted(turns, key=lambda t: len(_words(t.text)), reverse=True)[:4]
        picked = sorted(best, key=lambda t: t.start)
        parts = []
        for turn in picked:
            text = turn.text if len(turn.text) <= 240 else turn.text[:240].rsplit(" ", 1)[0] + " …"
            parts.append(f"[{turn.speaker}] {text}")
        summary = " ".join(parts) or " ".join(lines[:3])
    return {
        "summary": summary[:2000],
        "key_points": [],
        "decisions": [d["text"] for d in decisions],
        "decision_items": decisions,
        "actions": actions,
        "open_questions": [],
        "model": None,
        "backend": "extractive",
        "edited": False,
        "version": NOTES_VERSION,
    }


# --------------------------------------------------------------------------- parsing


def _as_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict) and str(item.get("text") or "").strip():
            out.append(str(item["text"]).strip())
    return out[:MAX_DECISIONS]


def _as_actions(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append({"owner": "Others", "text": item.strip()})
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("task") or "").strip()
            if not text:
                continue
            owner = str(item.get("owner") or "Others").strip() or "Others"
            entry = {"owner": owner[:80], "text": text[:500]}
            for key in ("due", "start", "line", "quote"):
                if item.get(key) not in (None, "", "null"):
                    entry[key] = item[key]
            out.append(entry)
    return out[:MAX_ACTIONS]


def _json_blob(raw: str):
    text = (raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for blob in candidates:
        for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob).replace("“", '"').replace("”", '"')):
            try:
                data = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def parse_model_notes(raw: str) -> dict | None:
    """Pull the notes object out of a model reply, including fenced blocks.

    Returns the loose structure (items may carry ``line``); grounding and
    owner resolution happen in ``finish_notes``.
    """
    data = _json_blob(raw)
    if not data:
        return None
    summary = data.get("summary") or data.get("overview") or ""
    if isinstance(summary, list):
        summary = " ".join(str(s) for s in summary)
    summary = str(summary).strip()
    if not summary:
        return None

    def items(key: str, limit: int) -> list[dict]:
        out = []
        for item in data.get(key) or []:
            if isinstance(item, str) and item.strip():
                out.append({"text": item.strip()})
            elif isinstance(item, dict) and str(item.get("text") or "").strip():
                entry = {"text": str(item["text"]).strip()[:500]}
                if item.get("line") is not None:
                    entry["line"] = item["line"]
                if str(item.get("quote") or "").strip():
                    entry["quote"] = str(item["quote"]).strip()[:300]
                out.append(entry)
        return out[:limit]

    return {
        "summary": summary[:4000],
        "key_points": items("key_points", MAX_KEY_POINTS * 2),
        "decisions": items("decisions", MAX_DECISIONS * 2),
        "actions": _as_actions(data.get("actions")),
        "open_questions": items("open_questions", MAX_QUESTIONS * 2),
    }


# --------------------------------------------------------------------------- grounding


def _line_number(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, list) and value:
        return _line_number(value[0])
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


def _similarity(a: str, b: str) -> float:
    wa, wb = set(_words(a)) - _STOP, set(_words(b)) - _STOP
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


_STOP = set("the a an and or to of in on for with at by is are was be will we i you it that this they "
            "he she them our my your their from as about into up out so do does did can could should would "
            "have has had not but if then than just also".split())


def ground(item: dict, turns: list[Turn], window: tuple[int, int] | None = None) -> Turn | None:
    """The turn an item came from: its ``line`` if valid, else the best text match."""
    by_number = {t.number: t for t in turns}
    item = {"text": item.get("quote") or item.get("text", ""), "line": item.get("line")}
    number = _line_number(item.get("line"))
    if number in by_number:
        candidate = by_number[number]
        # Accept the cited line, or a close neighbour that matches better.
        nearby = [by_number[n] for n in range(number - 2, number + 3) if n in by_number]
        best = max(nearby, key=lambda t: _similarity(item.get("text", ""), t.text))
        if _similarity(item.get("text", ""), best.text) > _similarity(item.get("text", ""), candidate.text) + 0.2:
            return best
        return candidate
    pool = turns
    if window:
        pool = [t for t in turns if window[0] <= t.number <= window[1]] or turns
    if not pool:
        return None
    best = max(pool, key=lambda t: _similarity(item.get("text", ""), t.text))
    return best if _similarity(item.get("text", ""), best.text) >= 0.35 else None


def resolve_owner(owner: str, names: list[str], source: Turn | None) -> str:
    raw = re.sub(r"\s+", " ", str(owner or "")).strip().strip(".")
    low = raw.lower()
    if low in _ME_ALIASES or low.startswith("me ") or low.startswith("me("):
        return "Me"
    for name in names:
        if low == name.lower():
            return name
    for name in names:  # "Sam (IT)" -> "Sam", "Sam Smith" -> "Sam"
        first = name.lower().split()[0] if name.strip() else ""
        if first and (low.startswith(first + " ") or low.startswith(first + "(") or low.split(" ")[0] == first):
            return name
    if not raw or low in {"unknown", "none", "null", "n/a", "someone", "tbd", "speaker", "unnamed user", "unknown speaker", "others", "the team", "team", "everyone", "all"}:
        return source.speaker if source else "Others"
    return raw[:80]


def _dedup(items: list[dict], key: str = "text", threshold: float = 0.75) -> list[dict]:
    out: list[dict] = []
    for item in items:
        dup = None
        for kept in out:
            if _similarity(item[key], kept[key]) >= threshold and (
                    "owner" not in item or item.get("owner") == kept.get("owner")):
                dup = kept
                break
        if dup is None:
            out.append(item)
        elif len(item[key]) > len(dup[key]):
            dup[key] = item[key]
            for extra in ("due",):
                if item.get(extra) and not dup.get(extra):
                    dup[extra] = item[extra]
    return out


def _clean_due(value) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none", "n/a", "not stated", "unspecified", "tbd", "unknown", "-"}:
        return None
    return text[:60]


def _quote_anywhere(quote: str, turns: list[Turn]) -> bool:
    return any(_similarity(quote, t.text) >= 0.9 for t in turns)


def finish_notes(parsed: dict, turns: list[Turn], window: tuple[int, int] | None = None) -> dict:
    """Ground items to turns, resolve owners, drop empties and duplicates."""
    names = participants(turns, None)

    def placed(entry: dict, src: Turn | None) -> dict:
        if src is not None:
            entry["start"] = round(src.start, 2)
            entry["line"] = src.number
        return entry

    def supported(item: dict, src: Turn | None) -> bool:
        """Drop items with no visible support near their source (copied or invented)."""
        if src is None:
            return _line_number(item.get("line")) is None and not item.get("quote")
        nearby = [t for t in turns if abs(t.number - src.number) <= 2]
        quote = str(item.get("quote") or "").strip()
        if quote:
            if max(_similarity(quote, t.text) for t in nearby) >= 0.6:
                return True
            # The quote may sit a little further away than the cited line.
            if any(_similarity(quote, t.text) >= 0.8 for t in turns if abs(t.number - src.number) <= 8):
                return True
            return False
        return max(_similarity(item.get("text", ""), t.text) for t in nearby) >= 0.15

    key_points, decisions, questions, actions = [], [], [], []
    for item in parsed.get("key_points") or []:
        key_points.append(placed({"text": item["text"]}, ground(item, turns, window)))
    for item in parsed.get("decisions") or []:
        if _VAGUE_DECISION.search(item["text"]):
            questions.append(placed({"text": item["text"]}, ground(item, turns, window)))
            continue
        src = ground(item, turns, window)
        if supported(item, src) or (item.get("quote") and _quote_anywhere(item["quote"], turns)):
            decisions.append(placed({"text": item["text"]}, src))
    for item in parsed.get("open_questions") or []:
        questions.append(placed({"text": item["text"]}, ground(item, turns, window)))
    for item in parsed.get("actions") or []:
        src = ground(item, turns, window)
        if not supported(item, src):
            log.info("Dropping an action with no support in the transcript: %s", item["text"][:80])
            continue
        if src is not None and not any(_COMMITMENT.search(t.text) for t in turns if abs(t.number - src.number) <= 1):
            log.info("Dropping an action with no commitment near its source: %s", item["text"][:80])
            continue
        owner = resolve_owner(item.get("owner"), names, src)
        subject = "|".join(re.escape(n) for n in {owner, "I", "We", "Me"} if n)
        text = re.sub(rf"^(?:(?:{subject})\s+(?:will|should|needs to|is going to|has to|to)|to)\s+", "",
                      item["text"].strip(), flags=re.IGNORECASE).strip()
        text = text[:1].upper() + text[1:] if text else text
        if not text or _PLEASANTRY.match(text):
            continue
        actions.append(placed({"owner": owner, "text": text, "due": _clean_due(item.get("due"))}, src))
    return {
        "summary": parsed.get("summary", "").strip(),
        "key_points": _dedup(key_points)[:MAX_KEY_POINTS],
        "decision_items": _dedup(decisions)[:MAX_DECISIONS],
        "actions": sorted(_dedup(actions), key=lambda a: a.get("start", 0))[:MAX_ACTIONS],
        "open_questions": _dedup(questions)[:MAX_QUESTIONS],
    }


# --------------------------------------------------------------------------- prompts


def _render(turns: list[Turn]) -> str:
    return "\n".join(f"L{t.number} [{clock(t.start)}] {t.speaker}: {t.text}" for t in turns)


def _people_note(names: list[str]) -> str:
    notes = []
    if "Me" in names:
        notes.append('"Me" is the person these notes are for (the owner of this computer); '
                     'use "Me" as the owner of their tasks.')
    if "Others" in names:
        notes.append('"Others" is the remote audio that could not be split by person, so it can be several '
                     'people; if the conversation makes clear who is speaking (someone is addressed by name, '
                     'or says who they are), use that name instead of "Others".')
    if any(re.match(r"^(SPEAKER_\d+|Speaker \d+)$", n) for n in names):
        notes.append("Speaker N labels come from automatic speaker separation; keep them as written.")
    notes.append("Speaker labels can be wrong for a line or two, so judge ownership from what is said.")
    return " ".join(notes)


_SCHEMA = (
    '{"summary": "<overview>", '
    '"key_points": [{"text": "<specific point>", "line": <L number>}], '
    '"decisions": [{"text": "<what was settled>", "quote": "<exact words from that line>", "line": <L number>}], '
    '"actions": [{"owner": "<speaker name, Me, or a named person>", "text": "<task starting with a verb>", '
    '"due": <stated deadline as a string, or null>, "quote": "<exact words from that line>", "line": <L number>}], '
    '"open_questions": [{"text": "<unresolved question or topic left for later>", "line": <L number>}]}'
)

_RULES = """Definitions:
- decision: something the people settled: a choice made, an approach agreed on, or something they agreed not to do. "Agreed to discuss X later" is NOT a decision (put it in open_questions or actions). Topics that were only talked about are not decisions, and one person's preference or proposal is not a decision unless the others accepted it. If nothing was settled, use [].
- action: a concrete follow-up task that someone committed to doing ("I'll send it", "I will message Dana") or was asked to do and accepted, to be done after this moment. Write it as a short imperative starting with a verb, with the specific object ("Forward the Slack question to Sam", not "Follow up"). owner: the person who will do it, using the speaker names above; if the task is for someone mentioned but not present, use their name. due: only if a time was stated ("by Monday", "next week"), otherwise null. Only include an action when the transcript has an explicit commitment or an accepted request; put the exact words in quote. Not actions: small talk, things finished during the call, opinions, suggestions nobody took on, topics to think about, and someone describing what they generally do. Do not restate decisions as actions unless a specific person took the task on. Many meetings, such as information-sharing calls, have no actions; then use [].
- key_points: the most important specific things said: facts, problems, numbers, proposals, concerns, with names. No filler.
- open_questions: unresolved questions or topics explicitly left for later.
- line: the L number of the transcript line where the item is said or committed to. quote: a few exact words copied from that line.
Rules: be specific and faithful to the transcript; do not invent details, owners, or dates. The transcript comes from speech recognition, so ignore garbled or nonsense lines and fix obvious mis-hearings only when the meaning is clear. Prefer fewer accurate items over many weak ones. Write in English."""


def prompt_for(segments: list[Segment], speakers: dict | None) -> str:
    """The single-pass prompt (kept for callers and tests)."""
    turns = build_turns(segments, speakers)
    return _single_prompt(turns)


def _single_prompt(turns: list[Turn], meeting_minutes: float | None = None) -> str:
    names = participants(turns, None)
    length = f" It lasted about {round(meeting_minutes)} minutes." if meeting_minutes else ""
    return (
        f"Below is a meeting transcript.{length} Each line is: L<number> [mm:ss] Speaker: text.\n"
        f"Speakers: {', '.join(names)}. {_people_note(names)}\n\n"
        f"Transcript:\n{_render(turns)}\n\n"
        "Write the meeting notes as one JSON object in exactly this shape:\n"
        f"{_SCHEMA}\n"
        "summary: 3 to 6 sentences: what the meeting was about, the main points of discussion, and where it "
        "ended up (outcomes, next steps). Mention people by name.\n"
        f"{_RULES}\n"
        f"Limits: at most {MAX_KEY_POINTS} key_points, {MAX_DECISIONS} decisions, {MAX_ACTIONS} actions.\n"
        "Reply with the JSON object only."
    )


def _part_prompt(turns: list[Turn], index: int, total: int, names: list[str]) -> str:
    span = f"{clock(turns[0].start)} to {clock(turns[-1].end)}"
    return (
        f"Below is part {index} of {total} ({span}) of a longer meeting transcript. "
        "Each line is: L<number> [mm:ss] Speaker: text.\n"
        f"Speakers in the meeting: {', '.join(names)}. {_people_note(names)}\n\n"
        f"Transcript part {index}:\n{_render(turns)}\n\n"
        "Write notes for this part as one JSON object in exactly this shape:\n"
        f"{_SCHEMA}\n"
        "summary: 2 to 4 sentences about what was discussed in this part and any outcome.\n"
        f"{_RULES}\n"
        "Limits: at most 6 key_points for this part.\n"
        "Reply with the JSON object only."
    )


def _reduce_prompt(parts: list[dict], names: list[str], meeting_minutes: float | None,
                   decisions: list[dict], actions: list[dict]) -> str:
    rows = []
    for number, part in enumerate(parts, 1):
        rows.append(f"Part {number} ({part['span']}): {part['summary']}")
        for point in part["key_points"]:
            rows.append(f"  - [{clock(point.get('start', 0))}] {point['text']}")
    cand = [f"D{i} [{clock(d.get('start', 0))}] {d['text']}" for i, d in enumerate(decisions, 1)]
    acts = []
    for i, a in enumerate(actions, 1):
        due = f" (due {a['due']})" if a.get("due") else ""
        acts.append(f"A{i} [{clock(a.get('start', 0))}] {a['owner']}: {a['text']}{due}")
    length = f" It lasted about {round(meeting_minutes)} minutes." if meeting_minutes else ""
    return (
        f"These are notes taken on consecutive parts of one meeting.{length} Speakers: {', '.join(names)}. "
        + _people_note(names) + "\n\n"
        + "\n".join(rows)
        + "\n\nCandidate decisions:\n" + ("\n".join(cand) or "(none)")
        + "\n\nCandidate action items:\n" + ("\n".join(acts) or "(none)")
        + "\n\nWrite the notes for the whole meeting as one JSON object in exactly this shape:\n"
        '{"summary": "<overview>", "key_points": [{"text": "<point>", "time": "mm:ss"}], '
        '"decisions": [{"ids": ["D1", "D4"], "text": "<merged decision>"}], '
        '"actions": [{"ids": ["A2"], "owner": "<owner>", "text": "<task>", "due": <deadline or null>}]}\n'
        "summary: at most 7 sentences: what the meeting was about, the main threads of discussion, and where it "
        "ended up (outcomes and next steps). Mention people by name. Do not narrate part by part.\n"
        f"key_points: the {MAX_KEY_POINTS} most important specific points of the whole meeting, merged and "
        "de-duplicated, each with the time of its source point.\n"
        "decisions and actions: consolidate the candidates. Merge candidates that say the same thing (list all "
        "their ids) and keep the clearest wording; drop candidates that are not real decisions or not real "
        "commitments (pleasantries, suggestions nobody took on, restated decisions). Only use the candidate "
        "ids given; do not add new items.\n"
        "Only use what the notes say. Reply with the JSON object only."
    )


# --------------------------------------------------------------------------- model pass


def chunk_turns(turns: list[Turn], max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_TURNS) -> list[list[Turn]]:
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    size = 0
    for turn in turns:
        length = len(turn.text) + 24
        if current and size + length > max_chars:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            size = sum(len(t.text) + 24 for t in current)
        current.append(turn)
        size += length
    if current and (not chunks or current[-1] is not chunks[-1][-1]):
        chunks.append(current)
    return chunks


def _ask(runner: Callable[[str], str], prompt: str) -> dict | None:
    """One prompt, one retry on a reply that is not the JSON we asked for."""
    for attempt in range(2):
        text = prompt if attempt == 0 else (
            prompt + "\n\nImportant: your previous reply was not a valid JSON object with a non-empty "
            '"summary". Reply with only the JSON object, starting with { and ending with }.')
        try:
            raw = runner(text)
        except Exception:
            log.exception("Local notes model failed")
            return None
        parsed = parse_model_notes(raw)
        if parsed:
            return parsed
        log.warning("Notes model reply was not valid JSON (attempt %d)", attempt + 1)
    return None


def _time_to_seconds(value) -> float | None:
    match = re.match(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*$", str(value or ""))
    if not match:
        return None
    h, m, s = match.groups()
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def _consolidate(chosen: list, candidates: list[dict], prefix: str, names: list[str]) -> list[dict]:
    """Apply the reduce step's merge/keep choices, keeping each kept item's source time."""
    out: list[dict] = []
    used: set[int] = set()
    for item in chosen:
        if not isinstance(item, dict):
            continue
        ids = []
        for raw in item.get("ids") or []:
            match = re.search(r"\d+", str(raw))
            if match and 1 <= int(match.group()) <= len(candidates) and int(match.group()) - 1 not in used:
                ids.append(int(match.group()) - 1)
        if not ids:
            continue
        used.update(ids)
        base = dict(candidates[min(ids, key=lambda i: candidates[i].get("start", 0))])
        text = str(item.get("text") or "").strip()
        if text:
            base["text"] = text[:500]
        if prefix == "A":
            if item.get("owner"):
                base["owner"] = resolve_owner(item["owner"], names, None)
            due = _clean_due(item.get("due"))
            if due:
                base["due"] = due
        out.append(base)
    if not out:
        return candidates
    # The merge step may combine candidates but not silently lose one: every
    # candidate here already passed the grounding and commitment checks.
    out.extend(candidates[i] for i in range(len(candidates)) if i not in used)
    return sorted(out, key=lambda item: item.get("start", 0))


def model_notes(turns: list[Turn], runner: Callable[[str], str]) -> dict | None:
    if not turns:
        return None
    minutes = (turns[-1].end - turns[0].start) / 60 if turns else None
    total_chars = sum(len(t.text) + 24 for t in turns)
    if total_chars <= SINGLE_PASS_CHARS:
        parsed = _ask(runner, _single_prompt(turns, minutes))
        if not parsed:
            return None
        notes = finish_notes(parsed, turns)
        notes["chunks"] = 1
        return notes

    names = participants(turns, None)
    chunks = chunk_turns(turns, CHUNK_CHARS, CHUNK_OVERLAP_TURNS)
    parts = []
    for index, chunk in enumerate(chunks, 1):
        parsed = _ask(runner, _part_prompt(chunk, index, len(chunks), names))
        if not parsed:
            continue
        part = finish_notes(parsed, turns, window=(chunk[0].number, chunk[-1].number))
        part["span"] = f"{clock(chunk[0].start)}-{clock(chunk[-1].end)}"
        parts.append(part)
    if not parts:
        return None
    merged_points = [p for part in parts for p in part["key_points"]]
    notes = {
        "summary": " ".join(part["summary"] for part in parts),
        "key_points": _dedup(merged_points)[:MAX_KEY_POINTS],
        "decision_items": _dedup([d for part in parts for d in part["decision_items"]])[:MAX_DECISIONS],
        "actions": _dedup([a for part in parts for a in part["actions"]])[:MAX_ACTIONS],
        "open_questions": _dedup([q for part in parts for q in part["open_questions"]])[:MAX_QUESTIONS],
        "chunks": len(chunks),
    }
    reduced = None
    try:
        reduced = _json_blob(runner(_reduce_prompt(parts, names, minutes, notes["decision_items"], notes["actions"])))
    except Exception:
        log.exception("Notes reduce step failed; joining the part summaries")
    if reduced and str(reduced.get("summary") or "").strip():
        notes["summary"] = str(reduced["summary"]).strip()[:4000]
        points = []
        for item in reduced.get("key_points") or []:
            text = item.get("text") if isinstance(item, dict) else item
            if not str(text or "").strip():
                continue
            entry = {"text": str(text).strip()[:500]}
            seconds = _time_to_seconds(item.get("time")) if isinstance(item, dict) else None
            near = [p for p in merged_points if "start" in p and seconds is not None and abs(p["start"] - seconds) <= 5]
            if near:
                entry["start"], entry["line"] = near[0]["start"], near[0].get("line")
            else:
                src = ground(entry, turns)
                if src is not None:
                    entry["start"], entry["line"] = round(src.start, 2), src.number
            points.append(entry)
        if points:
            notes["key_points"] = _dedup(points)[:MAX_KEY_POINTS]
        if isinstance(reduced.get("decisions"), list):
            notes["decision_items"] = _consolidate(reduced["decisions"], notes["decision_items"], "D", names)
        if isinstance(reduced.get("actions"), list):
            notes["actions"] = _consolidate(reduced["actions"], notes["actions"], "A", names)
    return notes


def build_notes(
    segments: list[Segment],
    speakers: dict | None,
    model_id: str = DEFAULT_LLM,
    complete: Callable[[str], str] | None = None,
    settings=None,
) -> dict:
    """Notes for one transcript. ``complete`` overrides the backend call (tests).

    The generative path is chosen from ``settings``: MLX when the machine has
    it, the operator's Ollama endpoint when one is configured, otherwise the
    extractive notes below - which are always returned labelled
    ``backend: "extractive"`` and never as generated.
    """
    fallback = extractive_notes(segments, speakers)
    turns = build_turns(segments, speakers)
    if not turns:
        return fallback
    runner = complete
    backend = "model" if runner is not None else None
    reported_model = model_id
    if runner is None:
        chosen = notes_backend(settings, mlx_available=mlx_lm_available())
        if chosen == "ollama":
            if settings is None:
                settings = Settings.load()
            # The notes must name the backend and model that actually ran, not
            # the MLX id this machine never loaded.
            reported_model = str(getattr(settings, "llm_ollama_model", "") or "") or model_id

            def runner(prompt: str) -> str:
                return generate(prompt, model_id, max_tokens=NOTES_MAX_TOKENS,
                                system=SYSTEM_PROMPT, settings=settings, backend="ollama")
            backend = "ollama"
        elif chosen == "mlx":
            def runner(prompt: str) -> str:
                return generate(prompt, model_id, max_tokens=NOTES_MAX_TOKENS, system=SYSTEM_PROMPT)
            backend = "mlx"
    if runner is None:
        return fallback
    started = time.time()
    try:
        notes = model_notes(turns, runner)
    except Exception:
        log.exception("Local notes model failed; using the extractive fallback")
        notes = None
    if not notes:
        return fallback
    for items in (notes["key_points"], notes["decision_items"], notes["actions"], notes["open_questions"]):
        for item in items:
            item.pop("line", None)
            item.pop("quote", None)
    return {
        "summary": notes["summary"],
        "key_points": notes["key_points"],
        "decisions": [d["text"] for d in notes["decision_items"]],
        "decision_items": notes["decision_items"],
        "actions": notes["actions"],
        "open_questions": notes["open_questions"],
        "model": reported_model,
        "backend": backend or "extractive",
        "edited": False,
        "version": NOTES_VERSION,
        "chunks": notes.get("chunks", 1),
        "seconds": round(time.time() - started, 1),
    }


# --------------------------------------------------------------------------- storage


def write_notes(library: Library, session_id: str, settings: Settings,
                complete: Callable[[str], str] | None = None, force: bool = True) -> dict:
    """Generate and save notes. With ``force=False`` hand-edited notes are kept."""
    meta = library.get(session_id)
    if not force:
        existing = library.notes(session_id)
        if existing and existing.get("edited"):
            log.info("Keeping edited notes for %s", session_id)
            return existing
    notes = build_notes(library.segments(session_id), meta.speakers, settings.llm_model, complete, settings=settings)
    library.save_notes(session_id, notes)
    return notes


def schedule_notes(library: Library, session_id: str, settings: Settings, bus=None) -> None:
    if not settings.summarize_after:
        return

    def run() -> None:
        try:
            notes = write_notes(library, session_id, settings, force=False)
        except Exception:
            log.exception("Could not write notes for %s", session_id)
            return
        if bus is not None:
            bus.publish({"type": "summary.ready", "session": session_id, "notes": notes})

    threading.Thread(target=run, name="listn-notes", daemon=True).start()


def _carry(items: list[dict], previous: list | None, key: str = "text") -> list[dict]:
    """Keep start/due from the previous items whose text is unchanged."""
    old = {str(p.get(key)).strip(): p for p in (previous or []) if isinstance(p, dict) and p.get(key)}
    for item in items:
        before = old.get(item[key])
        if before:
            for extra in ("start", "due"):
                if extra in before and extra not in item:
                    item[extra] = before[extra]
    return items


def edited_notes(summary: str, decisions: list[str], actions: list[dict], previous: dict | None,
                 key_points: list[str] | None = None) -> dict:
    base = dict(previous or {})
    base["summary"] = (summary or "").strip()[:4000]
    base["decisions"] = _as_str_list(decisions)
    base["decision_items"] = _carry([{"text": d} for d in base["decisions"]], base.get("decision_items"))
    actions_out = _as_actions(actions)
    for action in actions_out:
        action.pop("line", None)
    prev_actions = base.get("actions")
    base["actions"] = _carry(actions_out, prev_actions)
    if key_points is not None:
        points = [{"text": p} for p in _as_str_list(key_points)]
        base["key_points"] = _carry(points, base.get("key_points"))
    base["edited"] = True
    base.setdefault("backend", "edited")
    return base
