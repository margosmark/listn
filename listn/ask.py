"""Answer a question from the library, citing transcript lines."""

from __future__ import annotations

import json
import re

from listn.config import Settings
from listn.index import SearchIndex
from listn.library import fmt_clock
from listn.llm import mlx_lm_available, notes_backend


def _complete(prompt: str, settings: Settings) -> str:
    from listn.llm import generate

    return generate(prompt, settings.llm_model, max_tokens=500, settings=settings)


def _parse_answer(raw: str) -> tuple[str, list[int]] | None:
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
    answer = str(data.get("answer") or "").strip()
    cites = []
    for item in data.get("citations") or []:
        try:
            cites.append(int(item))
        except (TypeError, ValueError):
            continue
    if not answer:
        return None
    return answer, cites


def ask(index: SearchIndex, question: str, settings: Settings | None = None, complete=None) -> dict:
    question = (question or "").strip()
    hits = index.search(question, limit=6) if question else []
    citations = [{k: hit[k] for k in ("session_id", "title", "start", "end", "text", "speaker")} for hit in hits]
    if not question:
        return {"answer": "Ask something about your meetings.", "citations": []}
    if not hits:
        return {"answer": "Nothing in the library matches that.", "citations": []}
    settings = settings or Settings()
    lines = []
    for number, hit in enumerate(hits, 1):
        who = hit.get("speaker") or "speaker"
        lines.append(f"[{number}] {hit['title']} at {fmt_clock(hit['start'])} ({who}): {hit['text']}")
    prompt = (
        "Answer the question using only the excerpts. If they do not contain the answer, say so. "
        "Reply with JSON only: {\"answer\":\"...\",\"citations\":[1,2]} where citations are excerpt numbers.\n\n"
        f"Question: {question}\n\n" + "\n".join(lines)
    )
    runner = complete
    if runner is None and notes_backend(settings, mlx_available=mlx_lm_available()) is not None:
        runner = lambda text: _complete(text, settings)  # noqa: E731
    parsed = None
    if runner is not None:
        try:
            parsed = _parse_answer(runner(prompt))
        except Exception:
            parsed = None
    if parsed:
        answer, numbers = parsed
        chosen = []
        for number in numbers:
            if 1 <= number <= len(citations):
                chosen.append(citations[number - 1])
        if not chosen:
            chosen = citations[:3]
        return {"answer": answer, "citations": chosen}
    # No local LLM: quote the closest lines so the citations still jump to audio.
    quoted = " ".join(f"{item['text']}" for item in citations[:3])
    return {
        "answer": f"Closest lines: {quoted}",
        "citations": citations[:3],
        "backend": "excerpts",
    }
