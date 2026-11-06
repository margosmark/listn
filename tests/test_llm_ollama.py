"""Ollama notes backend: routing, request contract, fallback matrix, privacy.

Every test here talks to a local fake server on 127.0.0.1. Nothing downloads
weights, installs a runtime or reaches an operator's endpoint: on a machine
with no configured endpoint listn must make no outbound connection at all,
and on one with an endpoint it may talk to that endpoint and nothing else.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from listn import llm
from listn.ask import ask
from listn.config import Settings, SettingsError
from listn.index import SearchIndex
from listn.library import Library, Segment
from listn.summary import build_notes, extractive_notes

MARK_URL = "http://10.10.10.6:11434"
MARK_MODEL = "qwen3.8:27b"
SYSTEM_LINE = "You write accurate, specific meeting notes."

GOOD_REPLY = {
    "summary": "The team agreed to ship the export on Friday in Markdown.",
    "key_points": [{"text": "The export ships on Friday"}],
    "decisions": ["Use Markdown for the export"],
    "actions": [{"owner": "Me", "text": "Send the notes to Priya"}],
}


def settings_for(url: str, model: str = MARK_MODEL, backend: str = "auto") -> Settings:
    return Settings(llm_backend=backend, llm_ollama_url=url, llm_ollama_model=model).validated()


def _meeting(lib: Library):
    meta = lib.create("Planning", "live", speakers={"Me": "Me", "Others": "Others"})
    lib.save_segments(meta.id, [
        Segment(0, 2, "Let's ship the export on Friday.", "Me"),
        Segment(2, 5, "Agreed, we'll go with Markdown.", "Others"),
        Segment(5, 8, "I'll send the notes to Priya.", "Me"),
    ])
    return meta


@pytest.fixture
def fake_ollama():
    """Factory for fake Ollama services: ``(responder) -> (url, server)``."""
    servers = []

    def start(responder):
        class Handler(BaseHTTPRequestHandler):
            def _reply(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                request = {"path": self.path, "method": method,
                           "body": self.rfile.read(length),
                           "headers": dict(self.headers)}
                self.server.requests.append(request)
                # ``(status, payload)`` or ``(status, payload, {header: value})``
                # - the 302 regression test needs a Location header.
                reply = responder(request)
                status, payload = reply[0], reply[1]
                extra = reply[2] if len(reply) > 2 else {}
                raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for key, value in extra.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):  # noqa: N802
                self._reply("POST")

            def do_GET(self):  # noqa: N802
                # urllib follows a 302 with a bodyless GET, so a followed
                # redirect must be recorded here as a hit on this origin.
                self._reply("GET")

            def log_message(self, *args):  # keep pytest output clean
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.requests = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def good_response(request):
    return 200, {
        "model": MARK_MODEL,
        "message": {"role": "assistant", "content": json.dumps(GOOD_REPLY)},
        "prompt_eval_count": 12,
        "eval_count": 34,
        "eval_duration": 2_000_000_000,
    }


# --------------------------------------------------------------------------- configuration


def test_validate_ollama_url_normalises_a_bare_origin():
    assert llm.validate_ollama_url(MARK_URL + "/") == MARK_URL
    assert llm.validate_ollama_url("https://models.internal") == "https://models.internal"
    assert llm.validate_ollama_url("http://[::1]:11434") == "http://[::1]:11434"
    assert llm.validate_ollama_url("") == ""
    assert llm.validate_ollama_url(None) == ""


@pytest.mark.parametrize("bad", [
    "ftp://10.10.10.6:11434",
    "10.10.10.6:11434",
    "http://",
    "http://user:secret@10.10.10.6:11434",
    "http://10.10.10.6:11434/api",
    "http://10.10.10.6:11434?model=x",
    "http://10.10.10.6:notaport",
    "javascript:alert(1)",
])
def test_validate_ollama_url_refuses_anything_but_a_bare_http_origin(bad):
    with pytest.raises(ValueError):
        llm.validate_ollama_url(bad)


def test_validation_errors_never_echo_credentials():
    """N1: a rejected URL surfaces in SettingsError without its secrets."""
    for bad in ("ftp://user:secret@10.10.10.6:11434",
                "http://user:secret@",
                "1user:secret@10.10.10.6"):
        with pytest.raises(SettingsError) as caught:
            Settings(llm_ollama_url=bad).validated()
        assert "secret" not in str(caught.value), "credentials leaked into a settings error"


def test_settings_reject_an_unknown_backend():
    with pytest.raises(SettingsError):
        Settings(llm_backend="ollma").validated()


def test_settings_reject_an_endpoint_listn_wont_talk_to():
    with pytest.raises(SettingsError):
        Settings(llm_ollama_url="http://10.10.10.6:11434/api").validated()


def test_settings_require_url_and_model_when_ollama_is_selected():
    with pytest.raises(SettingsError):
        Settings(llm_backend="ollama").validated()


def test_settings_accept_marks_endpoint():
    settings = settings_for(MARK_URL + "/", backend="ollama")
    assert (settings.llm_ollama_url, settings.llm_ollama_model) == (MARK_URL, MARK_MODEL)


def test_env_overrides_win_over_settings_json(data_dir, monkeypatch):
    monkeypatch.setenv("LISTN_LLM_BACKEND", "ollama")
    monkeypatch.setenv("LISTN_OLLAMA_URL", MARK_URL)
    monkeypatch.setenv("LISTN_OLLAMA_MODEL", MARK_MODEL)
    settings = Settings.load()
    assert (settings.llm_backend, settings.llm_ollama_url, settings.llm_ollama_model) == (
        "ollama", MARK_URL, MARK_MODEL)


# --------------------------------------------------------------------------- routing


@pytest.mark.parametrize("backend,mlx,url,model,expected", [
    ("auto", False, "", MARK_MODEL, None),
    ("auto", False, MARK_URL, MARK_MODEL, "ollama"),
    ("auto", True, MARK_URL, MARK_MODEL, "mlx"),
    ("auto", True, "", "", "mlx"),
    ("extractive", True, MARK_URL, MARK_MODEL, None),
    ("mlx", False, "", "", None),
    ("ollama", False, MARK_URL, "", None),
    ("ollama", False, MARK_URL, MARK_MODEL, "ollama"),
])
def test_notes_backend_routing(backend, mlx, url, model, expected):
    settings = Settings(llm_backend=backend, llm_ollama_url=url, llm_ollama_model=model)
    assert llm.notes_backend(settings, mlx_available=mlx) is expected


# --------------------------------------------------------------------------- request contract


def test_generate_sends_the_contract_the_endpoint_expects(fake_ollama):
    url, server = fake_ollama(good_response)
    out = llm.generate("SECRET_PROMPT_TEXT", settings=settings_for(url),
                       backend="ollama", system=SYSTEM_LINE, max_tokens=123)
    assert json.loads(out) == GOOD_REPLY
    request = server.requests[0]
    assert request["path"] == "/api/chat"
    body = json.loads(request["body"])
    assert body["model"] == MARK_MODEL
    assert body["stream"] is False
    assert body["think"] is False, "thinking would burn the note budget on an empty content field"
    assert body["options"] == {"num_predict": 123, "temperature": 0.0}
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_LINE}
    assert body["messages"][1]["content"] == "SECRET_PROMPT_TEXT"


def test_notes_name_the_backend_and_model_that_actually_ran(data_dir, fake_ollama):
    url, server = fake_ollama(good_response)
    lib = Library()
    meta = _meeting(lib)
    notes = build_notes(lib.segments(meta.id), meta.speakers, settings=settings_for(url))
    assert notes["backend"] == "ollama"
    assert notes["model"] == MARK_MODEL, "notes must not claim an MLX model this machine never loaded"
    assert notes["summary"] == GOOD_REPLY["summary"]
    assert notes["actions"][0]["owner"] == "Me"
    assert server.requests, "one chat request per note run"


def test_ask_answers_through_the_endpoint(data_dir, fake_ollama):
    url, _server = fake_ollama(lambda request: (
        200, {"message": {"content": '{"answer":"The export ships Friday.","citations":[1]}'}}))
    lib = Library()
    planning = lib.create("Planning", "live")
    lib.save_segments(planning.id, [Segment(1.0, 3.0, "Let's ship the export on Friday.", "Me")])
    index = SearchIndex(lib)
    result = ask(index, "When does the export ship?", settings_for(url))
    assert result["answer"] == "The export ships Friday."
    assert result["citations"][0]["session_id"] == planning.id


# --------------------------------------------------------------------------- fallback matrix


def _unused_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.mark.parametrize("case", ["refused", "http_500", "not_json", "empty_reply", "bad_notes"])
def test_every_backend_failure_falls_back_to_extractive(data_dir, fake_ollama, case):
    lib = Library()
    meta = _meeting(lib)
    segments = lib.segments(meta.id)
    if case == "refused":
        url = f"http://127.0.0.1:{_unused_port()}"
    elif case == "http_500":
        url, _server = fake_ollama(lambda request: (500, {"error": "boom"}))
    elif case == "not_json":
        url, _server = fake_ollama(lambda request: (200, b"<html>not json</html>"))
    elif case == "empty_reply":
        url, _server = fake_ollama(lambda request: (200, {"message": {"content": "   "}}))
    else:
        url, _server = fake_ollama(lambda request: (200, {"message": {"content": "no json today"}}))

    notes = build_notes(segments, meta.speakers, settings=settings_for(url))

    assert notes["backend"] == "extractive", "a failed backend must never be labelled generated"
    assert notes == extractive_notes(segments, meta.speakers), "the transcript must not be lost"
    assert notes["summary"].strip(), "fallback notes must not be blank"


def test_timeout_is_an_ollama_error_and_never_empty_output(fake_ollama, monkeypatch):
    url, _server = fake_ollama(lambda request: (time.sleep(2.0) or 200,
                                                {"message": {"content": "late"}}))
    monkeypatch.setattr(llm, "OLLAMA_TIMEOUT_SECONDS", 0.2)
    with pytest.raises(llm.OllamaError) as caught:
        llm.ollama_generate("hello", MARK_MODEL, url)
    assert "hello" not in str(caught.value)


def test_a_timeout_during_notes_writes_extractive_notes(data_dir, fake_ollama, monkeypatch):
    url, _server = fake_ollama(lambda request: (time.sleep(2.0) or 200, good_response(request)[1]))
    monkeypatch.setattr(llm, "OLLAMA_TIMEOUT_SECONDS", 0.2)
    lib = Library()
    meta = _meeting(lib)
    notes = build_notes(lib.segments(meta.id), meta.speakers, settings=settings_for(url))
    assert notes["backend"] == "extractive"
    assert notes["summary"].strip()


def test_llm_backend_off_means_extractive_even_with_an_endpoint(data_dir, fake_ollama):
    url, server = fake_ollama(good_response)
    lib = Library()
    meta = _meeting(lib)
    notes = build_notes(lib.segments(meta.id), meta.speakers,
                        settings=settings_for(url, backend="extractive"))
    assert notes["backend"] == "extractive"
    assert server.requests == [], "extractive mode must not call the endpoint"


# --------------------------------------------------------------------------- egress + privacy


def test_generation_only_touches_the_configured_endpoint(data_dir, fake_ollama, monkeypatch):
    url, _server = fake_ollama(good_response)
    port = int(url.rsplit(":", 1)[1])
    targets = []
    real_connect = socket.socket.connect

    def spy(sock, address):
        targets.append(address)
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", spy)
    lib = Library()
    meta = _meeting(lib)
    notes = build_notes(lib.segments(meta.id), meta.speakers, settings=settings_for(url))
    assert notes["backend"] == "ollama"
    assert targets, "expected the one chat request"
    for address in targets:
        assert address[0] in ("127.0.0.1", "::1"), f"unexpected egress to {address}"
        assert address[1] == port, f"unexpected egress to {address}"


def test_a_302_is_refused_before_a_second_origin_is_contacted(fake_ollama, monkeypatch):
    """F1 regression: a redirect off the allowlisted origin is answered,
    never followed, and the redirect target's body is never read."""
    # The redirect target would answer with perfectly valid notes if contacted.
    target_url, target_server = fake_ollama(good_response)
    url, server = fake_ollama(
        lambda request: (302, {"error": "moved"}, {"Location": target_url + "/api/chat"}))
    target_port = int(target_url.rsplit(":", 1)[1])
    connects = []
    real_connect = socket.socket.connect

    def spy(sock, address):
        connects.append(address)
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", spy)

    with pytest.raises(llm.OllamaError) as caught:
        llm.ollama_generate("SECRET_PROMPT_TEXT", MARK_MODEL, url)

    message = str(caught.value)
    assert url in message, "the error must name the allowlisted origin"
    assert "302" in message, "the error must carry the HTTP status"
    assert "SECRET_PROMPT_TEXT" not in message, "the prompt leaked into the error"
    assert GOOD_REPLY["summary"] not in message, "output text leaked into the error"
    assert server.requests and server.requests[0]["path"] == "/api/chat", (
        "the allowlisted origin must receive the chat request")
    assert target_server.requests == [], "the redirect target must never be asked for anything"
    assert all(address[1] != target_port for address in connects), (
        f"no socket may open to the redirect target {target_url}")


def test_notes_never_come_from_a_redirect_target(data_dir, fake_ollama):
    """F1 regression end to end: a refused redirect falls back to extractive
    notes and the target's body never becomes generated notes."""
    target_url, target_server = fake_ollama(good_response)
    url, _server = fake_ollama(
        lambda request: (302, {}, {"Location": target_url + "/api/chat"}))
    lib = Library()
    meta = _meeting(lib)
    notes = build_notes(lib.segments(meta.id), meta.speakers, settings=settings_for(url))
    assert notes["backend"] == "extractive", "a refused redirect must never be labelled ollama"
    assert notes["summary"] != GOOD_REPLY["summary"]
    assert GOOD_REPLY["summary"] not in json.dumps(notes), (
        "the redirect target's body must not become notes")
    assert target_server.requests == []


def test_without_an_endpoint_no_connection_is_made(data_dir, monkeypatch):
    targets = []
    real_connect = socket.socket.connect

    def spy(sock, address):
        targets.append(address)
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", spy)
    lib = Library()
    meta = _meeting(lib)
    notes = build_notes(lib.segments(meta.id), meta.speakers, settings=settings_for(""))
    assert notes["backend"] == "extractive"
    assert targets == []


def test_no_prompt_transcript_or_reply_reaches_the_logs(data_dir, fake_ollama, caplog):
    lib = Library()
    meta = _meeting(lib)
    segments = lib.segments(meta.id)
    url, _server = fake_ollama(good_response)
    broken, _broken_server = fake_ollama(lambda request: (500, {"error": "boom"}))
    with caplog.at_level(logging.DEBUG):
        build_notes(segments, meta.speakers, settings=settings_for(url))
        build_notes(segments, meta.speakers, settings=settings_for(broken))
    text = caplog.text
    assert text, "expected the run to log something (numbers at least)"
    for secret in ["Let's ship the export on Friday.",
                   "Agreed, we'll go with Markdown.",
                   "I'll send the notes to Priya.",
                   json.dumps(GOOD_REPLY),
                   GOOD_REPLY["summary"]]:
        assert secret not in text, "transcript or reply content leaked into the log"
