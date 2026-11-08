# Decisions and roadmap

## Existing product boundaries

| Boundary | Rationale |
|---|---|
| Local transcription and private local storage | Audio and transcripts are sensitive; see README for model-download exceptions. |
| Loopback web UI and private local socket | Keep the existing browser and capture interfaces local. Preserve Host/origin checks and socket permissions. |
| Separate microphone and system tracks | Distinguish the user from other audio without claiming individual identity. |
| macOS capture separated from Python engine | The native capture UI sends audio to the shared engine rather than performing transcription itself. |

## Proposed next phase — KDE Plasma / CachyOS

Mark requested Linux desktop support focused on KDE, and raised the idea of a dashboard. The initial Python file and local web workflow is documented and exercised on Linux with synthetic audio; desktop capture parity is not claimed.

1. Establish a reproducible Linux development and test path for file transcription and the existing web interface.
2. Prototype microphone and system-output capture as separate tracks using the target desktop's audio stack. Verify permissions, device changes, silence, failure and pause handling on real hardware.
   A first prototype now exists in `listn/pipewire.py` with automated tests and a headless-daemon run documented in [LINUX_CAPTURE](LINUX_CAPTURE.md); verification on real hardware is still outstanding.
3. Add KDE tray controls and a dashboard based on the existing local web UI, subject to a concrete UX/design specification. Candidate information: recording status, input/output sources, activity/levels, transcript library and diagnostics. Clearly distinguish proposed from observed data; never present silence as a successful recording.
   A first read-only dashboard now ships at `/dashboard` (backed by `GET /api/dashboard`): recording status, per-track PipeWire state, transcription queue and recent recordings with track separation, rendered from real engine state with honest empty states.
   A minimal tray now exists (`listn tray`, `listn/tray.py` + `listn/tray_sni.py`): recording status, start/stop/pause, target pickers, open-privacy-settings and quit over StatusNotifierItem/dbusmenu, with failed/unreachable states shown as failures. It is a client of the loopback API, hides the controls listn cannot honour (meetings, auto-record, launch-at-login) instead of faking them, and its D-Bus contract is tested against a real session bus. Neither has a designed UX yet, and neither has been exercised on real hardware or inside a running plasmashell panel.
4. Evaluate packaging and hardware acceleration on CachyOS. Per-application capture, generative notes and other distributions are follow-up decisions, not prerequisites for documenting the repo.

## Acceptance principles for a future Linux release

- Demonstrate independent microphone and playback recording and transcription on the actual target environment; synthetic tests alone do not prove capture.
- Maintain macOS behavior, local privacy boundaries, private data permissions, explicit errors and pause semantics.
- Show what is running, what is unavailable and what needs action; avoid claiming feature parity or GPU performance without evidence.
- Document model downloads, hardware requirements and any feature differences. Do not introduce cloud processing or telemetry without approval.

## Generative notes on Linux: an Ollama endpoint, not a local runtime (2026-11-05)

Mark decided the L4 question: listn uses the Ollama service he already runs (`http://10.10.10.6:11434`, tag `qwen3.8:27b`) instead of embedding llama.cpp and instead of acquiring the pinned Qwen3-8B-GGUF weights. Consequences:

- listn ships no weights and installs no runtime. It is an HTTP client of one operator-configured endpoint (`llm_ollama_url` / `LISTN_OLLAMA_URL`), and configuring that endpoint is the opt-in. No endpoint → extractive notes and no outbound connection.
- The gate's "zero outbound connections during generation" and "≤9 GB peak RSS / unload ≤300 s" conditions were rewritten for a service listn does not own; the deviations and the measured numbers live in [55-linux-generative-notes-backend](55-linux-generative-notes-backend.md) (150.9 s for a 599-token single-chunk note at 4.8 tok/s, listn-side peak RSS 23.7 MiB).
- Extractive stays the default and the fallback, labelled `extractive`, and macOS MLX behaviour is unchanged.
- QA acceptance (`t_2bb863cd`, 2026-11-05/06) found the 180 s client budget (`OLLAMA_TIMEOUT_SECONDS`, pinned to the gate envelope) loses to a cold start: Ollama unloads after 5 idle min, and at 4.8 tok/s a 783-token note needs ~185 s warm+load — two cold runs timed out at 180.1 s and fell back to extractive, safe and honestly labelled. Full table, token-count dependency (599 vs 783 tokens → 150.9 s vs 165.0 s warm / >180 s cold), and the open headroom question live in [55-linux-generative-notes-backend](55-linux-generative-notes-backend.md). Whether the timeout stays pinned or carries headroom is Mark's call.
- Open, not measured: a multi-chunk meeting on this endpoint would be several requests at 4.8 tok/s and is expected to exceed the 180 s envelope. No quality or parity claim is made until QA measures it.
