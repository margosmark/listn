# Agent guidance

Read [README.md](README.md), [architecture](docs/ARCHITECTURE.md), [operations](docs/OPERATIONS.md), and [decisions and roadmap](docs/DECISIONS.md) before changing the project. Source and tests are authoritative when documentation disagrees.

## Guardrails

- Preserve local-first operation: loopback-only web service, origin and Host checks, private data directory and Unix socket permissions. Never expose transcripts, recordings, tokens, voiceprints or settings in logs or reports.
- Do not add cloud transcription, telemetry, outbound services or implicit model downloads without explicit approval. Existing first-use model downloads are described in README.
- Preserve the independent system and microphone tracks, truthful recording/failure states, pause behavior and macOS functionality. Never claim capture or transcription succeeded without evidence.
- Linux/KDE support and a dashboard are proposed work, not shipped functionality. Do not label them implemented without a working change and tests. Do not assume per-app capture, GPU acceleration or AI-generated Linux notes works.
- Avoid touching user recordings or deleting data during tests. Use isolated temporary data directories and synthetic audio. Do not commit secrets, models, generated recordings or local settings.

## Development

- Python engine: `listn/`; macOS Swift capture and tray: `macos/`; tests: `tests/`.
- Run relevant tests and report exact commands, results and environment limitations. `uv sync --group dev` then `uv run pytest` is the baseline when dependencies are available. A macOS build and real hardware audio checks require macOS; Linux capture requires a real Linux desktop/audio session.
- Existing scripts and Makefile primarily target macOS; inspect before running install, update, permission reset or uninstall commands. Never run destructive operations on a user's machine without authorization.
- Keep changes scoped; don't overwrite unrelated work. Update documentation and tests with behavioral changes. Record the commit and verify pushed refs before claiming completion.
