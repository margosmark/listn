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
3. Add KDE tray controls and a dashboard based on the existing local web UI, subject to a concrete UX/design specification. Candidate information: recording status, input/output sources, activity/levels, transcript library and diagnostics. Clearly distinguish proposed from observed data; never present silence as a successful recording.
4. Evaluate packaging and hardware acceleration on CachyOS. Per-application capture, generative notes and other distributions are follow-up decisions, not prerequisites for documenting the repo.

## Acceptance principles for a future Linux release

- Demonstrate independent microphone and playback recording and transcription on the actual target environment; synthetic tests alone do not prove capture.
- Maintain macOS behavior, local privacy boundaries, private data permissions, explicit errors and pause semantics.
- Show what is running, what is unavailable and what needs action; avoid claiming feature parity or GPU performance without evidence.
- Document model downloads, hardware requirements and any feature differences. Do not introduce cloud processing or telemetry without approval.
