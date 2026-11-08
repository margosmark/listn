# Linux generative notes backend: options and decision

**Status:** decision made and implemented (Ollama), measured against Mark's own service on 2026-11-05. The options brief below is kept as the reasoning that preceded the decision; its llama.cpp recommendation is superseded.

## Decision (Mark, 2026-11-05): use the Ollama service he already runs

- Endpoint `http://10.10.10.6:11434`, model tag `qwen3.8:27b` — 27.3 B parameters, Q4_K_M, 17,741,872,154 B, digest `aaee06c39dcf2437cde036998d960e1fc1494b8191be7cc9657d01e509097813`, as reported by `GET /api/tags`.
- listn ships no weights, installs no runtime and never runs `ollama pull`: it is an HTTP client of that one endpoint, and the model's licence is Mark's to have checked on his own host.
- This supersedes the llama.cpp/`llama-cpp-python` recommendation below, the pinned Qwen3-8B-GGUF acquisition plan, and the gate's "zero outbound connections" wording (deviations listed at the end).

### What shipped

| Setting (env override) | Meaning |
|---|---|
| `llm_backend` (`LISTN_LLM_BACKEND`) | `auto` (default) = MLX when installed → Ollama when an endpoint is configured → extractive. `ollama` / `mlx` force one and fail loudly at settings load if incomplete; `extractive` forces the old behaviour. |
| `llm_ollama_url` (`LISTN_OLLAMA_URL`) | bare `http(s)://host[:port]` origin. Empty by default — that emptiness *is* the opt-in gate. |
| `llm_ollama_model` (`LISTN_OLLAMA_MODEL`) | the tag served there, e.g. `qwen3.8:27b`. |

- Seam: `listn.llm.generate(..., settings=..., backend=...)`, chosen by `listn.llm.notes_backend` in `summary.build_notes` and `ask`. The request is `POST <origin>/api/chat` with `stream:false`, `think:false` (Qwen3 thinking otherwise spends the note budget on a private reasoning field and can leave `content` empty) and `options.num_predict` set to the note budget. Proxies are bypassed so a stray `HTTP_PROXY` cannot reroute a transcript, and redirects are never followed: a 3xx is answered as an `OllamaError` with its status, so the allowlisted origin stays the only endpoint listn ever contacts (security review F1).
- Honest labels: notes report `backend:"ollama"` and `model:<tag>` only when that run really used it. Every failure path returns the extractive notes labelled `backend:"extractive"` — never blank, never mislabelled.
- Fallback matrix, test-covered: connection refused, HTTP 5xx, refused 3xx redirect, non-JSON body, empty `content`, timeout, unparseable model reply → extractive output with the transcript intact.
- Privacy: logs carry token counts and durations only; `OllamaError` carries origin + HTTP status only.

### Measured (2026-11-05, against Mark's service, single-pass fixture note)

| Metric | Measured |
|---|---|
| Result | `backend: ollama`, `model: qwen3.8:27b`, 1 chunk, grounded actions/decisions |
| Prompt / generated tokens | 983 / 599 |
| Generation speed | 4.8 tok/s (listn's own log line) |
| Wall time | 150.9 s — inside the gate's ≤180 s, but with little headroom |
| listn-process peak RSS | 23.7 MiB (HTTP client only) |
| Model-side RSS | not measured: the weights live on the Ollama host, which listn does not own |
| Unreachable endpoint | falls back to `backend: extractive` with the transcript intact, no transcript text in logs |

Stated because the margin is thin: 150.9 s is a *single-chunk* note over a 517-character transcript. A meeting longer than `SINGLE_PASS_CHARS` issues one request per chunk plus a reduce call, so multi-chunk wall time is **not measured** and would be expected to exceed 180 s at this speed. The feature therefore stays opt-in and no parity or performance claim is made.

### QA acceptance findings (2026-11-05/06, `t_2bb863cd` on PR #23): timeout, cold-start fallback, multi-chunk (F1/F2)

- **Client budget:** `OLLAMA_TIMEOUT_SECONDS = 180.0` (`listn/llm.py`) applies to one `/api/chat` request via `opener.open(request, timeout=...)`. It is deliberately the same number as the gate's ≤180 s envelope — pinned, with no headroom for a model load.
- **Keep-alive / cold path:** listn sends no `keep_alive`, so the server default applies — observed twice at **5 min** (`expires_at` = last use + 5 m, e.g. `2026-11-06T03:24:20Z` for a 22:19:20 use). Any note after ≥5 idle minutes pays a model load first (the load itself is fast, ~17 s: resident again 17 s after a cold run started).
- **Cold fallback (F1), measured on this box, product path unmodified:**

  | run | model state at start | wall | result |
  |---|---|---|---|
  | 22:04:46 (probe C) | cold (last use 21:40 → unload) | **180.1 s** | `TimeoutError` → `backend: extractive`, transcript intact |
  | 22:16:20 (probe D, `D0` asserts resident=False) | cold, verified unloaded | **180.1 s** | `TimeoutError` → `backend: extractive`, transcript intact |
  | 22:08:25 (probe C re-run) | resident | **165.0 s** | `backend: ollama`, 1149 prompt / 783 generated, 4.8 tok/s |

- **Token-count dependency:** at the measured 4.8 tok/s a ~900-token note needs ~163 s of pure generation, so load + prompt eval + generation lands at ~185 s for the 783-token fixture — over budget — while the implementer's smaller 599-token fixture fits at 150.9 s warm. Wall time scales with generated tokens; the envelope has no room for a cold load on a long note.
- **Safe degradation:** the cold path is not a privacy or honesty defect — the timeout surfaces as `OllamaError`, `summary.build_notes` falls back to extractive notes labelled `backend: "extractive"`, the transcript is intact, and no transcript text reaches the logs.
- **Open owner question:** whether `OLLAMA_TIMEOUT_SECONDS` stays pinned to the gate's envelope or carries headroom (e.g. envelope + load allowance) is an owner decision, not an implementer one. Headroom trades slower worst-case notes for fewer silent cold fallbacks.
- **Multi-chunk (F2), explicitly unassessed:** per criterion 4 no pass is recorded for multi-chunk meetings. At this speed an N-chunk meeting is N requests plus a reduce call, so end-to-end time is unbounded by any 180 s envelope. The single-chunk figures above must not be read as covering it; evidence in `t_2bb863cd` attachments (`probe_c.out`, `probe_c_warm.out`, `probe_d.out`, `probe_c_measurements.json`, `probe_d_measurements.json`).

### Deviations from the gate, forced by the owner decision

1. "Zero outbound connections during generation" → **no connections except the single configured endpoint** (endpoint validation plus a socket-egress test; zero connections when no endpoint is configured).
2. "≤9 GB peak RSS, unload ≤300 s" described a model process listn would have owned. With Ollama, listn-side peak RSS is measured instead (23.7 MiB) and unload is the Ollama host's own idle policy.
3. Offline weight checksum/allowlist is moot: listn obtains no weights. The endpoint string itself is the allowlist — bare origin, no credentials, path or query, no proxy.

## Recommendation at a glance

Prototype **llama.cpp** first, as an optional Linux backend behind the existing `listn.llm.generate(prompt, model_id, max_tokens, system, temperature) -> str` seam, with a single-thread executor and idle unload already in place.[17] Start with an explicitly approved, locally staged Qwen3-8B Q4_K_M GGUF (5.03 GB on the model card) only if Mark accepts its storage and laptop-RAM budget; treat that size as weights-on-disk, not total inference memory.[7] This recommendation is about the best fit for a controlled, local-first integration—not a measured claim that its notes are as good or fast as the macOS MLX default.

**Until a backend is selected, installed, and verified, retain `extractive` as the Linux behavior and label it honestly.** Model and runtime installation must not trigger an unapproved weight download. The repo's current MLX path does permit a first-use Hub download (`listn/llm.py`, `listn/hub.py`); this brief does not authorize extending that behavior to Linux, as distinct from model-side one-time setup.[12]

| Option | Quality/size candidate | Laptop memory (planning estimate, not benchmark) | CachyOS install/update | Integration fit |
|---|---|---|---|---|
| llama.cpp | Qwen3-8B Q4_K_M GGUF: 5.03 GB weights; 8.2B parameters, Qwen3 instruct model.[6][7] | At least weight storage plus runtime/context/KV-cache overhead; plan for more than 5 GB resident. Exact peak depends on context, batch, backend, and GPU offload; measure on target hardware. | Build/pin a tested binary or use a trusted Arch/CachyOS package if Mark chooses one; update separately from model. CachyOS uses Arch-style package-management options, but this brief does not verify a llama.cpp package in its repos.[11] | Best fit if shipped as an optional embedded Python binding, or a listn-managed loopback child process. Directly supports Qwen3/GGUF. More packaging/model lifecycle work than Ollama. |
| Ollama | Same Qwen3 family has an Ollama route; exact tag quantization and size must be pinned and checked before approval.[4][7] | Quantized 8B class is a plausible laptop target, but actual resident memory, default context, and KV cache depend on the chosen tag/configuration; do not infer RAM from the label alone.[5] | Official Linux install/update is a downloaded installer script or archive plus optional systemd service—not the normal app-venv-only install path.[3][5] Whether Mark wants that system-wide service is open. | Lowest initial adapter work: local OpenAI-compatible `/v1/chat/completions`; however, listn would depend on a separately managed daemon, its bind settings, and model naming.[4] |
| Native CTranslate2 LLM | CT2 docs list Llama, Qwen 2.5, and Qwen 3 as supported Transformers architectures; model conversion required.[9] | Int8 weight storage has an approximate 1-byte/parameter floor (~8 GB for 8B weights) before tokenizer, runtime, and KV-cache; actual memory and usable CPU speed are unmeasured. CT2 documents int8 CPU support on x86-64 with MKL/oneDNN, and multiple quantization modes.[10] | Python package can live in an optional `uv` extra, but generation/conversion also requires compatible Transformers tooling and native wheels/build support. Keep conversion a separate explicit operator step, not startup work.[8][9] | Possible direct Python `Generator` adapter, but requires CT2-format conversion, chat-template handling, tokenization, generation stopping/decoding and tested Qwen3 behavior. More bespoke work than llama.cpp's established GGUF/server path.[9] |

## Option details

### 1. llama.cpp — preferred candidate to prototype

- **Quality and model size:** Qwen3-8B is an 8.2B-parameter instruct-capable model; Qwen's official GGUF card lists Q4_K_M at 5.03 GB and supports Qwen3 through llama.cpp.[6][7] This is a concrete, comparable pilot candidate, not evidence of parity with the existing Qwen3-30B-A3B 4-bit MLX default. The source card describes Qwen3's thinking/non-thinking modes; notes should use non-thinking mode and constrain output tokens so deliberation does not leak into notes or waste time.[6]
- **Memory:** 5.03 GB is the GGUF file size only.[7] Leave headroom for context KV cache, temporary buffers, tokenizer and Python; CPU-only and partial GPU-offload budgets vary. No memory or latency has been measured on Mark's laptop, and the 30B-A3B MLX result in README is a different model/hardware.
- **CachyOS install/update:** build/pin llama.cpp or use an explicitly selected Arch-compatible package; the evidence here establishes CachyOS has Pacman and other update flows, not a package's current availability or maintenance status.[11] Do not silently compile or download an executable on first notes generation. Keep runtime updates separate and documented.
- **Offline / downloads:** once the executable and approved GGUF are staged, inference can be local/offline. Make acquisition explicit: user chooses the model and approves the source, license, exact revision/quantization and expected bytes; an explicit setup/download command fetches only that approved artifact into a private cache under the listn data dir (0700), verifies checksum, then the runtime uses the local path with no Hub lookup. Missing/unapproved weights must not trigger a fetch. Model card links to Hugging Face and describes a `-hf` convenience path; listn must not use an auto-fetch convenience route.[7]
- **License:** llama.cpp runtime is MIT; the proposed Qwen3 weights are separately Apache-2.0. A GGUF quantization's provenance/license should be checked too; runtime licensing does not grant model rights.[1][13]
- **Seam / wiring:** keep platform choice internal to `listn.llm`, with the current call signature and shared single-worker lifecycle. Add an optional Linux implementation that accepts only an allowlisted local GGUF path, applies its compatible chat template, normalizes stop/output behavior, and returns plain text to existing JSON parsing/grounding. Prefer a library binding for in-process load/unload; if using `llama-server`, listn must spawn and supervise its own child bound to `127.0.0.1`, use a random/private port, and stop it when idle—never attach to arbitrary ambient services.[2] Keep logs free of prompts, transcripts, outputs, credentials and tokens.
- **Fallback:** absent optional runtime, absent approved weights, load/generation error, invalid model response after existing retry: use current `extractive_notes` and retain `backend: "extractive"`. Show a clear feature-unavailable/error reason without logging transcript content. Do not represent extractive notes as generated.

### 2. Ollama — easiest local API, weaker ownership fit

- **Quality and size:** Ollama documents Qwen3 access via its local OpenAI-compatible API; Qwen's official GGUF card also gives a Q4_K_M size reference for the underlying 8B family.[4][7] Do not assume an Ollama model tag uses that exact quantization/file size: approve and pin an exact tag and inspect its manifest/size before recommending a download.
- **Memory:** same broad 8B quantized laptop class, with additional uncertainty from Ollama tag defaults, context settings, and service runtime. Ollama exposes model process size and CPU/GPU placement for inspection; that is an operational metric, not a guarantee for a different device or tag.[5] Measure peak RSS and context settings on target hardware before setting support claims.
- **CachyOS install/update:** the documented Linux path downloads/runs Ollama's install script or extracts its archive, and can create a systemd service; updates rerun the installer.[3][5] This is less contained than an optional Python package and raises decisions about system-wide privileges, service account, autostart and service update ownership. Do not configure `OLLAMA_HOST=0.0.0.0`; bind to loopback only.[5]
- **Offline / downloads:** local API use needs a model already downloaded, and `ollama pull`/`ollama run` can acquire weights.[4] Only a user-confirmed setup step may pull a selected, pinned model; listn itself must never run a pull or rely on `run` implicitly fetching. Ollama's documented local workflow explicitly pulls a model first.[16] With approved weights present, local inference can run offline, but cloud features/account sign-in are expressly out of scope.[4]
- **License:** Ollama runtime is MIT; each model's license is separate. Qwen3-8B is Apache-2.0; an exact Ollama distribution/tag still needs an audit before approval.[6][15]
- **Seam / wiring:** easy HTTP adapter to `http://127.0.0.1:11434/v1/chat/completions` or Ollama's native API, mapping prompt/system/max tokens/temperature and extracting message content.[4] A system service may not respect listn's own private data dir or lifecycle; validate loopback binding, Host/origin isolation, outbound behavior and no cloud routing. Don't auto-start/reconfigure system Ollama without an explicit operator decision.
- **Fallback:** service unavailable, no approved local model, unexpected cloud/model route, request failure or malformed notes → existing extractive fallback, marked `extractive`. Never silently pull a model or claim that an HTTP request succeeded as a notes-generation success.

### 3. CTranslate2 native Python path — technically viable, more bespoke

- **Quality and size:** CT2 lists Qwen 3 and Llama among supported Transformers architectures and exposes a conversion pipeline; that establishes architecture support, not verified Qwen3 instruction/chat-template behavior in listn.[9] For an 8.2B model, int8 weights alone are roughly 8 GB by arithmetic; context/KV cache and runtime add memory. Q4 AWQ in CT2 is GPU-limited according to its quantization guide, so it is not an easy CPU-laptop route.[10]
- **Memory/performance:** CT2 supports int8 CPU on x86-64 when its Intel MKL or oneDNN backend is available; ARM and other architectures have different support and conversions. An 8B int8 CPU model may fit some laptops, but no RAM, speed, or quality result for target hardware has been measured.[10]
- **CachyOS install/update:** CTranslate2 is MIT and has Python bindings; optional packaging could lock the Python runtime in the app environment. In contrast, converting a checkpoint pulls/uses Transformers and model conversion dependencies and produces another local artifact; pin and test Python, CTranslate2, Transformers and conversion versions.[8][9][14]
- **Offline / downloads:** conversion and tokenizer/model staging can involve network access, but must be a separate explicit, approved operator action. Pin source repo/revision, weights, license, conversion options, output directory and checksums; runtime loads only the converted local path and tokenizer with offline-only mode. Do not invoke `ct2-transformers-converter` or `from_pretrained` on a normal notes request.[9]
- **License:** CTranslate2 runtime is MIT; Qwen3 source weights are Apache-2.0; derived weights retain applicable model terms.[13][14]
- **Seam / wiring:** implement a Linux-specific `Generator` adapter on the existing worker thread and idle unload schedule. Add explicit tokenizer/template handling (including Qwen3 `/no_think` or disabled-thinking template where supported), encode prompt and special tokens, call generator with bounded max length, decode only the answer, and handle EOS/stop IDs. Convert and test the exact chosen checkpoint before committing to the route.[6][9]
- **Fallback:** missing import, unconverted/unapproved files, unsupported CPU features, load/generation failure or unparsable result → `extractive`; no attempt to convert/download at runtime.

## Shared product and security constraints

- No remote/cloud API, telemetry, background model fetch, implicit package install, or first-use model download. A model requires an affirmative Mark decision before its weights are obtained; the explicit installer/acquisition command must disclose model, license, source/revision, quantization, size/checksum and cache location, then run offline afterward.
- Keep weights/cache under a private data subdirectory (0700 directory and restrictive file permissions), separate from transcripts as appropriate. Do not put models in Git, logs, or support bundles. Do not log prompts, transcript fragments, generated notes, tokens, or request bodies.
- Any local HTTP child must bind to `127.0.0.1` only; preserve server Host/origin checks and never trust an arbitrary local endpoint as a substitute for private process ownership.
- Inference failure is not recording failure and must not overwrite valid extractive output with empty/partial generated content. Keep notes provenance `backend` accurate and preserve the existing JSON validation, grounding and fallback checks.
- Do not touch macOS behavior or change its MLX model/download flow in this decision task.

## Decisions Mark must make before implementation

1. **Approve a Linux runtime path?** Recommendation: approve an isolated llama.cpp prototype, or request Ollama's user-managed daemon instead. Keep extractive as default until a measured release decision.
2. **Approve a model and terms?** Proposed evaluation candidate: Qwen3-8B Q4_K_M GGUF, 5.03 GB source-file size, Apache-2.0 weights. Confirm model family, exact upstream repository/revision, quantization, license, and whether to obtain weights at all.[7][13]
3. **Memory and latency budget:** Which laptop/RAM/CPU/GPU is the support target, what peak RAM and generation latency are acceptable, and should inference be CPU-only or permit GPU offload? No hardware evidence is included here.
4. **Installation ownership:** allow an optional app-managed llama.cpp binary/library, build from source, or require system package management? For Ollama, approve a separate service/system install and its update/autostart policy? For CTranslate2, approve added Python/native dependencies and conversion pipeline?
5. **Explicit acquisition UX:** who runs the one-time fetch, where model files live, how the user confirms license/checksum/storage, and whether the feature is disabled until that action completes?
6. **Acceptance evidence:** approve a fixed real/synthetic transcript fixture set and review criteria for grounded summary quality, malformed output/retry/fallback, peak RSS and latency on the target Linux laptop. Until those results exist, no Linux generative parity/performance claim.

## Sources

[1] https://github.com/ggml-org/llama.cpp
[2] https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
[3] https://docs.ollama.com/linux
[4] https://docs.ollama.com/api/openai-compatibility
[5] https://docs.ollama.com/faq
[6] https://huggingface.co/Qwen/Qwen3-8B
[7] https://huggingface.co/Qwen/Qwen3-8B-GGUF
[8] https://github.com/OpenNMT/CTranslate2
[9] https://github.com/OpenNMT/CTranslate2/blob/master/docs/guides/transformers.md
[10] https://opennmt.net/CTranslate2/quantization.html
[11] https://wiki.cachyos.org/configuration/post_install_setup
[12] https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md
[13] https://huggingface.co/Qwen/Qwen3-8B/raw/main/LICENSE
[14] https://raw.githubusercontent.com/OpenNMT/CTranslate2/master/LICENSE
[15] https://github.com/ollama/ollama/raw/main/LICENSE
[16] https://docs.ollama.com/quickstart
[17] https://github.com/margosmark/listn/blob/main/listn/llm.py
