/* listn dashboard — reads GET /api/dashboard only.
 *
 * Every value rendered here comes from the engine snapshot; there is no
 * mock or fallback data. When the server says nothing is running or no
 * recording exists, the page says exactly that. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);

const STAGES = {
  queued: "Queued", waiting: "Waiting for the live recording to finish", decoding: "Reading audio",
  "loading-model": "Loading model", transcribing: "Transcribing", speakers: "Identifying speakers",
  done: "Done", error: "Failed",
};

const STATE_LABELS = {
  starting: "Starting", "loading-model": "Loading model", listening: "Recording",
  paused: "Paused", finishing: "Finishing", done: "Done", error: "Failed",
  "recording-only": "Recording (no transcription)",
};

const CHIP_CLASS = { recording: "recording", processing: "processing", done: "done", failed: "failed" };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function chip(kind, label) {
  return el("span", { class: `chip ${CHIP_CLASS[kind] || "idle"}` },
    el("span", { class: "dot" }), label);
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function when(created) {
  if (!created) return "";
  const d = new Date(created);
  if (Number.isNaN(d.getTime())) return created;
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/* ---------- live panel ---------- */

function renderTracks(tracks) {
  if (!tracks) return null;
  const row = el("div", { class: "track-row" });
  for (const [name, info] of Object.entries(tracks)) {
    const state = info.state || "unknown";
    const healthy = state === "recording" || state === "stopped";
    const err = state === "failed" || info.error;
    const stateEl = el("div", {
      class: `track-state ${err ? "bad" : healthy ? "ok" : ""}`,
      text: err ? `Failed: ${info.error || state}` : `${state[0].toUpperCase()}${state.slice(1)}`,
    });
    row.append(el("div", { class: "track" },
      el("div", { class: "track-name", text: name === "system" ? "System output" : name === "mic" ? "Microphone" : name }),
      stateEl,
      el("div", { class: "track-detail", text: `${info.seconds != null ? info.seconds.toFixed(1) : "0.0"} s · ${info.frames || 0} frames` })));
  }
  return row;
}

function renderLive(snap) {
  const body = $("#live-body");
  body.replaceChildren();
  const live = snap.live;
  if (!live) {
    body.append(el("div", { class: "empty-block" },
      el("p", { text: "Nothing is recording." }),
      el("p", {}, "The listn server is running but no session is active. Start one from the ", el("a", { href: "/" }, "main app"), ".")));
    return;
  }
  const stateName = STATE_LABELS[live.state] || live.state;
  const kind = live.state === "error" ? "failed"
    : (live.state === "finishing" || live.paused) ? "processing" : "recording";
  const captureLabel = live.device && live.device.kind === "pipewire" ? "PipeWire two-track"
    : live.capture === "ingest" ? "capture socket"
    : live.capture === "pipewire" ? "PipeWire" : "input device";
  const head = el("div", { class: "live-top" },
    chip(kind, stateName),
    el("span", { class: "live-title", text: live.title || "Untitled recording" }),
    el("span", { class: "live-sub", text: `elapsed ${clock(live.elapsed)} · ${captureLabel}` }));
  body.append(head);

  const grid = el("div", { class: "stat-grid" },
    stat("Level", `${live.level_db != null ? live.level_db.toFixed(1) : "—"} dB`),
    stat("Segments", String(live.segments || 0)),
    stat("Queued chunks", String(live.pending_chunks || 0)),
    stat("Dropped blocks", String(live.dropped_blocks || 0)));
  if (live.silent_tracks && live.silent_tracks.length) {
    grid.append(stat("Silent tracks", live.silent_tracks.join(", ")));
  }
  body.append(grid);

  const tracks = renderTracks(live.tracks);
  if (tracks) body.append(tracks);
  else body.append(el("p", { class: "live-sub", text: "Track separation is shown for PipeWire two-track captures." }));
}

function stat(k, v) {
  return el("div", { class: "stat" },
    el("div", { class: "k", text: k }),
    el("div", { class: "v", text: v }));
}

/* ---------- jobs ---------- */

function renderJobs(snap) {
  const body = $("#jobs-body");
  body.replaceChildren();
  if (!snap.jobs || snap.jobs.length === 0) {
    body.append(el("div", { class: "empty-block" }, el("p", { text: "No files are being processed." })));
    return;
  }
  const list = el("div", { class: "rec-list" });
  for (const job of snap.jobs) {
    const stage = STAGES[job.stage] || job.stage;
    const kind = job.stage === "error" ? "failed" : "processing";
    const pct = Math.round((job.progress || 0) * 100);
    list.append(el("div", { class: "rec" },
      chip(kind, stage),
      el("div", { class: "rec-main" },
        el("div", { class: "rec-title", text: job.title || "Untitled" }),
        job.detail ? el("div", { class: "rec-sub", text: job.detail }) : null),
      el("div", { class: "rec-tx" }, el("span", { class: "segs", text: `${pct}%` }))));
  }
  body.append(list);
}

/* ---------- recordings ---------- */

function recordingChip(rec) {
  if (rec.live) return "recording";
  if (rec.status === "recording") return "recording";
  if (rec.status === "transcribing") return "processing";
  if (rec.status === "error") return "failed";
  if (rec.status === "done") return "done";
  return "idle";
}

function transcriptionLabel(rec) {
  const tx = rec.transcription || {};
  if (rec.live) return "Recording";
  if (tx.state === "processing") return tx.job ? `Transcribing · ${Math.round((tx.job.progress || 0) * 100)}%` : "Processing";
  if (tx.state === "failed") return "Transcription failed";
  if (tx.state === "recording") return "Recording";
  if (tx.state === "done") return tx.segments ? "Transcribed" : "No speech recognised";
  return "Unknown";
}

function renderRecordings(snap) {
  const body = $("#recordings-body");
  body.replaceChildren();
  const recs = snap.recordings || [];
  $("#rec-count").textContent = snap.recordings_total
    ? `(${snap.recordings_total} total)` : "";
  if (recs.length === 0) {
    body.append(el("div", { class: "empty-block" },
      el("p", { text: "No recordings yet." }),
      el("p", { text: "When a recording or transcription finishes, it appears here with its tracks and status." })));
    return;
  }
  const list = el("div", { class: "rec-list" });
  for (const rec of recs) {
    const tx = rec.transcription || {};
    const main = el("div", { class: "rec-main" },
      el("div", { class: "rec-title", text: rec.title || "Untitled" }),
      el("div", { class: "rec-sub", text: `${when(rec.created_at)}${rec.source_name ? ` · ${rec.source_name}` : ""}${rec.duration ? ` · ${rec.duration}s` : ""}` }),
      rec.error ? el("div", { class: "error-text", text: rec.error })
        : rec.warning ? el("div", { class: "rec-sub", text: rec.warning }) : null);
    const tracks = el("div", { class: "rec-tracks" },
      el("b", { text: rec.track_summary || "No audio kept" }));
    list.append(el("div", { class: "rec" },
      chip(recordingChip(rec), rec.status_label || rec.status || "Unknown"),
      main,
      tracks,
      el("div", { class: "rec-tx" },
        el("div", { text: transcriptionLabel(rec) }),
        el("div", { class: "segs muted", text: tx.segments != null ? `${tx.segments} lines` : "" }))));
  }
  body.append(list);
}

/* ---------- shell ---------- */

async function refresh() {
  let snap;
  try {
    const res = await fetch("/api/dashboard", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    snap = await res.json();
  } catch (err) {
    $("#live-body").replaceChildren(
      el("div", { class: "empty-block" },
        el("p", { text: "Dashboard unavailable." }),
        el("p", { class: "error-text", text: `Could not reach the listn server: ${err.message}` })));
    return;
  }
  $("#dash-version").textContent = `listn ${snap.version || ""}`;
  $("#dash-platform").textContent = snap.platform || "";
  $("#dash-updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
  $("#dash-foot").textContent = snap.data_dir
    ? `Engine state only — data directory: ${snap.data_dir}` : "";
  renderLive(snap);
  renderJobs(snap);
  renderRecordings(snap);
  // Poll faster while something is happening, slower when idle.
  setTimeout(refresh, (snap.live || (snap.jobs && snap.jobs.length)) ? 2000 : 8000);
}

refresh();
