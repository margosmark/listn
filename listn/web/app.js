const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const LANGUAGES = [
  ["", "Auto-detect"], ["en", "English"], ["es", "Spanish"], ["fr", "French"], ["de", "German"],
  ["it", "Italian"], ["pt", "Portuguese"], ["nl", "Dutch"], ["sv", "Swedish"], ["pl", "Polish"],
  ["tr", "Turkish"], ["ru", "Russian"], ["uk", "Ukrainian"], ["ar", "Arabic"], ["hi", "Hindi"],
  ["ja", "Japanese"], ["ko", "Korean"], ["zh", "Chinese"],
];
const MODEL_NOTES = {
  "tiny": "tiny · fastest", "base": "base · fast", "small": "small · balanced",
  "medium": "medium · accurate", "large-v3": "large-v3 · most accurate", "large-v3-turbo": "large-v3-turbo · default live model on Apple Silicon",
};
const BACKEND_NOTES = {
  "auto": "Auto: MLX on Apple Silicon when installed, otherwise faster-whisper.",
  "faster-whisper": "faster-whisper: CPU (int8), built-in voice activity detection. Works everywhere.",
  "mlx": "MLX: runs on the Apple Silicon GPU. Install with pip install 'listn[mlx]'.",
  "openai-whisper": "openai-whisper: the original implementation (slowest). Install with pip install 'listn[openai]'.",
};
const STAGES = {
  queued: "Queued", waiting: "Waiting for the live recording to finish", decoding: "Reading audio",
  "loading-model": "Loading model", transcribing: "Transcribing", speakers: "Identifying speakers",
  done: "Done", error: "Failed",
};

const state = {
  status: null,
  settings: {},
  live: null,
  liveSegments: [],
  jobs: new Map(),
  pickedFiles: [],
  selected: null,
  query: "",
};

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

function toast(message, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

function clock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(c));
  return node;
}

function highlight(text, query) {
  if (!query) return [text];
  const out = [];
  const lower = text.toLowerCase(), q = query.toLowerCase();
  let i = 0;
  while (true) {
    const j = lower.indexOf(q, i);
    if (j < 0) { out.push(text.slice(i)); break; }
    out.push(text.slice(i, j), el("mark", { text: text.slice(j, j + q.length) }));
    i = j + q.length;
  }
  return out;
}

/* ---------- navigation ---------- */
function showView(name) {
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.view === name));
  $$(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "library") loadSessions();
  history.replaceState(null, "", `#${name}`);
}
$$(".tab").forEach(t => t.addEventListener("click", () => showView(t.dataset.view)));

/* ---------- settings ---------- */
function fillSelect(select, options, value) {
  select.replaceChildren(...options.map(([v, label]) => el("option", { value: v, text: label })));
  select.value = value ?? "";
}

function renderSettings() {
  const s = state.settings, st = state.status;
  const models = st.models.map(m => [m, MODEL_NOTES[m] || m]);
  $$("select[data-setting=model]").forEach(sel => fillSelect(sel, models, s.model));
  $$("select[data-setting=language]").forEach(sel => fillSelect(sel, LANGUAGES, s.language || ""));
  fillSelect($("#backend"), st.backends.map(b => {
    const avail = b === "auto" || st.backend_available[b];
    return [b, avail ? b : `${b} (not installed)`];
  }), s.backend);
  $("#backend-hint").textContent = `${BACKEND_NOTES[s.backend] || ""} Currently using: ${st.resolved_backend}.`;
  $("#save-audio").checked = !!s.save_audio;
  $("#diarize-after").checked = !!s.diarize_after;
  $("#summarize-after").checked = s.summarize_after !== false;
  $("#offer-meetings").checked = s.offer_meetings !== false;
  $("#auto-record").checked = !!s.auto_record_meetings;
  $("#launch-at-login").checked = !!s.launch_at_login;
  if (document.activeElement !== $("#llm-model")) $("#llm-model").value = s.llm_model || "";
  $("#chunk").value = s.chunk_seconds;
  $("#chunk-label").textContent = `${s.chunk_seconds}s`;
  $("#hf-token-state").textContent = s.hf_token ? "A token is saved (or HF_TOKEN is set)." : "Not set. Only needed the first time you label speakers.";
  $("#data-dir").textContent = st.data_dir;
  const diarHint = !st.diarization_available
    ? "Needs the optional diarization extra: pip install 'listn[diarize]'."
    : (!s.hf_token ? "First use downloads the pyannote model; add a Hugging Face token in Settings." : "");
  $("#diarize-hint").textContent = diarHint;
  $("#file-diarize-hint").textContent = diarHint;
  $("#diarize-after").disabled = !st.diarization_available;
  $("#file-diarize").disabled = !st.diarization_available;
}

async function saveSetting(key, value) {
  try {
    state.settings = await api("PUT", "/api/settings", { [key]: value });
    state.status = await api("GET", "/api/status");
    renderSettings();
  } catch (err) { toast(err.message, "error"); }
}

$$("[data-setting]").forEach(input => {
  input.addEventListener("change", () => {
    const key = input.dataset.setting;
    let value = input.type === "checkbox" ? input.checked : input.value;
    if (key === "chunk_seconds") value = Number(value);
    if (key === "language" && value === "") value = null;
    saveSetting(key, value);
  });
});
$("#chunk").addEventListener("input", e => { $("#chunk-label").textContent = `${e.target.value}s`; });
$("#hf-token").addEventListener("change", e => {
  const v = e.target.value.trim();
  e.target.value = "";
  saveSetting("hf_token", v || null);
});
$("#settings-btn").addEventListener("click", () => $("#settings-dialog").showModal());

/* ---------- devices ---------- */
async function loadDevices() {
  const sel = $("#device");
  try {
    const d = await api("GET", "/api/devices");
    const opts = d.inputs.map(dev => {
      const tag = dev.kind === "blackhole" ? " · system audio" : dev.kind === "aggregate" ? " · aggregate" : "";
      return [dev.name, `${dev.name} (${dev.inputs} ch${tag})`];
    });
    if (!opts.length) opts.push(["", "No input devices found"]);
    fillSelect(sel, opts, d.selected || "");
    renderGuidance(d.guidance);
  } catch (err) {
    renderGuidance([{ level: "error", title: "Could not list devices", detail: err.message }]);
  }
}

function renderGuidance(notes) {
  $("#guidance").replaceChildren(...notes.map(n => {
    const parts = n.detail.split(/`([^`]+)`/).map((p, i) => i % 2 ? el("code", { text: p }) : p);
    return el("div", { class: `note ${n.level}` }, el("strong", { text: n.title }), ...parts);
  }));
}

$("#device").addEventListener("change", e => saveSetting("device", e.target.value || null));
$("#refresh-devices").addEventListener("click", () => loadDevices().then(() => toast("Devices rescanned")));
$("#test-device").addEventListener("click", async () => {
  const btn = $("#test-device");
  btn.disabled = true;
  $("#test-result").textContent = "Listening for 2 seconds…";
  $("#test-meters").replaceChildren();
  try {
    const r = await api("POST", "/api/devices/test", { device: $("#device").value || null, seconds: 2 });
    $("#test-result").textContent = r.advice;
    $("#test-meters").replaceChildren(...r.per_channel.map(c => el("div", { class: "ch" },
      el("span", { text: `ch ${c.channel}` }),
      el("div", { class: "bar" }, el("div", { style: `width:${Math.max(0, Math.min(100, (c.peak_db + 60) / 60 * 100))}%` })),
      el("span", { text: `${c.peak_db} dB` }))));
  } catch (err) {
    $("#test-result").textContent = "";
    toast(err.message, "error");
  } finally { btn.disabled = false; }
});

/* ---------- live ---------- */
function renderLive() {
  const live = state.live;
  const active = live && !["done", "error"].includes(live.state);
  const btn = $("#record-btn");
  btn.classList.toggle("recording", !!active);
  btn.querySelector(".record-label").textContent = active ? (live.state === "finishing" || live.state === "diarizing" ? "Finishing…" : "Stop") : "Start";
  btn.disabled = !!active && ["finishing", "diarizing"].includes(live.state);
  btn.setAttribute("aria-label", active ? "Stop recording" : "Start recording");
  $$(".controls select, .controls input, #test-device, #refresh-devices").forEach(c => {
    if (c.id === "diarize-after" && !state.status?.diarization_available) return;
    c.disabled = !!active;
  });
  if (live) {
    $("#timer").textContent = clock(live.elapsed);
    const stats = [];
    if (live.device) {
      const bits = [live.device.name];
      if (live.device.channels) bits.push(`${live.device.channels} ch`);
      if (live.device.samplerate) bits.push(`${(live.device.samplerate / 1000).toFixed(1)} kHz`);
      stats.push(bits.join(" · "));
    }
    if (live.model) stats.push(`${live.model} on ${live.backend}`);
    if (live.pending_chunks > 1) stats.push(`${live.pending_chunks} chunks waiting`);
    if (live.dropped_blocks) stats.push(`${live.dropped_blocks} audio blocks dropped`);
    $("#live-stats").replaceChildren(...stats.map(s => el("span", { text: s })));
  }
  $("#open-live-session").hidden = !(live && live.state === "done");
}

function setLiveStatus(text) { $("#live-status").textContent = text; }

function appendLiveSegments(segs) {
  const box = $("#live-transcript");
  $("#live-empty")?.remove();
  for (const s of segs) {
    const who = s.speaker
      ? el("span", { class: s.speaker === "Me" ? "speaker me" : "speaker", text: `${s.speaker}:` })
      : null;
    box.append(el("div", { class: "seg" }, el("span", { class: "ts", text: clock(s.start) }), el("span", {}, who, document.createTextNode(s.text))));
  }
  if ($("#autoscroll").checked) box.scrollTop = box.scrollHeight;
}

$("#record-btn").addEventListener("click", async () => {
  const live = state.live;
  const active = live && !["done", "error"].includes(live.state);
  try {
    if (active) {
      await api("POST", "/api/live/stop");
      state.live.state = "finishing";
      setLiveStatus("Finishing the last chunks…");
    } else {
      $("#live-transcript").replaceChildren();
      state.liveSegments = [];
      state.live = await api("POST", "/api/live/start", { title: $("#live-title").value.trim() || null });
      setLiveStatus("Loading model… audio is already being captured.");
    }
    renderLive();
  } catch (err) {
    toast(err.message, "error");
    setLiveStatus(err.message);
  }
});
$("#open-live-session").addEventListener("click", () => {
  if (state.live) { showView("library"); openSession(state.live.id); }
});

function setMeter(db) {
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  $("#meter-fill").style.width = `${pct}%`;
}

/* ---------- files ---------- */
function renderPicked() {
  $("#file-picked").replaceChildren(...state.pickedFiles.map(f =>
    el("div", {}, el("span", { text: f.name }), el("span", { class: "muted", text: `${(f.size / 1048576).toFixed(1)} MB` }))));
  $("#file-submit").disabled = !state.pickedFiles.length;
  $("#file-submit").textContent = state.pickedFiles.length > 1 ? `Transcribe ${state.pickedFiles.length} files` : "Transcribe";
}
$("#file-input").addEventListener("change", e => { state.pickedFiles = [...e.target.files]; renderPicked(); });
const dz = $("#dropzone");
["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", e => { state.pickedFiles = [...e.dataTransfer.files]; renderPicked(); });

$("#file-submit").addEventListener("click", async () => {
  const files = state.pickedFiles;
  if (!files.length) return;
  const btn = $("#file-submit");
  btn.disabled = true;
  for (const f of files) {
    btn.textContent = `Uploading ${f.name}…`;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("title", "");
    fd.append("diarize_speakers", $("#file-diarize").checked ? "true" : "false");
    try {
      const job = await api("POST", "/api/files", fd);
      state.jobs.set(job.id, job);
      renderJobs();
    } catch (err) { toast(`${f.name}: ${err.message}`, "error"); }
  }
  state.pickedFiles = [];
  $("#file-input").value = "";
  renderPicked();
});

function renderJobs() {
  const box = $("#jobs");
  if (!state.jobs.size) { box.replaceChildren(el("p", { class: "muted empty-small", text: "No files in progress." })); return; }
  box.replaceChildren(...[...state.jobs.values()].map(j => {
    const indeterminate = ["queued", "waiting", "decoding", "loading-model", "speakers"].includes(j.stage) || !STAGES[j.stage];
    const label = STAGES[j.stage] || j.stage;
    const done = j.stage === "done", failed = j.stage === "error";
    return el("div", { class: "job" },
      el("div", { class: "job-top" },
        el("span", { class: "job-title", text: j.title }),
        done || failed
          ? el("button", { class: "btn ghost small", onclick: () => { showView("library"); openSession(j.id); }, text: "Open" })
          : el("button", { class: "btn ghost small", onclick: () => api("POST", `/api/jobs/${j.id}/cancel`).catch(e => toast(e.message, "error")), text: "Cancel" })),
      el("div", { class: `progress ${indeterminate && !done && !failed ? "indeterminate" : ""}` },
        el("div", { style: `width:${Math.round((j.progress || 0) * 100)}%` })),
      el("span", { class: failed ? "small-text" : "muted small-text", style: failed ? "color:var(--danger)" : null,
        text: failed ? (j.error || "Failed") : done ? "Done" : `${label}${j.stage === "transcribing" ? ` · ${Math.round(j.progress * 100)}%` : ""}` }));
  }));
}

/* ---------- library ---------- */
let searchTimer;
$("#search").addEventListener("input", e => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.query = e.target.value.trim(); loadSessions(); }, 200);
});

async function loadSessions() {
  const box = $("#sessions");
  try {
    const { results } = await api("GET", `/api/sessions?q=${encodeURIComponent(state.query)}`);
    if (!results.length) {
      box.replaceChildren(el("div", { class: "empty" },
        el("p", { text: state.query ? `Nothing matches “${state.query}”.` : "No recordings yet. Start a live session or transcribe a file." })));
      return;
    }
    box.replaceChildren(...results.map(r => {
      const m = r.meta;
      const when = new Date(m.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
      const sub = [when, clock(m.duration), m.source === "live" ? "Live" : (m.source_name || "File")].join(" · ");
      const badge = m.status !== "done" ? el("span", { class: `badge ${m.status}`, text: m.status }) : null;
      const hit = state.query && r.hits.length
        ? el("div", { class: "session-hit" }, `${r.hit_count} match${r.hit_count > 1 ? "es" : ""}: `, ...highlight(r.hits[0].text, state.query))
        : (r.preview ? el("div", { class: "session-hit", text: r.preview }) : null);
      return el("button", { class: `session ${state.selected === m.id ? "active" : ""}`, onclick: () => openSession(m.id) },
        el("div", { class: "session-title" }, el("span", { text: m.title }), badge),
        el("div", { class: "session-sub", text: sub }), hit);
    }));
  } catch (err) { box.replaceChildren(el("div", { class: "empty" }, el("p", { text: err.message }))); }
}

async function openSession(id, seekAt) {
  state.selected = id;
  $$(".session").forEach(s => s.classList.remove("active"));
  const detail = $("#detail");
  let data;
  try { data = await api("GET", `/api/sessions/${id}`); }
  catch (err) { toast(err.message, "error"); return; }
  const { meta, segments, has_audio } = data;
  loadSessions();

  const title = el("input", { class: "title-input", type: "text", value: meta.title, maxlength: "200", "aria-label": "Title" });
  title.addEventListener("change", async () => {
    try { await api("PATCH", `/api/sessions/${id}`, { title: title.value }); loadSessions(); }
    catch (err) { toast(err.message, "error"); }
  });
  title.addEventListener("keydown", e => { if (e.key === "Enter") title.blur(); });

  const when = new Date(meta.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  const metaLine = el("div", { class: "detail-meta" },
    el("span", { text: when }), el("span", { text: clock(meta.duration) }),
    el("span", { text: meta.source === "live" ? `Live · ${meta.source_name || "input"}` : `File · ${meta.source_name || ""}` }),
    meta.model ? el("span", { text: `${meta.model} · ${meta.backend || ""}` }) : null,
    (meta.attendees || []).length ? el("span", { text: meta.attendees.join(", ") }) : null,
    el("span", { text: `${segments.length} segments` }));

  const audio = has_audio ? el("audio", { controls: true, preload: "metadata", src: `/api/sessions/${id}/audio` }) : null;

  const exportMenu = el("div", { class: "export-menu" });
  const menu = el("div", { class: "menu", hidden: true },
    ...[["txt", "Plain text"], ["md", "Markdown"], ["srt", "SubRip subtitles"], ["vtt", "WebVTT subtitles"], ["json", "JSON"]]
      .map(([f, label]) => el("a", { href: `/api/sessions/${id}/export?format=${f}`, download: "" }, label, el("small", { text: `.${f}` }))));
  const exportBtn = el("button", { class: "btn primary small", text: "Export", onclick: e => { e.stopPropagation(); menu.hidden = !menu.hidden; } });
  exportMenu.append(exportBtn, menu);
  document.addEventListener("click", () => { menu.hidden = true; }, { once: true });

  const copyBtn = el("button", { class: "btn ghost small", text: "Copy text", onclick: async () => {
    const res = await fetch(`/api/sessions/${id}/export?format=txt`);
    await navigator.clipboard.writeText(await res.text());
    toast("Transcript copied");
  } });
  const busy = meta.status === "transcribing" || meta.status === "recording";
  const retrans = has_audio ? el("button", { class: "btn ghost small", text: "Retranscribe", disabled: busy, onclick: () => retranscribe(id, false) }) : null;
  const speakersBtn = has_audio && state.status.diarization_available
    ? el("button", { class: "btn ghost small", text: meta.diarized ? "Relabel speakers" : "Label speakers", disabled: busy, onclick: () => retranscribe(id, true) }) : null;
  const reveal = el("button", { class: "btn ghost small", text: "Show in Finder", onclick: async () => {
    const r = await api("POST", `/api/sessions/${id}/reveal`);
    if (!r.ok) toast(r.path);
  } });
  const del = el("button", { class: "btn danger small", text: "Delete", disabled: busy, onclick: async () => {
    if (!confirm(`Delete “${meta.title}” and its audio? This can't be undone.`)) return;
    try {
      await api("DELETE", `/api/sessions/${id}`);
      state.selected = null;
      detail.replaceChildren(el("div", { class: "empty" }, el("p", { text: "Deleted." })));
      loadSessions();
    } catch (err) { toast(err.message, "error"); }
  } });

  const notes = data.notes;
  const notesBox = renderNotes(id, notes, busy);
  const head = el("div", { class: "detail-head" }, title, metaLine, notesBox,
    meta.error ? el("div", { class: "error-box", text: meta.error }) : null,
    audio,
    el("div", { class: "detail-actions" }, exportMenu, copyBtn, retrans, speakersBtn, el("span", { class: "sep" }), reveal, del));

  const speakerIds = [...new Set(segments.map(s => s.speaker).filter(Boolean))];
  if (speakerIds.length) {
    const names = { ...meta.speakers };
    head.append(el("div", { class: "speakers" }, ...speakerIds.map(sid => {
      const inp = el("input", { type: "text", value: names[sid] || sid });
      inp.addEventListener("change", async () => {
        names[sid] = inp.value.trim() || sid;
        await api("PATCH", `/api/sessions/${id}`, { speakers: names });
        openSession(id);
      });
      return el("label", {}, "Rename", inp);
    })));
  }

  const body = el("div", { class: "transcript" });
  if (!segments.length) {
    body.append(el("div", { class: "empty" }, el("p", { text: busy ? "Transcription in progress…" : "No speech was transcribed in this session." })));
  }
  const marks = (data.bookmarks || []).map(mark => ({ kind: "bookmark", ...mark }));
  const rows = segments.map(s => {
    const label = s.speaker ? (meta.speakers?.[s.speaker] || s.speaker) : null;
    const ts = el(audio ? "button" : "span", { class: `ts ${audio ? "seek" : ""}`, text: clock(s.start), title: audio ? "Play from here" : null });
    if (audio) ts.addEventListener("click", () => { audio.currentTime = s.start; audio.play(); });
    const row = el("div", { class: "seg" }, ts,
      el("span", {}, label ? el("span", { class: (s.speaker === "Me" || label === "Me") ? "speaker me" : "speaker", text: `${label}:` }) : null, ...highlight(s.text, state.query)));
    row.dataset.start = s.start; row.dataset.end = s.end;
    return row;
  });
  const markRows = marks.map(mark => {
    const ts = el(audio ? "button" : "span", { class: `ts ${audio ? "seek" : ""}`, text: clock(mark.t) });
    if (audio) ts.addEventListener("click", () => { audio.currentTime = mark.t; audio.play(); });
    const row = el("div", { class: "seg bookmark" }, ts, el("span", { text: mark.note || "Bookmark" }));
    row.dataset.start = mark.t; row.dataset.end = mark.t + 0.4;
    return row;
  });
  const ordered = [...rows, ...markRows].sort((a, b) => +a.dataset.start - +b.dataset.start);
  body.append(...ordered);
  if (audio) {
    audio.addEventListener("timeupdate", () => {
      const t = audio.currentTime;
      rows.forEach(r => r.classList.toggle("active", t >= +r.dataset.start && t < +r.dataset.end));
    });
  }
  detail.replaceChildren(head, body);
  const firstHit = body.querySelector("mark");
  if (seekAt != null && audio) {
    audio.currentTime = seekAt;
    audio.play().catch(() => {});
    const row = rows.find(r => Math.abs(+r.dataset.start - seekAt) < 0.05) || rows.find(r => seekAt >= +r.dataset.start && seekAt < +r.dataset.end);
    if (row) row.scrollIntoView({ block: "center" });
  } else if (firstHit) firstHit.scrollIntoView({ block: "center" });
}

$("#ask-form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = $("#ask-input").value.trim();
  const box = $("#ask-answer");
  if (!question) { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren(el("p", { class: "muted", text: "Looking through the library…" }));
  try {
    const data = await api("POST", "/api/ask", { question });
    const cites = (data.citations || []).map(hit => el("button", {
      class: "cite",
      text: `${hit.title} · ${clock(hit.start)}`,
      title: hit.text,
      onclick: () => openSession(hit.session_id, hit.start),
    }));
    box.replaceChildren(el("p", { text: data.answer }), cites.length ? el("div", { class: "cites" }, ...cites) : null);
  } catch (err) {
    box.replaceChildren(el("p", { text: err.message }));
  }
});

function renderNotes(id, notes, busy) {
  const box = el("div", { class: "notes" });
  const summary = el("textarea", { class: "notes-summary", rows: "4", placeholder: "Summary" });
  summary.value = notes?.summary || "";
  const decisions = el("textarea", { class: "notes-list", rows: "3", placeholder: "One decision per line" });
  decisions.value = (notes?.decisions || []).join("\n");
  const actions = el("textarea", { class: "notes-list", rows: "3", placeholder: "Owner: action" });
  actions.value = (notes?.actions || []).map(a => `${a.owner}: ${a.text}`).join("\n");
  const how = notes ? (notes.edited ? "edited" : (notes.backend || "")) : "";
  box.append(
    el("div", { class: "notes-head" }, el("h3", { text: "Notes" }), how ? el("span", { class: "muted small-text", text: how }) : null),
    el("label", {}, "Summary", summary),
    el("label", {}, "Decisions", decisions),
    el("label", {}, "Action items", actions),
    el("div", { class: "detail-actions" },
      el("button", { class: "btn ghost small", text: "Save notes", onclick: async () => {
        const parsed = actions.value.split("\n").map(line => line.trim()).filter(Boolean).map(line => {
          const cut = line.indexOf(":");
          return cut > 0 ? { owner: line.slice(0, cut).trim(), text: line.slice(cut + 1).trim() } : { owner: "Others", text: line };
        });
        try {
          await api("PUT", `/api/sessions/${id}/summary`, {
            summary: summary.value,
            decisions: decisions.value.split("\n").map(s => s.trim()).filter(Boolean),
            actions: parsed,
          });
          toast("Notes saved");
        } catch (err) { toast(err.message, "error"); }
      } }),
      el("button", { class: "btn ghost small", text: notes ? "Regenerate" : "Summarize", disabled: busy, onclick: async () => {
        try { await api("POST", `/api/sessions/${id}/summary`); openSession(id); }
        catch (err) { toast(err.message, "error"); }
      } })),
  );
  return box;
}

async function retranscribe(id, diarize) {
  try {
    const job = await api("POST", `/api/sessions/${id}/retranscribe`, { diarize });
    state.jobs.set(job.id, job);
    renderJobs();
    toast(diarize ? "Transcribing with speaker labels…" : "Retranscribing with current settings…");
    openSession(id);
  } catch (err) { toast(err.message, "error"); }
}

/* ---------- events ---------- */
function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = msg => {
    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case "live.started": state.live = ev.status; renderLive(); break;
      case "live.level":
        if (state.live) { state.live.elapsed = ev.elapsed; state.live.pending_chunks = ev.pending; state.live.dropped_blocks = ev.dropped; }
        setMeter(ev.level_db); renderLive(); break;
      case "live.status":
        if (state.live && ev.state) state.live.state = ev.state;
        setLiveStatus(ev.message === "Listening" ? "Listening… transcript appears every few seconds." : ev.message);
        renderLive(); break;
      case "live.segments": state.liveSegments.push(...ev.segments); appendLiveSegments(ev.segments); break;
      case "live.error": toast(ev.message, "error"); setLiveStatus(ev.message); break;
      case "live.stopped":
        if (state.live) state.live.state = "done";
        setMeter(-120);
        setLiveStatus(`Saved · ${clock(ev.meta.duration)} · ${state.liveSegments.length} segments.`);
        renderLive();
        if (state.selected === ev.session || $("#view-library").classList.contains("active")) loadSessions();
        if (state.selected === ev.session) openSession(ev.session);
        break;
      case "summary.ready":
        if (state.selected === ev.session) openSession(ev.session);
        break;
      case "job.update": state.jobs.set(ev.job.id, { ...state.jobs.get(ev.job.id), ...ev.job }); renderJobs(); break;
      case "job.done":
        state.jobs.set(ev.job.id, { ...ev.job, stage: ev.error ? "error" : "done", error: ev.error });
        renderJobs();
        if (ev.error && ev.error !== "Cancelled.") toast(`${ev.job.title}: ${ev.error}`, "error");
        else if (!ev.error) toast(`${ev.job.title} is ready`);
        if ($("#view-library").classList.contains("active")) loadSessions();
        if (state.selected === ev.job.id) openSession(ev.job.id);
        break;
    }
  };
  es.onerror = () => setLiveStatus(state.live && state.live.state !== "done" ? "Reconnecting to listn…" : $("#live-status").textContent);
}

/* ---------- boot ---------- */
async function boot() {
  try {
    state.status = await api("GET", "/api/status");
  } catch (err) {
    toast(`listn server unreachable: ${err.message}`, "error");
    return;
  }
  state.settings = state.status.settings;
  renderSettings();
  await loadDevices();
  for (const j of state.status.jobs) state.jobs.set(j.id, j);
  renderJobs();
  if (state.status.live) {
    const { live, segments } = await api("GET", "/api/live");
    state.live = live;
    state.liveSegments = segments;
    if (segments.length) appendLiveSegments(segments);
    if (!["done", "error"].includes(live.state)) setLiveStatus("Recording…");
  }
  renderLive();
  connectEvents();
  const view = location.hash.slice(1);
  if (["live", "files", "library"].includes(view)) showView(view);
}
boot();
