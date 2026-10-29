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
  $("#vad-chunks").checked = s.vad_chunks !== false;
  $("#match-speakers").checked = s.match_speakers !== false;
  $("#battery-saver").checked = s.battery_saver !== false;
  if (document.activeElement !== $("#retention-days")) {
    $("#retention-days").value = s.audio_retention_days ?? 0;
  }
  const threshold = Number(s.speaker_threshold ?? 0.55);
  $("#speaker-threshold").value = threshold;
  $("#speaker-threshold-label").textContent = threshold.toFixed(2);
  if (document.activeElement !== $("#vocabulary")) {
    $("#vocabulary").value = (s.vocabulary || []).join("\n");
  }
  if (document.activeElement !== $("#llm-model")) $("#llm-model").value = s.llm_model || "";
  $("#chunk").value = s.chunk_seconds;
  $("#chunk-label").textContent = `${s.chunk_seconds}s`;
  $("#hf-token-state").textContent = s.hf_token
    ? "A token is saved (or HF_TOKEN is set). It is sent only to Hugging Face, for the one-time model download."
    : "Not set. Only needed for the first speaker-model download.";
  $("#data-dir").textContent = st.data_dir;
  const diar = st.diarization || { installed: st.diarization_available, ready: false, message: "" };
  $("#speaker-model-state").textContent = diar.message || "";
  let diarHint = diar.message || "";
  if (diar.installed && diar.ready) {
    diarHint = s.diarize_after
      ? `After you stop, Others is split into Speaker 1, Speaker 2… Me stays Me. ${diar.cached || diar.loaded ? "The model loads offline." : diar.message}`
      : "Speaker labels are off. You can still label a saved session from the library.";
    if (s.diarize_after_auto) diarHint += " (Automatic: on while a token is set or the model is cached.)";
  }
  $("#diarize-hint").textContent = diarHint;
  $("#file-diarize-hint").textContent = diar.installed ? (diar.ready ? "" : diar.message) : diar.message;
  $("#diarize-after").disabled = !diar.installed;
  $("#file-diarize").disabled = !diar.installed || !diar.ready;
  if (!$("#file-diarize").dataset.touched) $("#file-diarize").checked = !!(s.diarize_after && diar.ready);
}

async function loadVoices() {
  const box = $("#voices");
  try {
    const data = await api("GET", "/api/speakers");
    $("#speaker-threshold-hint").textContent = data.neural
      ? `Cosine similarity of ${data.method_label} voice prints. 0.55 is calibrated; a name also needs a ${data.margin.toFixed(2)} lead over the next voice. Higher means fewer wrong names.`
      : "The voice model isn't loaded yet, so matching uses the rough spectrum fallback (threshold 0.72). It switches over after the first speaker labelling.";
    $("#reenroll-btn").disabled = !data.neural;
    if (!data.speakers.length) {
      box.replaceChildren(el("small", { class: "muted", text: "None yet. Name a speaker in a saved session and listn will recognise them next time." }));
      return;
    }
    box.replaceChildren(...data.speakers.map(p => el("div", { class: "voice" },
      el("span", { text: p.name }),
      el("small", { class: "muted", text: `${p.clips} sample${p.clips === 1 ? "" : "s"}` }),
      p.needs_reenroll ? el("small", { class: "badge warn", text: p.saved_clips ? "re-learn" : "name again in a meeting" }) : null,
      el("button", { type: "button", class: "btn ghost small", text: "Forget", onclick: async () => {
        if (!confirm(`Forget ${p.name}'s voice and its saved clips?`)) return;
        try { await api("DELETE", `/api/speakers/${encodeURIComponent(p.name)}`); loadVoices(); }
        catch (err) { toast(err.message, "error"); }
      } }))));
  } catch (err) { box.replaceChildren(el("small", { class: "muted", text: err.message })); }
}
$("#reenroll-btn").addEventListener("click", async () => {
  try {
    const r = await api("POST", "/api/speakers/reenroll");
    toast(r.needs_reenroll.length
      ? `Re-learned ${r.reenrolled.length}. Name ${r.needs_reenroll.join(", ")} again in a meeting.`
      : `Re-learned ${r.reenrolled.length} voice${r.reenrolled.length === 1 ? "" : "s"}.`);
    loadVoices();
  } catch (err) { toast(err.message, "error"); }
});
$("#file-diarize").addEventListener("change", e => { e.target.dataset.touched = "1"; });

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
    if (["chunk_seconds", "speaker_threshold", "audio_retention_days", "min_free_gb"].includes(key)) value = Number(value);
    if (key === "language" && value === "") value = null;
    saveSetting(key, value);
  });
});
$("#chunk").addEventListener("input", e => { $("#chunk-label").textContent = `${e.target.value}s`; });
$("#speaker-threshold").addEventListener("input", e => { $("#speaker-threshold-label").textContent = Number(e.target.value).toFixed(2); });
$("#enroll-btn").addEventListener("click", async () => {
  const name = $("#enroll-name").value.trim();
  const file = $("#enroll-file").files[0];
  if (!name || !file) { toast("Type a name and choose a WAV clip.", "error"); return; }
  const body = new FormData();
  body.append("name", name);
  body.append("file", file);
  try {
    const res = await fetch("/api/speakers/enroll", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Could not save that voice.");
    $("#enroll-state").textContent = `${data.name} · ${data.clips} clip${data.clips === 1 ? "" : "s"}`;
    toast(`Learned ${data.name}`);
  } catch (err) { toast(err.message, "error"); }
});
$("#hf-token").addEventListener("change", e => {
  const v = e.target.value.trim();
  e.target.value = "";
  saveSetting("hf_token", v || null);
});
$("#settings-btn").addEventListener("click", () => { loadVoices(); $("#settings-dialog").showModal(); });
$("#vocabulary").addEventListener("change", e => {
  const words = e.target.value.split("\n").map(s => s.trim()).filter(Boolean);
  saveSetting("vocabulary", words);
});

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
  const paused = live && live.state === "paused";
  btn.classList.toggle("recording", !!active);
  btn.querySelector(".record-label").textContent = active ? (live.state === "finishing" || live.state === "diarizing" ? "Finishing…" : "Stop") : "Start";
  btn.disabled = !!active && ["finishing", "diarizing"].includes(live.state);
  btn.setAttribute("aria-label", active ? "Stop recording" : "Start recording");
  const pauseBtn = $("#pause-btn");
  pauseBtn.hidden = !active || ["finishing", "diarizing"].includes(live && live.state);
  pauseBtn.textContent = paused ? "Resume" : "Pause";
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
    const gap = s.text === "[paused]";
    box.append(el("div", { class: gap ? "seg gap" : "seg" }, el("span", { class: "ts", text: clock(s.start) }), el("span", {}, who, document.createTextNode(gap ? "Paused" : s.text))));
  }
  if ($("#autoscroll").checked) box.scrollTop = box.scrollHeight;
}

async function openPrivacy(pane) {
  try {
    const r = await api("POST", "/api/privacy/open", { pane });
    if (!r.opened) toast("Open System Settings > Privacy & Security on the Mac.");
  } catch (err) { toast(err.message, "error"); }
}

function showPrivacyBanner(message, pane) {
  const banner = $("#privacy-banner");
  $("#privacy-message").textContent = message;
  banner.dataset.pane = pane || "microphone";
  banner.hidden = false;
}

$("#privacy-open").addEventListener("click", () => openPrivacy($("#privacy-banner").dataset.pane || "microphone"));

$("#pause-btn").addEventListener("click", async () => {
  const paused = state.live && state.live.state === "paused";
  try { await api("POST", paused ? "/api/live/resume" : "/api/live/pause"); }
  catch (err) { toast(err.message, "error"); }
});

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
      $("#privacy-banner").hidden = true;
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
    const speakerStep = j.stage === "speakers";
    const indeterminate = ["queued", "waiting", "decoding", "loading-model"].includes(j.stage)
      || (speakerStep && !(j.progress > 0.03)) || !STAGES[j.stage];
    const label = speakerStep && j.detail ? j.detail : (j.kind === "speakers" && j.stage === "queued" ? "Queued for speaker labels" : (STAGES[j.stage] || j.stage));
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
        text: failed ? (j.error || "Failed") : done ? "Done" : `${label}${j.stage === "transcribing" || (speakerStep && j.progress > 0.03) ? ` · ${Math.round(j.progress * 100)}%` : ""}` }));
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
  const diar = state.status.diarization || {};
  const speakerJob = state.jobs.get(id);
  const labelling = ["queued", "running"].includes(meta.speaker_state) && speakerJob && !["done", "error"].includes(speakerJob.stage);
  const speakersBtn = has_audio && diar.installed
    ? el("button", { class: "btn ghost small", text: meta.diarized ? "Label speakers again" : "Label speakers",
      disabled: busy || labelling || !diar.ready, title: diar.ready ? "Split Others into Speaker 1, Speaker 2… (Me stays Me)" : diar.message,
      onclick: () => labelSpeakers(id) }) : null;
  const speakerLine = speakerStatus(meta, speakerJob, diar, has_audio);
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
  const notesBox = renderNotes(id, notes, busy, audio);
  const head = el("div", { class: "detail-head" }, title, metaLine, notesBox,
    meta.error ? el("div", { class: "error-box" }, meta.error,
      meta.silent ? el("button", { class: "btn secondary small", text: "Open Privacy Settings", onclick: () => openPrivacy("microphone") }) : null) : null,
    meta.warning ? el("div", { class: "note warn", text: meta.warning }) : null,
    meta.audio_removed ? el("div", { class: "note", text: "The audio was deleted by the retention rule. The transcript and notes are still here." }) : null,
    audio,
    speakerLine,
    el("div", { class: "detail-actions" }, exportMenu, copyBtn, retrans, speakersBtn, el("span", { class: "sep" }), reveal, del));

  const speakerIds = [...new Set(segments.map(s => s.speaker).filter(Boolean))].filter(sid => sid !== "Me");
  if (speakerIds.length) {
    const names = { ...meta.speakers };
    head.append(el("div", { class: "speakers" }, ...speakerIds.map(sid => {
      const inp = el("input", { type: "text", value: names[sid] || sid, maxlength: "60" });
      const match = (meta.speaker_matches || {})[sid];
      inp.addEventListener("change", async () => {
        const person = inp.value.trim();
        if (!person || person === (names[sid] || sid)) return;
        try {
          const r = await api("POST", `/api/sessions/${id}/speakers/name`, { label: sid, name: person });
          toast(r.profile ? `${person} will be recognised in future meetings` : (r.note || `Renamed to ${person}`));
          openSession(id);
        } catch (err) { toast(err.message, "error"); }
      });
      return el("label", { title: match && match.name ? `Recognised by voice (score ${match.score})` : null },
        sid === "Others" ? "Rename Others" : "Who is this?", inp);
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
    const who = el("span", {}, label ? el("span", { class: (s.speaker === "Me" || label === "Me") ? "speaker me" : "speaker", text: `${label}:` }) : null, ...highlight(s.text, state.query));
    const row = el("div", { class: "seg" }, ts, who);
    if (has_audio && s.speaker !== "Me" && label !== "Me" && s.text !== "[paused]") {
      const name = el("input", {
        class: "who", type: "text", placeholder: "This is…", maxlength: "60",
        value: label && label !== "Others" ? label : "",
        "aria-label": "Name this speaker",
      });
      name.addEventListener("change", async () => {
        const person = name.value.trim();
        if (!person) return;
        try {
          await api("POST", "/api/speakers/correct", { session_id: id, start: s.start, name: person });
          toast(`Learned ${person}`);
          openSession(id);
        } catch (err) { toast(err.message, "error"); }
      });
      row.append(name);
    }
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

function renderNotes(id, notes, busy, audio) {
  const box = el("div", { class: "notes" });
  const summary = el("textarea", { class: "notes-summary", rows: "4", placeholder: "Summary" });
  summary.value = notes?.summary || "";
  const points = el("textarea", { class: "notes-list", rows: "3", placeholder: "One key point per line" });
  points.value = (notes?.key_points || []).map(p => p.text).join("\n");
  const decisions = el("textarea", { class: "notes-list", rows: "3", placeholder: "One decision per line" });
  decisions.value = (notes?.decisions || []).join("\n");
  const actions = el("textarea", { class: "notes-list", rows: "3", placeholder: "Owner: action" });
  actions.value = (notes?.actions || []).map(a => `${a.owner}: ${a.text}`).join("\n");
  const how = notes ? (notes.edited ? "edited" : (notes.backend || "")) : "";
  const model = notes?.model && !notes.edited ? notes.model.split("/").pop() : "";

  // Where each item was said, linked to the audio.
  const seekable = (t) => {
    if (t == null) return el("span", { class: "ts", text: "--:--" });
    const ts = el(audio ? "button" : "span", { class: `ts ${audio ? "seek" : ""}`, text: clock(t), title: audio ? "Play from here" : null });
    if (audio) ts.addEventListener("click", () => { audio.currentTime = t; audio.play(); });
    return ts;
  };
  const refs = [];
  for (const a of notes?.actions || []) {
    if (a.start == null && !a.due) continue;
    refs.push({ t: a.start, kind: "Action", text: `${a.owner}: ${a.text}${a.due ? ` (due ${a.due})` : ""}` });
  }
  for (const d of notes?.decision_items || []) if (d.start != null) refs.push({ t: d.start, kind: "Decision", text: d.text });
  for (const q of notes?.open_questions || []) refs.push({ t: q.start, kind: "Open", text: q.text });
  for (const p of notes?.key_points || []) if (p.start != null) refs.push({ t: p.start, kind: "Point", text: p.text });
  refs.sort((a, b) => (a.t ?? 1e9) - (b.t ?? 1e9));
  const refList = refs.length ? el("details", { class: "notes-refs" },
    el("summary", { text: `In the recording (${refs.length})` }),
    ...refs.map(r => el("div", { class: "notes-ref" }, seekable(r.t), el("span", { class: "tag", text: r.kind }), el("span", { text: r.text })))) : null;

  box.append(
    el("div", { class: "notes-head" }, el("h3", { text: "Notes" }),
      how ? el("span", { class: "muted small-text", text: model ? `${how} · ${model}` : how }) : null),
    el("label", {}, "Summary", summary),
    el("label", {}, "Key points", points),
    el("label", {}, "Decisions", decisions),
    el("label", {}, "Action items", actions),
    refList,
    el("div", { class: "detail-actions" },
      el("button", { class: "btn ghost small", text: "Save notes", onclick: async () => {
        const parsed = actions.value.split("\n").map(line => line.trim()).filter(Boolean).map(line => {
          const cut = line.indexOf(":");
          return cut > 0 ? { owner: line.slice(0, cut).trim(), text: line.slice(cut + 1).trim() } : { owner: "Others", text: line };
        });
        try {
          await api("PUT", `/api/sessions/${id}/summary`, {
            summary: summary.value,
            key_points: points.value.split("\n").map(s => s.trim()).filter(Boolean),
            decisions: decisions.value.split("\n").map(s => s.trim()).filter(Boolean),
            actions: parsed,
          });
          toast("Notes saved");
        } catch (err) { toast(err.message, "error"); }
      } }),
      el("button", { class: "btn ghost small", text: notes ? "Regenerate" : "Summarize", disabled: busy, onclick: async (ev) => {
        if (notes?.edited && !confirm("Replace your edited notes with new ones from the transcript?")) return;
        const btn = ev.currentTarget;
        const label = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Writing notes…";
        box.classList.add("notes-busy");
        toast("Writing notes with the local model. This can take up to a minute or two on a long meeting.");
        const started = Date.now();
        try {
          const result = await api("POST", `/api/sessions/${id}/summary`);
          const secs = Math.round((Date.now() - started) / 1000);
          const n = (result && result.actions) ? result.actions.length : 0;
          toast(`Notes updated in ${secs}s (${n} action item${n === 1 ? "" : "s"})`);
          openSession(id);
        } catch (err) {
          toast(err.message, "error");
          btn.disabled = false;
          btn.textContent = label;
          box.classList.remove("notes-busy");
        }
      } })),
  );
  return box;
}

const SPEAKER_STATES = {
  queued: "Speaker labels are queued.",
  running: "Identifying speakers…",
  cancelled: "Speaker labelling was cancelled.",
};

function speakerStatus(meta, job, diar, hasAudio) {
  const active = job && job.kind === "speakers" || (job && job.stage === "speakers");
  if (active && !["done", "error"].includes(job.stage)) {
    const box = el("div", { class: "note speaker-progress", id: "speaker-progress" },
      el("span", { class: "speaker-progress-text" }),
      el("div", { class: "progress" }, el("div", {})),
      el("button", { class: "btn ghost small", text: "Cancel", onclick: () => api("POST", `/api/jobs/${meta.id}/cancel`).catch(e => toast(e.message, "error")) }));
    paintSpeakerProgress(box, job);
    return box;
  }
  const text = meta.speaker_note || SPEAKER_STATES[meta.speaker_state];
  if (text) return el("div", { class: `note ${["error", "skipped"].includes(meta.speaker_state) ? "warn" : ""}`, text });
  if (hasAudio && diar.installed && !diar.ready && !meta.diarized) return el("div", { class: "note", text: diar.message });
  return null;
}

async function labelSpeakers(id) {
  try {
    const job = await api("POST", `/api/sessions/${id}/diarize`);
    state.jobs.set(job.id, job);
    renderJobs();
    toast("Identifying speakers on the Others track…");
    openSession(id);
  } catch (err) { toast(err.message, "error"); }
}

function paintSpeakerProgress(box, job) {
  const pct = Math.round((job.progress || 0) * 100);
  box.querySelector(".speaker-progress-text").textContent = `${job.detail || "Identifying speakers"}${pct > 3 ? ` · ${pct}%` : ""}`;
  const bar = box.querySelector(".progress");
  bar.classList.toggle("indeterminate", pct <= 3);
  bar.firstElementChild.style.width = `${pct}%`;
}

function refreshSpeakerProgress(job) {
  if (state.selected !== job.id) return;
  const box = $("#speaker-progress");
  if (box) paintSpeakerProgress(box, job);
  else if (state.speakerViewFor !== job.id) {
    state.speakerViewFor = job.id;
    openSession(job.id);
  }
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
      case "live.silent":
        showPrivacyBanner(ev.message, ev.pane);
        toast(ev.message, "error");
        break;
      case "live.paused":
        if (state.live) state.live.state = "paused";
        setLiveStatus("Paused. The gap stays in the transcript.");
        renderLive();
        break;
      case "live.resumed":
        if (state.live) state.live.state = ev.state || "listening";
        setLiveStatus("Listening…");
        renderLive();
        break;
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
      case "job.update":
        state.jobs.set(ev.job.id, { ...state.jobs.get(ev.job.id), ...ev.job });
        renderJobs();
        if (ev.job.stage === "speakers") refreshSpeakerProgress(ev.job);
        break;
      case "speakers.skipped":
        toast(ev.message, "error");
        if (state.selected === ev.session) openSession(ev.session);
        break;
      case "job.done":
        state.jobs.set(ev.job.id, { ...ev.job, stage: ev.error ? "error" : "done", error: ev.error });
        renderJobs();
        if (ev.job.kind === "speakers") {
          if (ev.error && ev.error !== "Cancelled.") toast(`${ev.job.title}: ${ev.error}`, "error");
          else if (!ev.error) toast(`${ev.job.title}: ${ev.meta.speaker_note || "speakers labelled"}`);
        }
        else if (ev.error && ev.error !== "Cancelled.") toast(`${ev.job.title}: ${ev.error}`, "error");
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
