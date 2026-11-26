"use strict";
// Chat UI + training-history sidebar. Streams replies from /api/chat (SSE over fetch),
// lists trained checkpoints from /api/runs, and can switch tinygpt to an older one.

const $ = (id) => document.getElementById(id);
const chat = $("chat"), empty = $("empty"), banner = $("banner");
const backendSel = $("backend"), modelSel = $("model"), modelWrap = $("modelWrap");
const input = $("input"), sendBtn = $("send"), stopBtn = $("stop");
const runsBox = $("runs"), sidebar = $("sidebar"), scrim = $("scrim");

let messages = [];
let streaming = false;
let controller = null;
let status = {};

// ---- theme (light by default; persisted) -----------------------------------
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("themeBtn").innerHTML = t === "dark" ? "&#9681;" : "&#9680;";
  try { localStorage.setItem("advat-theme", t); } catch {}
}
applyTheme((() => { try { return localStorage.getItem("advat-theme") || "light"; } catch { return "light"; } })());
$("themeBtn").addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

// ---- helpers ---------------------------------------------------------------
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
function scrollDown() { chat.scrollTop = chat.scrollHeight; }
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// Escape first, then allow only **bold**; drop gpt-oss citation markers like 【functions.x】.
function formatAnswer(s) {
  return esc(s.replace(/【[^】]*】/g, "")).replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
}

function addBubble(role, text = "") {
  empty.hidden = true;
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const av = document.createElement("div");
  av.className = "avatar"; av.textContent = role === "user" ? "You" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble"; bubble.textContent = text;
  wrap.append(av, bubble); chat.appendChild(wrap); scrollDown();
  return bubble;
}

function setStreaming(on) {
  streaming = on; sendBtn.disabled = on; stopBtn.hidden = !on; input.disabled = on;
  // A reply can change backend state (tinygpt loads on first use), so re-read the banner.
  if (!on) refreshStatus();
}

// ---- training history ------------------------------------------------------
function sparkline(curve) {
  const vals = curve.map((p) => p.val).filter((v) => Number.isFinite(v));
  if (vals.length < 2) return "";
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const W = 240, H = 32;
  const pts = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * W;
    const y = H - ((v - lo) / span) * (H - 4) - 2;  // lower loss = higher on the chart
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.5"
      stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}

function fmtVal(v) { return (v === null || v === undefined) ? "–" : Number(v).toFixed(3); }

function runCard(r) {
  const c = r.config || {};
  const arch = (c.n_layer != null)
    ? `L${c.n_layer}/H${c.n_head}/E${c.n_embd}` : "";
  const params = r.params ? `${(r.params / 1e6).toFixed(2)}M` : "";
  const iters = r.iters != null ? `${r.iters.toLocaleString()} it` : "";
  const mins = r.minutes != null ? `${Number(r.minutes).toFixed(1)} min` : "";
  const meta = [params, arch, iters, mins].filter(Boolean).join(" · ");
  const el = document.createElement("div");
  el.className = "run" + (r.active ? " active" : "");
  el.dataset.path = r.path;
  el.innerHTML = `
    <div class="run-top">
      <span class="run-name" title="${esc(r.name)}">${esc(r.name)}</span>
      ${r.active ? '<span class="badge">active</span>' : ""}
    </div>
    <div class="run-val">val <b>${fmtVal(r.best_val)}</b></div>
    <div class="run-meta">${esc(meta)}</div>
    ${sparkline(r.curve || [])}
    ${r.error ? `<div class="run-err">⚠ ${esc(r.error)}</div>` : ""}
    <div class="run-foot">
      <span class="run-date">${esc(r.trained_at_label || "")}</span>
      <button class="run-use">Use</button>
    </div>`;
  el.querySelector(".run-use").addEventListener("click", (e) => { e.stopPropagation(); useRun(r.path); });
  el.addEventListener("click", () => useRun(r.path));
  return el;
}

async function loadRuns() {
  try {
    const r = await fetch("/api/runs");
    const data = await r.json();
    $("runsRoot").textContent = "in " + (data.root || "checkpoints/");
    runsBox.innerHTML = "";
    if (!data.runs.length) {
      runsBox.innerHTML = `<p class="runs-empty">No trained models yet.<br>
        Train one with <code>python3 train.py</code> — it saves to <code>checkpoints/</code>
        and will show up here.</p>`;
      return;
    }
    for (const run of data.runs) runsBox.appendChild(runCard(run));
  } catch {
    runsBox.innerHTML = `<p class="runs-empty">Could not load history.</p>`;
  }
}

async function useRun(path) {
  try {
    const r = await fetch("/api/use_ckpt", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await r.json();
    if (data.error) { banner.hidden = false; banner.textContent = "⚠ " + data.error; return; }
    backendSel.value = "tinygpt";          // switch to the local model you just picked
    await refreshStatus(); loadModels(); updateHint(); await loadRuns();
    closeSidebar();
  } catch (e) {
    banner.hidden = false; banner.textContent = "⚠ " + e.message;
  }
}

// ---- backend status + model list ------------------------------------------
async function refreshStatus() {
  try {
    const r = await fetch("/api/health");
    const data = await r.json(); status = {};
    for (const b of data.backends) status[b.name] = b;
  } catch {}
  updateBanner();
}

function updateBanner() {
  const b = status[backendSel.value];
  modelWrap.style.display = backendSel.value === "groq" ? "" : "none";
  if (!b) { banner.hidden = true; return; }
  banner.classList.toggle("ok", b.available);
  if (!b.available) banner.textContent = "⚠ " + b.reason + "  —  " + b.describe;
  else banner.textContent = (backendSel.value === "groq" ? "☁ " : "🔒 ") + b.describe;
  banner.hidden = false;
}

// Groq's /models list has no "type" field, so classify by id. Only `chat` entries can
// answer in this UI; the others are listed for visibility but will error if picked.
function modelKind(id) {
  const s = id.toLowerCase();
  if (/whisper/.test(s)) return { label: "speech-to-text", chat: false };
  if (/orpheus|tts|playai/.test(s)) return { label: "voice (text-to-speech)", chat: false };
  if (/guard/.test(s)) return { label: "safety filter", chat: false };
  if (/embed/.test(s)) return { label: "embeddings", chat: false };
  if (/allam/.test(s)) return { label: "chat · Arabic", chat: true };
  return { label: "chat", chat: true };
}

async function loadModels() {
  if (backendSel.value !== "groq") { modelSel.innerHTML = ""; return; }
  modelSel.innerHTML = "<option>loading…</option>";
  try {
    const r = await fetch("/api/models?backend=groq");
    const { models, default: def } = await r.json();
    modelSel.innerHTML = "";
    if (!models.length) { modelSel.innerHTML = `<option value="">(no models — set GROQ_API_KEY)</option>`; return; }
    // Chat models first, then the rest; each option shows its type after the id.
    const sorted = [...models].sort((a, b) =>
      (modelKind(a).chat ? 0 : 1) - (modelKind(b).chat ? 0 : 1) || a.localeCompare(b));
    for (const m of sorted) {
      const o = document.createElement("option");
      o.value = m; o.textContent = `${m}  ·  ${modelKind(m).label}`;
      if (m === def) o.selected = true; modelSel.appendChild(o);
    }
  } catch { modelSel.innerHTML = `<option value="">(could not load models)</option>`; }
}

// ---- send / stream ---------------------------------------------------------
async function send(text) {
  messages.push({ role: "user", content: text });
  addBubble("user", text); input.value = ""; autosize();
  const bubble = addBubble("assistant", ""); bubble.classList.add("cursor");
  setStreaming(true); controller = new AbortController();

  const body = {
    messages, backend: backendSel.value, model: modelSel.value || undefined,
    temperature: parseFloat($("temperature").value),
    max_tokens: parseInt($("maxTokens").value, 10),
    top_k: parseInt($("topK").value, 10),
  };

  let acc = "", errored = false;
  // Lookups (ClinVar, gnomAD, ...) show as small rows above the answer text.
  let toolBox = null, textEl = null;
  const toolRows = new Map();
  const parts = () => {
    if (!textEl) {
      bubble.textContent = "";
      toolBox = document.createElement("div"); toolBox.className = "tools";
      textEl = document.createElement("div"); textEl.className = "answer";
      bubble.append(toolBox, textEl);
    }
  };
  const showTool = (t) => {
    parts();
    const key = `${t.name}|${t.query}`;
    let row = toolRows.get(key);
    if (!row) { row = document.createElement("div"); row.className = "tool"; toolBox.appendChild(row); toolRows.set(key, row); }
    row.replaceChildren();
    row.classList.toggle("bad", t.status === "done" && !t.ok);
    const head = document.createElement("span");
    head.textContent = t.status === "running"
      ? `⏳ Looking up ${t.query || t.name}…`
      : `📚 ${t.label || t.name}: ${t.query} — ${t.summary || ""}`;
    row.appendChild(head);
    for (const url of t.links || []) {
      const a = document.createElement("a");
      a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer";
      try { a.textContent = new URL(url).hostname.replace(/^www\./, ""); } catch { a.textContent = "link"; }
      row.append(" ", a);
    }
    if (t.sources && t.sources.length) row.title = t.sources.join("\n");
    scrollDown();
  };
  try {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: controller.signal,
    });
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, sep); buf = buf.slice(sep + 2);
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        let ev; try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
        if (ev.tool) showTool(ev.tool);
        else if (ev.notice) { parts(); const n = document.createElement("div"); n.className = "tool"; n.textContent = "⏳ " + ev.notice; toolBox.appendChild(n); scrollDown(); }
        else if (ev.delta) {
          acc += ev.delta;
          // Chat models: light formatting. tinygpt: raw text, exactly as generated.
          if (backendSel.value === "groq") { parts(); textEl.innerHTML = formatAnswer(acc); }
          else bubble.textContent = acc;
          scrollDown();
        }
        else if (ev.error) { bubble.textContent = "⚠ " + ev.error; bubble.classList.add("err"); errored = true; }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") { bubble.textContent = "⚠ " + e.message; bubble.classList.add("err"); errored = true; }
  } finally {
    bubble.classList.remove("cursor"); setStreaming(false); controller = null;
    if (!errored && acc) messages.push({ role: "assistant", content: acc });
    else if (errored) messages.pop();
    input.focus();
  }
}

// ---- sidebar (mobile) ------------------------------------------------------
function openSidebar() { sidebar.classList.add("open"); scrim.hidden = false; }
function closeSidebar() { sidebar.classList.remove("open"); scrim.hidden = true; }

// ---- events ----------------------------------------------------------------
$("composer").addEventListener("submit", (e) => {
  e.preventDefault(); const t = input.value.trim(); if (t && !streaming) send(t);
});
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
});
stopBtn.addEventListener("click", () => controller && controller.abort());
$("clearBtn").addEventListener("click", () => {
  messages = []; chat.querySelectorAll(".msg").forEach((n) => n.remove()); empty.hidden = false;
});
$("settingsBtn").addEventListener("click", () => { $("settings").hidden = !$("settings").hidden; });
$("refreshRuns").addEventListener("click", loadRuns);
$("menuBtn").addEventListener("click", openSidebar);
scrim.addEventListener("click", closeSidebar);
backendSel.addEventListener("change", () => { updateBanner(); loadModels(); updateHint(); });

function updateHint() {
  const k = status[backendSel.value]?.kind;
  $("emptyHint").textContent = k === "completion"
    ? "tinygpt continues your text (a small local model, not a chat assistant)."
    : "Pick a backend above and send a message.";
}

// ---- training modal --------------------------------------------------------
const trainModal = $("trainModal");
let pollTimer = null;

function openTrain() { trainModal.hidden = false; loadDatasets(); refreshTrainStatus(); }
function closeTrain() { trainModal.hidden = true; if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function loadDatasets(select) {
  try {
    const r = await fetch("/api/datasets");
    const { datasets } = await r.json();
    const sel = $("dataset");
    sel.innerHTML = "";
    if (!datasets.length) {
      sel.innerHTML = `<option value="">(none — prepare one below)</option>`;
    } else {
      for (const d of datasets) {
        const o = document.createElement("option");
        const toks = d.tokens_train ? ` · ${(d.tokens_train / 1e3).toFixed(0)}k tok` : "";
        o.value = d.name; o.textContent = d.name + toks; sel.appendChild(o);
      }
      if (select) sel.value = select;
      if (!$("runName").value) $("runName").value = sel.value + "_run";
    }
  } catch {}
}

async function doPrepare(payload, isFile) {
  const msg = $("prepMsg"); msg.textContent = "Preparing… (downloading / tokenizing)";
  try {
    const opts = isFile ? { method: "POST", body: payload }
                        : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
    const r = await fetch("/api/prepare", opts);
    const data = await r.json();
    if (data.error) { msg.textContent = "⚠ " + data.error; return; }
    msg.textContent = "✓ prepared dataset: " + data.name;
    await loadDatasets(data.name);
  } catch (e) { msg.textContent = "⚠ " + e.message; }
}

function drawCurve(curve) {
  const svg = $("liveCurve");
  const vals = (curve || []).map((p) => p.val).filter((v) => Number.isFinite(v));
  if (vals.length < 2) { svg.innerHTML = ""; return; }
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1, W = 400, H = 90;
  const pts = vals.map((v, i) => `${((i / (vals.length - 1)) * W).toFixed(1)},${(H - ((v - lo) / span) * (H - 10) - 5).toFixed(1)}`).join(" ");
  svg.innerHTML = `<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>`;
}

function renderStatus(s) {
  const live = $("live");
  if (s.state === "idle") { live.hidden = true; return; }
  live.hidden = false;
  const running = s.state === "running";
  $("startTrain").hidden = running; $("stopTrain").hidden = !running;
  const done = s.state === "done";
  const rc = done ? (s.returncode === 0 ? " ✓ finished" : ` ✗ exited (code ${s.returncode})`) : "";
  $("liveStat").innerHTML =
    `<span>${esc(s.name)}</span> · ${running ? "training" : "done"}${rc} · ` +
    `iter <b>${s.iter ?? "–"}</b>/${s.max_iters} · val <b>${s.val != null ? s.val.toFixed(3) : "–"}</b> · ${s.elapsed}s`;
  drawCurve(s.curve);
  $("liveSample").textContent = s.sample ? "sample: " + s.sample : "";
  $("liveLog").textContent = (s.log || []).join("\n");
  $("liveLog").scrollTop = $("liveLog").scrollHeight;
  if (done && pollTimer) { clearInterval(pollTimer); pollTimer = null; loadRuns(); }
}

async function refreshTrainStatus() {
  try { renderStatus(await (await fetch("/api/train/status")).json()); } catch {}
}

async function startTraining() {
  const body = {
    dataset: $("dataset").value, name: $("runName").value,
    preset: $("preset").value, max_iters: parseInt($("maxIters").value, 10), device: $("device").value,
  };
  if (!body.dataset) { $("prepMsg").textContent = "⚠ pick or prepare a dataset first"; return; }
  const r = await fetch("/api/train/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json();
  if (data.error) { $("prepMsg").textContent = "⚠ " + data.error; return; }
  $("prepMsg").textContent = "";
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refreshTrainStatus, 1500);
  refreshTrainStatus();
}

$("trainBtn").addEventListener("click", openTrain);
$("trainClose").addEventListener("click", closeTrain);
trainModal.addEventListener("click", (e) => { if (e.target === trainModal) closeTrain(); });
$("startTrain").addEventListener("click", startTraining);
$("stopTrain").addEventListener("click", async () => { await fetch("/api/train/stop", { method: "POST" }); refreshTrainStatus(); });
$("prepUpload").addEventListener("click", () => {
  const f = $("corpusFile").files[0];
  if (!f) { $("prepMsg").textContent = "⚠ choose a .txt file first"; return; }
  const fd = new FormData(); fd.append("file", f); fd.append("tokenizer", "char");
  if ($("corpusName").value) fd.append("name", $("corpusName").value);
  doPrepare(fd, true);
});
$("dataset").addEventListener("change", () => { $("runName").value = $("dataset").value + "_run"; });

// ---- init ------------------------------------------------------------------
backendSel.value = document.documentElement.dataset.backend || "groq";
(async () => { await refreshStatus(); await loadModels(); await loadRuns(); updateHint(); autosize(); })();
