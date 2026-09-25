// Du'a Companion: one front end, two engines.
//   server engine: audio streams to app/server.py over a WebSocket
//   device engine: Whisper runs in a Web Worker, tracker.js in the page;
//                  nothing leaves the device (used when there's no server)
import { CorpusIndex, Tracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const SR = 16000;

// -- engines ----------------------------------------------------------------
class ServerEngine {
  kind = "server";
  async prepare() {}
  async start(onUpdate) {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      onUpdate({ dua: m.dua, segment: m.segment, token: m.token, pause: m.pause_at_line_end, heard: m.heard, ms: m.step_ms,
        candidates: (m.candidates || []).map((c) => ({ id: c.id, p: c.p })) });
    };
    await new Promise((ok, err) => ((ws.onopen = ok), (ws.onerror = err)));
    this.ws = ws;
  }
  push(chunk) {
    if (this.ws?.readyState === 1) this.ws.send(chunk.buffer);
  }
  lock(duaId) {
    this.ws?.send(`lock:${duaId}`);
  }
  stop() {
    this.ws?.close();
    this.ws = null;
  }
}

class DeviceEngine {
  kind = "device";
  constructor(corpus, model) {
    this.corpus = corpus;
    this.tracker = new Tracker(new CorpusIndex(corpus));
    this.model = model;
    this.window = 6 * SR;
    this.buf = new Float32Array(this.window);
  }
  prepare(onProgress) {
    if (this.ready) return this.ready;
    this.worker = new Worker("asr-worker.js", { type: "module" });
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = ({ data }) => {
        if (data.type === "progress") onProgress?.(data.progress);
        else if (data.type === "ready") resolve();
        else if (data.type === "error" && !this.onUpdate) reject(new Error(data.message));
        else if (data.type === "text") this._onText(data);
      };
    });
    this.worker.postMessage({ type: "load", model: this.model, webgpu: params.has("webgpu") });
    return this.ready;
  }
  lock(duaId) {
    // Follow only this du'a: identification is skipped entirely.
    this.tracker = new Tracker(new CorpusIndex(this.corpus.filter((d) => d.id === duaId)));
  }
  async start(onUpdate) {
    this.onUpdate = onUpdate;
    this.tracker = new Tracker(new CorpusIndex(this.corpus));
    Object.assign(this, { filled: 0, total: 0, lastSent: 0, lastUpdate: 0, busy: false, live: true });
  }
  push(chunk) {
    const { buf, window: W } = this;
    if (chunk.length >= W) buf.set(chunk.subarray(chunk.length - W));
    else {
      buf.copyWithin(0, chunk.length);
      buf.set(chunk, W - chunk.length);
    }
    this.filled = Math.min(W, this.filled + chunk.length);
    this.total += chunk.length;
    this._maybeSend();
  }
  _maybeSend() {
    // Skip stale hops rather than queue them: only the newest window matters.
    if (!this.live || this.busy || this.total - this.lastSent < SR) return;
    this.busy = true;
    this.lastSent = this.total;
    const audio = this.buf.slice(this.window - this.filled);
    this.worker.postMessage({ type: "transcribe", id: this.total, audio }, [audio.buffer]);
  }
  _onText(data) {
    this.busy = false;
    if (!this.live) return;
    const dt = (data.id - this.lastUpdate) / SR;
    this.lastUpdate = data.id;
    const p = this.tracker.update(data.text, dt);
    this.onUpdate({ dua: p.dua, segment: p.segment, token: p.token, pause: p.atLineEnd && !data.text, heard: data.text, ms: data.ms,
      candidates: p.candidates.map(([id, prob]) => ({ id, p: prob })) });
    this._maybeSend();
  }
  stop() {
    this.live = false;
  }
}

// -- app state ----------------------------------------------------------------
const state = { duas: {}, engine: null, ctx: null, worklet: null, node: null, source: null, stream: null,
  mediaSource: null, dua: null, segment: null, token: null, listeningSince: 0, chosen: null, room: null };

function setState(s) {
  document.body.dataset.state = s;
}

async function init() {
  const corpus = await fetch("corpus.json").then((r) => r.json());
  for (const d of corpus) state.duas[d.id] = d;
  const mode = await fetch("api/mode").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  state.engine = mode?.mode === "server"
    ? new ServerEngine()
    : new DeviceEngine(corpus, params.get("model") || "whisper-base-quran-dua");
  $("footnote").textContent = state.engine.kind === "device"
    ? "Runs entirely on this device. No audio leaves it."
    : "";

  $("start").onclick = () => begin(micSource);
  $("play").onclick = showRecordings;
  $("choose").onclick = openPicker;
  $("picker").onclick = (e) => e.target === $("picker") && closePicker();
  $("search").oninput = () => fillPicker($("search").value);
  addEventListener("keydown", (e) => e.key === "Escape" && closePicker());
  $("file").onchange = (e) => e.target.files[0] && begin(fileSource(URL.createObjectURL(e.target.files[0])));
  $("stop").onclick = end;
  $("menu-btn").onclick = () => ($("menu").hidden = !$("menu").hidden);
  $("opt-en").onchange = (e) => document.body.classList.toggle("no-en", !e.target.checked);
  $("opt-tl").onchange = (e) => document.body.classList.toggle("no-tl", !e.target.checked);
  $("opt-full").onchange = (e) => {
    document.body.classList.toggle("show-full", e.target.checked);
    $("full").hidden = !e.target.checked;
    scrollFull();
  };
  if (params.has("debug")) $("debug").hidden = false;
  // Rooms are relayed by app/server.py, so sharing needs it (not a static host).
  $("share").hidden = mode?.mode !== "server";
  $("share").onclick = share;
  $("share-card").onclick = (e) => e.target === $("share-card") && ($("share-card").hidden = true);
  if (params.get("watch")) watch(params.get("watch"));
}

// -- majlis mode: one phone listens, other screens follow -------------------------
const wsBase = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;

// What the listening phone renders, it also relays to the room (if one is open).
function update(u) {
  render(u);
  if (state.room?.readyState === 1) state.room.send(JSON.stringify(u));
}

async function share() {
  $("menu").hidden = true;
  if (!state.room) {
    // Unambiguous letters only: people read this aloud across a room.
    const code = Array.from({ length: 5 }, () => "ACDEFHJKMNPRTUVWXY"[Math.floor(Math.random() * 18)]).join("");
    const ws = new WebSocket(`${wsBase()}/ws/room/${code}?role=host`);
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if ("viewers" in m) $("share-viewers").textContent = m.viewers ? `${m.viewers} following` : "No one following yet";
    };
    await new Promise((ok, err) => ((ws.onopen = ok), (ws.onerror = err)));
    Object.assign(state, { room: ws, roomCode: code });
    if (state.last) ws.send(JSON.stringify(state.last));
  }
  const url = `${location.origin}${location.pathname}?watch=${state.roomCode}`;
  $("share-code").textContent = state.roomCode;
  $("share-copy").textContent = "Copy link";
  $("share-copy").onclick = async () => {
    await navigator.clipboard?.writeText(url).catch(() => {});
    $("share-copy").textContent = "Copied";
  };
  $("share-card").hidden = false;
  try {
    window.qrcode ?? (await new Promise((ok, err) => {
      const s = Object.assign(document.createElement("script"),
        { src: "https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js", onload: ok, onerror: err });
      document.head.append(s);
    }));
    const qr = window.qrcode(0, "M");
    qr.addData(url);
    qr.make();
    $("qr").innerHTML = qr.createSvgTag({ cellSize: 6, margin: 2, scalable: true });
  } catch {
    $("qr").textContent = url; // offline: the link still works
  }
}

function watch(code) {
  document.body.classList.add("watching");
  $("dua-en").textContent = "Waiting for the reciter";
  setState("listening");
  $("now-tl").textContent = `Following room ${code.toUpperCase()}`;
  const connect = () => {
    const ws = new WebSocket(`${wsBase()}/ws/room/${encodeURIComponent(code)}`);
    ws.onmessage = (e) => {
      const u = JSON.parse(e.data);
      if (u.ended) {
        $("dua-en").textContent = "The reciter has stopped";
        return;
      }
      render(u);
    };
    ws.onclose = () => setTimeout(connect, 2000); // phones sleep and wake: keep trying
  };
  connect();
  $("stop").onclick = () => (location.href = location.pathname);
}

// -- audio ----------------------------------------------------------------------
async function micSource(ctx) {
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true },
  });
  return ctx.createMediaStreamSource(state.stream);
}

function fileSource(url) {
  return async (ctx) => {
    const player = $("player");
    player.src = url;
    if (!state.mediaSource) {
      // A media element can join an audio graph only once: build it once.
      state.mediaSource = ctx.createMediaElementSource(player);
      state.mediaSource.connect(ctx.destination);
    }
    player.onended = end;
    player.play();
    return state.mediaSource;
  };
}

async function showRecordings() {
  if (state.engine.kind === "device") return $("file").click();
  const list = $("samples");
  if (!list.hidden) return (list.hidden = true);
  const recs = await fetch("api/recordings").then((r) => r.json()).catch(() => []);
  const items = recs.map((r) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.textContent = `${r.dua_name} · ${r.reciter}`;
    b.onclick = () => begin(fileSource(`audio/${r.id}`));
    li.append(b);
    return li;
  });
  const own = document.createElement("li");
  const ob = document.createElement("button");
  ob.textContent = "Choose a file…";
  ob.onclick = () => $("file").click();
  own.append(ob);
  list.replaceChildren(own, ...items);
  list.hidden = false;
}

async function begin(makeSource) {
  $("samples").hidden = true;
  if (state.engine.kind === "device") {
    setState("loading");
    $("start").disabled = true;
    $("load").hidden = false;
    $("hint").textContent = "Preparing… (only the first time)";
    try {
      await state.engine.prepare((p) => ($("load-bar").style.width = `${Math.round(p)}%`));
    } catch (e) {
      $("hint").textContent = "Couldn't load the speech model on this device.";
      $("start").disabled = false;
      return setState("idle");
    }
    $("load").hidden = true;
    $("start").disabled = false;
  }
  state.ctx ??= new AudioContext();
  await state.ctx.resume();
  state.worklet ??= state.ctx.audioWorklet.addModule("capture-worklet.js");
  await state.worklet;
  let source;
  try {
    source = await makeSource(state.ctx);
  } catch {
    $("hint").textContent = "Microphone access is needed to follow along.";
    return setState("idle");
  }
  await state.engine.start(update);
  if (state.chosen) state.engine.lock(state.chosen);
  const node = new AudioWorkletNode(state.ctx, "capture");
  node.port.onmessage = (e) => state.engine.push(e.data);
  source.connect(node);
  Object.assign(state, { node, source, dua: null, segment: null, token: null, listeningSince: Date.now() });
  showListening();
}

function end() {
  const { node, source, stream } = state;
  if (source && node) source.disconnect(node);
  if (source === state.mediaSource) $("player").pause();
  stream?.getTracks().forEach((t) => t.stop());
  state.engine?.stop();
  if (state.room?.readyState === 1) state.room.send(JSON.stringify({ ended: true }));
  Object.assign(state, { node: null, source: null, stream: null, dua: null, segment: null, chosen: null });
  $("guesses").hidden = true;
  $("menu").hidden = true;
  $("hint").textContent = "Tap to begin";
  setState("idle");
}

// -- rendering ------------------------------------------------------------------
function showListening() {
  setState("listening");
  $("dua-ar").textContent = "";
  $("dua-en").textContent = "Listening";
  $("progress").style.width = "0";
  $("prev").textContent = $("next").textContent = $("now-ar").textContent = $("now-en").textContent = "";
  $("now-tl").textContent = "Begin reciting";
}

function render(u) {
  state.last = u;
  if (params.has("debug")) {
    $("heard").textContent = u.heard || "…";
    $("latency").textContent = `${Math.round(u.ms)} ms`;
  }
  if (!u.dua) {
    if (state.dua) return; // hold the last place through a brief lapse in confidence
    const waited = Date.now() - state.listeningSince;
    if (waited > 15000) $("now-tl").textContent = "Keep reciting, I'm finding your place";
    showGuesses(waited > 5000 ? u.candidates : []);
    return;
  }
  $("guesses").hidden = true;
  const dua = state.duas[u.dua];
  if (u.dua !== state.dua) {
    state.dua = u.dua;
    state.segment = null;
    $("dua-ar").textContent = dua.name_ar;
    $("dua-en").textContent = dua.name_en;
    buildFull(dua);
    setState("following");
  }
  if (u.segment !== state.segment) moveTo(dua, u.segment);
  paintWords(u.token);
  $("next").classList.toggle("coming", !!u.pause);
}

// -- choosing a du'a --------------------------------------------------------------
function openPicker() {
  $("search").value = "";
  fillPicker("");
  $("picker").hidden = false;
  $("search").focus();
}

function closePicker() {
  $("picker").hidden = true;
}

function fillPicker(query) {
  const q = query.trim().toLowerCase();
  const items = Object.values(state.duas)
    .filter((d) => !q || d.name_en.toLowerCase().includes(q) || d.name_ar.includes(query.trim()))
    .sort((a, b) => a.name_en.localeCompare(b.name_en))
    .map((d) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      const en = document.createElement("span");
      en.textContent = d.name_en;
      const ar = document.createElement("span");
      ar.className = "ar";
      ar.lang = "ar";
      ar.textContent = d.name_ar;
      b.append(en, ar);
      b.onclick = () => {
        closePicker();
        state.chosen = d.id;
        begin(micSource);
      };
      li.append(b);
      return li;
    });
  $("picker-list").replaceChildren(...items);
}

function showGuesses(candidates) {
  const likely = candidates.filter((c) => c.p >= 0.08).slice(0, 3);
  $("guesses").hidden = !likely.length;
  $("guess-chips").replaceChildren(
    ...likely.map((c) => {
      const d = state.duas[c.id];
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = d.name_en;
      const ar = document.createElement("span");
      ar.className = "ar";
      ar.textContent = d.name_ar;
      b.append(ar);
      b.onclick = () => {
        state.engine.lock(c.id);
        $("guesses").hidden = true;
      };
      return b;
    }),
  );
}

function lineText(dua, id) {
  return dua.segments.find((s) => s.id === id)?.ar ?? "";
}

function moveTo(dua, segment) {
  const stage = $("stage");
  const first = state.segment === null;
  state.segment = segment;
  state.token = null;
  const idx = dua.segments.findIndex((s) => s.id === segment);
  $("progress").style.width = `${((idx + 1) / dua.segments.length) * 100}%`;
  const swap = () => {
    const s = dua.segments[idx];
    $("prev").textContent = lineText(dua, segment - 1);
    $("next").textContent = lineText(dua, segment + 1);
    $("now-ar").replaceChildren(
      ...s.ar.split(/\s+/).filter(Boolean).flatMap((w, i) => {
        const span = document.createElement("span");
        span.textContent = w;
        span.dataset.i = i;
        return [span, " "];
      }),
    );
    $("now-tl").textContent = s.tl || "";
    $("now-en").textContent = s.en || "";
    stage.classList.remove("moving");
  };
  if (first) swap();
  else {
    stage.classList.add("moving");
    setTimeout(swap, 180);
  }
  for (const p of $("full").children) {
    const id = Number(p.dataset.id);
    p.classList.toggle("now", id === segment);
    p.classList.toggle("past", id < segment);
  }
  scrollFull();
}

function paintWords(token) {
  if (token === state.token) return;
  state.token = token;
  for (const span of $("now-ar").querySelectorAll("span")) {
    const i = Number(span.dataset.i);
    span.classList.toggle("said", i < token);
    span.classList.toggle("w", i === token);
  }
}

function buildFull(dua) {
  $("full").replaceChildren(
    ...dua.segments.map((s) => {
      const p = document.createElement("p");
      p.dataset.id = s.id;
      p.lang = "ar";
      p.textContent = s.ar;
      return p;
    }),
  );
}

function scrollFull() {
  if ($("full").hidden) return;
  $("full").querySelector("p.now")?.scrollIntoView({ block: "center", behavior: "smooth" });
}

init();
