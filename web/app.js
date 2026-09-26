// Du'a Companion: one front end, two engines.
//   server engine: audio streams to app/server.py over a WebSocket
//   device engine: Whisper runs in a Web Worker, tracker.js in the page;
//                  nothing leaves the device (used when there's no server)
import { CorpusIndex, DEFAULTS, RECITER, Tracker } from "./tracker.js";
import { Highlight } from "./display.js";
import { VoiceLevel, voiceBandDb } from "./voice.js";
import { SessionLog, clearSessions, listSessions, sessionFile, shareFiles } from "./session-log.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const SR = 16000;
// Debug sessions (session-log.js): on for now; ?log=0 turns them off.
const log = new SessionLog({ enabled: params.get("log") !== "0" });
const cands = (list) => list.map(([id, p]) => [id, Number(p.toFixed(3))]);

// -- engines ----------------------------------------------------------------
class ServerEngine {
  kind = "server";
  async prepare() {}
  async start(onUpdate) {
    // ?lead=0 turns off showing the predicted current position (for comparing by feel).
    // ?words=ctc: the server's CTC word follower places the word (docs/results/follower.md).
    const q = ["lead", "pauses"].filter((k) => params.get(k) === "0").map((k) => `${k}=0`)
      .concat(`follow=${following()}`, params.get("words") === "ctc" ? ["words=ctc"] : []).join("&");
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${q ? "?" + q : ""}`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "word") {
        log.event("wordstep", { end: m.t, ms: m.step_ms, dua: m.dua, seg: m.segment, token: m.token });
        return onUpdate({ word: true, dua: m.dua, segment: m.segment, token: m.token, ms: m.step_ms });
      }
      log.event("hop", { end: m.t, asr_ms: m.step_ms, text: m.heard, quiet: m.quiet, dua: m.dua, seg: m.segment,
        token: m.token, eol: m.pause_at_line_end, dua_p: m.dua_confidence, seg_p: m.segment_confidence,
        unknown: m.unknown, speed: m.speed, cand: cands((m.candidates || []).map((c) => [c.id, c.p])) });
      onUpdate({ dua: m.dua, segment: m.segment, token: m.token, speed: m.speed, unknown: m.unknown, pause: m.pause_at_line_end, heard: m.heard, ms: m.step_ms,
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
  follow(mode) {
    if (this.ws?.readyState === 1) this.ws.send(`follow:${mode}`);
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
    const t0 = performance.now();
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = ({ data }) => {
        if (data.type === "progress") onProgress?.(data.progress);
        else if (data.type === "ready") {
          this.loadMs = Math.round(performance.now() - t0);
          resolve();
        } else if (data.type === "error" && !this.onUpdate) reject(new Error(data.message));
        else if (data.type === "error") log.event("error", { where: "asr", message: data.message });
        else if (data.type === "text") this._onText(data);
      };
    });
    this.worker.postMessage({ type: "load", model: this.model, webgpu: params.has("webgpu") });
    return this.ready;
  }
  lock(duaId) {
    // Follow only this du'a: identification is skipped entirely.
    this.tracker = new Tracker(new CorpusIndex(this.corpus.filter((d) => d.id === duaId)), followConfig());
  }
  async start(onUpdate) {
    this.onUpdate = onUpdate;
    this.tracker = new Tracker(new CorpusIndex(this.corpus), followConfig());
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
    // Audio captured while Whisper ran is how far the reciter has moved on: show where they are now.
    const delay = (this.total - data.id) / SR;
    const lead = params.get("lead") === "0" ? 0 : delay + this.tracker.cfg.displayLead;
    const quiet = params.get("pauses") === "0" ? 0 : data.quiet ?? 0; // seconds the reciter has been silent (asr-worker.js)
    const p = this.tracker.update(data.text, dt, lead, quiet);
    const still = quiet > this.tracker.cfg.stillAfter; // they've stopped: so does the gliding highlight
    if (log.live) {
      // `now_*`: where the evidence alone puts them (no lead, no holding through
      // pauses), to tell recognition errors from display ones.
      const now = this.tracker.position();
      log.event("hop", { end: data.id / SR, asr_ms: data.ms, text: data.text, quiet, dt, lead,
        dua: p.dua, seg: p.segment, token: p.token, eol: p.atLineEnd, dua_p: p.duaConfidence, seg_p: p.segmentConfidence,
        now_seg: now.segment, now_token: now.token, unknown: this.tracker.null, speed: this.tracker.speed,
        cand: cands(p.candidates) });
    }
    this.onUpdate({ dua: p.dua, segment: p.segment, token: p.token, speed: still ? 0 : this.tracker.speed, unknown: this.tracker.null, pause: p.atLineEnd && (!data.text || still), heard: data.text, ms: data.ms,
      candidates: p.candidates.map(([id, prob]) => ({ id, p: prob })) });
    this._maybeSend();
  }
  follow() {
    const { leadCrossQuiet, retreatAfter } = DEFAULTS;
    Object.assign(this.tracker.cfg, { leadCrossQuiet, retreatAfter }, followConfig());
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
  state.mode = mode;
  state.engine = mode?.mode === "server"
    ? new ServerEngine()
    : new DeviceEngine(corpus, params.get("model") || "whisper-base-aug-v4");
  log.upload = log.enabled && !!mode?.sessions;
  $("footnote").textContent = state.engine.kind === "device"
    ? (log.upload ? "Runs on this device. Debug sessions go to this server." : "Runs entirely on this device. No audio leaves it.")
    : "";
  if (log.upload) log.sendPending().then(showSessions); // anything a closed tab left behind
  showSessions();
  $("sessions-send").onclick = sendSessions;
  $("sessions-clear").onclick = async () => {
    if (!confirm("Delete the debug sessions saved on this device?")) return;
    await clearSessions().catch(() => {});
    showSessions();
  };
  watchForDebugging();

  $("start").onclick = () => begin(micSource);
  $("play").onclick = showRecordings;
  $("choose").onclick = openPicker;
  $("picker").onclick = (e) => e.target === $("picker") && closePicker();
  $("search").oninput = () => fillPicker($("search").value);
  addEventListener("keydown", (e) => e.key === "Escape" && closePicker());
  $("file").onchange = (e) => {
    const f = e.target.files[0];
    if (f) begin(fileSource(URL.createObjectURL(f), f.name));
  };
  $("stop").onclick = () => end("stop");
  $("menu-btn").onclick = () => ($("menu").hidden = !$("menu").hidden);
  $("opt-en").onchange = (e) => {
    document.body.classList.toggle("no-en", !e.target.checked);
    log.event("option", { translation: e.target.checked });
  };
  $("opt-tl").onchange = (e) => {
    document.body.classList.toggle("no-tl", !e.target.checked);
    log.event("option", { transliteration: e.target.checked });
  };
  $("mode-page").onclick = () => setMajlis(false);
  $("mode-majlis").onclick = () => setMajlis(true);
  setMajlis(params.has("watch") || params.has("majlis") || stored("majlis") === "1", false);
  // Majlis hides the top bar; bring it back briefly when someone moves or taps.
  let chromeTimer;
  const showChrome = () => {
    document.body.classList.add("chrome");
    clearTimeout(chromeTimer);
    chromeTimer = setTimeout(() => $("menu").hidden && document.body.classList.remove("chrome"), 3000);
  };
  addEventListener("pointermove", showChrome);
  addEventListener("pointerdown", showChrome);
  if (params.has("debug")) $("debug").hidden = false;
  // Rooms are relayed by app/server.py, so sharing needs it (not a static host).
  $("share").hidden = !mode;
  $("share").onclick = share;
  $("share-card").onclick = (e) => e.target === $("share-card") && ($("share-card").hidden = true);
  if (params.get("watch")) watch(params.get("watch"));
}

// -- debug sessions (session-log.js) --------------------------------------------------
// What the log can't see from the recognizer: the user and the page around it.
function watchForDebugging() {
  // Tapping a line marks it: "I'm here" when the display is wrong is the best
  // ground truth a live session can have.
  $("text").addEventListener("click", (e) => {
    const ln = e.target.closest(".ln");
    if (!ln || !state.dua) return;
    const i = Number(ln.dataset.i);
    log.event("tap", { seg: state.duas[state.dua].segments[i].id, shown: state.segment });
    ln.classList.remove("tapped");
    void ln.offsetWidth; // restart the flash
    ln.classList.add("tapped");
  });
  // Scrolling by hand usually means hunting for the place.
  let from = null;
  addEventListener("touchstart", () => (from = scrollY), { passive: true });
  addEventListener("touchend", () => {
    if (from !== null && Math.abs(scrollY - from) > 40) log.event("scroll", { dy: Math.round(scrollY - from) });
    from = null;
  }, { passive: true });
  addEventListener("error", (e) => log.event("error", { where: "page", message: String(e.message) }));
  addEventListener("unhandledrejection", (e) => log.event("error", { where: "page", message: String(e.reason) }));
}

async function showSessions() {
  const all = await listSessions().catch(() => []);
  const unsent = all.filter((r) => !r.sent && r.id !== log.live?.rec.id);
  $("sessions").hidden = !log.enabled || !all.length;
  const min = Math.round(all.reduce((n, r) => n + r.seconds, 0) / 60);
  $("sessions-info").textContent = `Debug log: ${all.length} session${all.length === 1 ? "" : "s"} (${min} min) on this device`
    + (unsent.length ? `, ${unsent.length} not sent` : "");
  $("sessions-send").textContent = "send";
  $("sessions-send").hidden = log.upload && !unsent.length;
  if (log.upload) return;
  // By hand: the unsent ones (or the latest), built now so a tap on "send" can
  // open the share sheet straight away.
  const ids = (unsent.length ? unsent.slice(-5) : all.slice(-1)).map((r) => r.id);
  state.sendFiles = Promise.all(ids.map(sessionFile)).catch(() => []);
}

async function sendSessions() {
  $("sessions-send").textContent = "sending…";
  try {
    if (log.upload) await log.sendPending();
    else await shareFiles(await state.sendFiles);
  } catch (e) {
    $("sessions-send").textContent = e.name === "AbortError" ? "send" : "couldn't send";
    return;
  }
  showSessions();
}

function stored(key, value) {
  // Only a convenience: storage can be missing (private windows) or throw.
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {}
  return null;
}

// Page mode: someone reading along, who stops now and then; the display waits
// for them. Majlis mode: a reciter who flows through their breaths (tracker.js RECITER).
function following() {
  return document.body.classList.contains("majlis") ? "reciter" : "reading";
}

function followConfig() {
  return following() === "reciter" ? RECITER : {};
}

function setMajlis(on, remember = true) {
  document.body.classList.toggle("majlis", on);
  state.engine?.follow?.(following());
  log.event("follow", { mode: following() });
  $("mode-page").setAttribute("aria-checked", String(!on));
  $("mode-majlis").setAttribute("aria-checked", String(on));
  if (remember) stored("majlis", on ? "1" : "0");
  // A projector wants the whole screen; ask, but carry on if refused.
  if (on && remember && !document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
  if (!on && remember && document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
  scrollToNow();
}

// When and where each du'a is customarily recited, shown under its title.
const NOTES = {
  "dua-kumayl": "Thursday nights · taught by Imam Ali (a) to Kumayl ibn Ziyad",
  "dua-iftitah": "Every night of Ramadan",
  "dua-abu-hamza-thumali": "Pre-dawn in Ramadan · from Imam Zayn al-Abidin (a)",
  "dua-baha": "Pre-dawn in Ramadan",
  "dua-arafat": "The Day of Arafah · from Imam Husayn (a)",
  "dua-nudbah": "Friday mornings and the two Eids",
  "dua-aahad": "Each morning, for Imam al-Mahdi (a)",
  "dua-jawshan-kabir": "The nights of Qadr",
  "dua-mujeer": "The 13th, 14th and 15th nights of Ramadan",
  "dua-makaramakhlaq": "Al-Sahifa al-Sajjadiyya, 20",
  "dua-munajat-taibeen": "The Fifteen Whispered Prayers, 1 · Imam Zayn al-Abidin (a)",
  "dua-simaat": "The last hour of Friday",
  "dua-tasbih-suhoor": "Suhoor in Ramadan",
  "ziyarat-ashura": "The Day of Ashura, and any day",
};
const WEEKDAYS = { saturday: "Saturday", sunday: "Sunday", monday: "Monday", tuesday: "Tuesday",
  wednesday: "Wednesday", thursday: "Thursday", friday: "Friday" };

function noteFor(id) {
  if (NOTES[id]) return NOTES[id];
  const day = id.match(/^(?:dua|ziyarat)-(\w+day)$/)?.[1];
  if (WEEKDAYS[day]) return `Recited on ${WEEKDAYS[day]}`;
  const r = id.match(/^dua-ramadan-(\d+)(-night)?$/);
  if (r) return `${r[2] ? "Night" : "Day"} ${r[1]} of Ramadan`;
  return "";
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
  $("listen-msg").textContent = `Following room ${code.toUpperCase()}`;
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

function fileSource(url, name) {
  const source = async (ctx) => {
    const player = $("player");
    player.src = url;
    if (!state.mediaSource) {
      // A media element can join an audio graph only once: build it once.
      state.mediaSource = ctx.createMediaElementSource(player);
      state.mediaSource.connect(ctx.destination);
    }
    player.onended = () => end("file ended");
    player.play();
    return state.mediaSource;
  };
  source.label = `file:${name}`;
  return source;
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
    b.onclick = () => begin(fileSource(`audio/${r.id}`, r.id));
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
  if (!state.ctx) {
    state.ctx = new AudioContext();
    // A suspended context (phone locked, a call) stops the audio, and the log's clock with it.
    state.ctx.onstatechange = () => log.event("audio", { state: state.ctx.state });
  }
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
  const track = state.stream?.getAudioTracks()[0];
  const mic = track?.getSettings() ?? {};
  log.start({
    engine: state.engine.kind, model: state.engine.model ?? state.mode?.model, source: makeSource.label ?? "mic",
    chosen: state.chosen, follow: following(), params: location.search, ua: navigator.userAgent,
    screen: [innerWidth, innerHeight, devicePixelRatio], audio_rate: state.ctx.sampleRate, load_ms: state.engine.loadMs,
    mic: track && { label: track.label, rate: mic.sampleRate, echo: mic.echoCancellation, noise: mic.noiseSuppression,
      agc: mic.autoGainControl },
    tracker: state.engine.tracker?.cfg,
  });
  const node = new AudioWorkletNode(state.ctx, "capture");
  node.port.onmessage = (e) => {
    log.audio(e.data);
    state.engine.push(e.data);
  };
  source.connect(node);
  // Feeds the listening star (meter below); a dead end, like the capture node.
  state.analyser ??= new AnalyserNode(state.ctx, { fftSize: 2048, smoothingTimeConstant: 0 });
  source.connect(state.analyser);
  Object.assign(state, { node, source, dua: null, segment: null, token: null, listeningSince: Date.now() });
  showListening();
  meter();
}

// While the du'a is being found, the star answers the reciter's voice: it glows
// and swells as they recite, turns a little faster, and breathes when they pause.
// voice.js turns the microphone into a calm 0..1 level; here a soft spring sits
// between that level and the size, and only transform and opacity change, which
// the compositor animates without repainting.
function meter() {
  const girih = document.querySelector(".girih");
  const glow = document.querySelector(".voice-glow");
  const { analyser } = state;
  const spectrum = new Float32Array(analyser.frequencyBinCount);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const voice = new VoiceLevel();
  const debug = params.has("debug") && $("latency");
  let size = 0, sizeV = 0; // spring toward the level
  let spin = 3, angle = 0; // deg/s and deg
  let last = performance.now();
  const frame = (now) => {
    if (document.body.dataset.state !== "listening" || !state.source) {
      girih.style.transform = glow.style.opacity = "";
      return;
    }
    const dt = Math.min(0.1, (now - last) / 1000); // a background tab can stall for seconds
    last = now;
    analyser.getFloatFrequencyData(spectrum);
    const db = voiceBandDb(spectrum, state.ctx.sampleRate, analyser.fftSize);
    const level = voice.update(db, dt);
    if (debug) debug.textContent = `mic ${db.toFixed(0)} dB · room ${voice.floor.toFixed(0)} · voice ${voice.peak.toFixed(0)} · level ${level.toFixed(2)}`;

    // Slightly underdamped spring (k = 120, damping ratio 0.8): settles with a hint of give.
    sizeV += (120 * (level - size) - 2 * 0.8 * Math.sqrt(120) * sizeV) * dt;
    size += sizeV * dt;
    spin += (3 + 24 * level - spin) * (1 - Math.exp(-dt / 0.5)); // drifts at 3 deg/s, up to 27 while reciting
    angle = (angle + spin * dt) % 360;

    glow.style.opacity = Math.min(1, level * 1.15).toFixed(3);
    if (!still) {
      const breath = 0.015 * Math.sin((now / 1000) * (2 * Math.PI / 4.5)) * (1 - level);
      girih.style.transform = `rotate(${angle.toFixed(2)}deg) scale(${(1 + 0.12 * size + breath).toFixed(4)})`;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function end(reason) {
  const { node, source, stream } = state;
  if (source && node) source.disconnect(node);
  if (source && state.analyser) source.disconnect(state.analyser);
  if (source === state.mediaSource) $("player").pause();
  stream?.getTracks().forEach((t) => t.stop());
  state.engine?.stop();
  if (state.room?.readyState === 1) state.room.send(JSON.stringify({ ended: true }));
  log.end(reason).then(showSessions);
  Object.assign(state, { node: null, source: null, stream: null, dua: null, segment: null, chosen: null, hl: null,
    preview: false, guessesShown: "" });
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
  $("text").replaceChildren();
  $("folio-head").hidden = true;
  $("listen-msg").textContent = "Begin reciting";
}

// ?words=ctc: word messages place the line and word directly (no glide) while they
// keep coming; the tracker's updates still identify the du'a and drive the rest.
function renderWord(u) {
  state.wordAt = performance.now();
  state.wordLive = !!u.dua && u.dua === state.dua;
  if (!state.wordLive) return; // not locked yet, or the tracker changed du'a: its updates lead
  state.hl = null; // stops the glide loop
  if (u.segment !== state.segment) moveTo(state.duas[u.dua], u.segment);
  paintWords(u.token);
}

function render(u) {
  if (u.word) return renderWord(u);
  state.last = u;
  if (params.has("debug")) {
    $("heard").textContent = u.heard || "…";
    $("latency").textContent = `${Math.round(u.ms)} ms`;
  }
  if (!u.dua) {
    if (state.dua) return; // hold the last place through a brief lapse in confidence
    const waited = Date.now() - state.listeningSince;
    if (waited > 10000 && u.unknown > 0.9) listenMsg("I don't know this du'a yet");
    else if (waited > 15000) listenMsg("Keep reciting, I'm finding your place");
    showGuesses(waited > 5000 ? u.candidates : []);
    return;
  }
  $("guesses").hidden = true;
  const dua = state.duas[u.dua];
  if (u.dua !== state.dua) {
    log.event("dua", { dua: u.dua });
    state.dua = u.dua;
    state.segment = null;
    $("dua-ar").textContent = dua.name_ar;
    $("dua-en").textContent = dua.name_en;
    buildText(dua);
    setState("following");
  }
  const words = state.wordLive && performance.now() - state.wordAt < 1000;
  if (!words) {
    if (u.segment !== state.segment) moveTo(dua, u.segment);
    glide(u);
  }
  state.lines.get((words ? state.segment : u.segment) + 1)?.classList.toggle("coming", !!u.pause);
  if (!!u.pause !== !!state.preview) log.event("preview", { on: !!u.pause });
  state.preview = !!u.pause;
}

function listenMsg(text) {
  if ($("listen-msg").textContent === text) return;
  $("listen-msg").textContent = text;
  log.event("msg", { text });
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
      const note = noteFor(d.id);
      if (note) en.append(Object.assign(document.createElement("span"), { className: "note", textContent: note }));
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
  const top = likely[0]?.p || 1;
  const shown = likely.map((c) => c.id).join(" ");
  if (shown !== (state.guessesShown ?? "")) log.event("guesses", { ids: likely.map((c) => c.id) });
  state.guessesShown = shown;
  $("guesses").hidden = !likely.length;
  $("guess-chips").replaceChildren(
    ...likely.map((c) => {
      const d = state.duas[c.id];
      const b = document.createElement("button");
      b.className = "chip";
      b.style.opacity = (0.45 + 0.55 * (c.p / top)).toFixed(2); // the likeliest stands out as it firms up
      b.textContent = d.name_en;
      const ar = document.createElement("span");
      ar.className = "ar";
      ar.textContent = d.name_ar;
      b.append(ar);
      b.onclick = () => {
        log.event("lock", { dua: c.id, from: "guess" });
        state.engine.lock(c.id);
        $("guesses").hidden = true;
      };
      return b;
    }),
  );
}

const arabicNumber = new Intl.NumberFormat("ar-EG");

// A verse-end rosette carrying the line number, as in a mushaf.
function marker(n) {
  const mark = document.createElement("span");
  mark.className = "mark";
  mark.setAttribute("aria-hidden", "true");
  mark.innerHTML = '<svg viewBox="-50 -50 100 100"><use href="#rosette" x="-50" y="-50" width="100" height="100"/></svg>';
  mark.append(Object.assign(document.createElement("b"), { textContent: arabicNumber.format(n) }));
  return mark;
}

// The whole du'a on one page; the recited line grows and brightens as it's reached.
function buildText(dua) {
  $("head-ar").textContent = dua.name_ar;
  $("head-en").textContent = dua.name_en;
  $("head-note").textContent = noteFor(dua.id);
  $("folio-head").hidden = false;
  state.lines = new Map();
  $("text").replaceChildren(
    ...dua.segments.map((s, i) => {
      const ln = document.createElement("div");
      ln.className = "ln";
      ln.dataset.i = i;
      const ar = Object.assign(document.createElement("p"), { className: "ar", lang: "ar" });
      ar.append(s.ar, " ", marker(i + 1));
      const inner = document.createElement("div");
      inner.append(
        Object.assign(document.createElement("p"), { className: "tl", textContent: s.tl || "" }),
        Object.assign(document.createElement("p"), { className: "en", textContent: s.en || "" }),
      );
      const gloss = Object.assign(document.createElement("div"), { className: "gloss" });
      gloss.append(inner);
      ln.append(ar, gloss);
      state.lines.set(s.id, ln);
      return ln;
    }),
  );
}

// Only the line being recited is split into words; the rest stay plain text.
function setWords(ln, split) {
  const ar = ln.querySelector(".ar");
  const text = state.duas[state.dua].segments[ln.dataset.i].ar;
  const words = split
    ? text.split(/\s+/).filter(Boolean).flatMap((w, i) => {
      const span = Object.assign(document.createElement("span"), { className: "wd", textContent: w });
      span.dataset.i = i;
      return [span, " "];
    })
    : [text, " "];
  ar.replaceChildren(...words, ar.querySelector(".mark"));
}

function moveTo(dua, segment) {
  const prev = state.lines.get(state.segment);
  if (prev) setWords(prev, false);
  state.segment = segment;
  state.token = null;
  const idx = dua.segments.findIndex((s) => s.id === segment);
  log.event("line", { seg: segment, idx });
  $("progress").style.width = `${((idx + 1) / dua.segments.length) * 100}%`;
  for (const [id, ln] of state.lines) {
    ln.classList.toggle("now", id === segment);
    ln.classList.toggle("past", id < segment);
    ln.classList.toggle("next", id === segment + 1);
    ln.classList.remove("coming");
  }
  setWords(state.lines.get(segment), true);
  scrollToNow();
}

// Between updates the highlight glides at the reciter's pace (display.js)
// instead of hopping a word or two once a second.
function glide(u) {
  const n = state.lines.get(state.segment).querySelectorAll(".wd").length;
  if (u.token == null || !n || params.get("glide") === "0") return paintWords(u.token);
  state.hl ??= new Highlight();
  state.hl.update(performance.now() / 1000, `${u.dua}:${u.segment}`, u.token, 0, n, u.speed ?? 1);
  if (state.gliding) return;
  state.gliding = true;
  const frame = () => {
    if (!state.hl?.line || !state.lines?.get(state.segment)) return (state.gliding = false);
    paintWords(state.hl.word(performance.now() / 1000));
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function paintWords(token) {
  if (token === state.token) return;
  state.token = token;
  log.event("word", { token });
  for (const span of state.lines.get(state.segment).querySelectorAll(".wd")) {
    const i = Number(span.dataset.i);
    span.classList.toggle("said", i < token);
    span.classList.toggle("w", i === token);
  }
}

function scrollToNow() {
  // The line grows over half a second: centre on it now, then again once it has settled.
  const ln = state.lines?.get(state.segment);
  if (!ln) return;
  const behavior = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  const centre = () => ln.scrollIntoView({ block: "center", inline: "nearest", behavior });
  requestAnimationFrame(centre);
  clearTimeout(state.recentre);
  state.recentre = setTimeout(centre, 550);
}

init();
