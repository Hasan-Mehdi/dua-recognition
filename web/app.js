// Du'a Companion: one front end, one engine (DeviceEngine), with its models in one of two places.
//   device: Whisper and the CTC model run in Web Workers; nothing leaves the device
//   server: they run on app/server.py (RemoteEar), the audio streamed to it over a WebSocket
// Either way the tracker, the stream decoder and the highlight run here, the same code.
import { CorpusIndex, DEFAULTS, RECITER, Tracker } from "./tracker.js";
import { Highlight } from "./display.js";
import { LiveQuiet } from "./gate.js";
import { FOLLOW_DEFAULTS, LocalFollower } from "./follower.js";
import { STREAM_DEFAULTS, StreamFollower } from "./stream-follower.js";
import { VoiceLevel, voiceBandDb } from "./voice.js";
import { SessionLog, clearSessions, listSessions, sessionFile, shareFiles } from "./session-log.js";
import { kids } from "./kids.js";
import { FrameCommitter, PracticeChecker, linesOf } from "./practice.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const SR = 16000;
// Debug sessions (session-log.js): on for now; ?log=0 turns them off.
const log = new SessionLog({ enabled: params.get("log") !== "0" });
const cands = (list) => list.map(([id, p]) => [id, Number(p.toFixed(3))]);

// Epoch ms (performance.timeOrigin + now): comparable with the ASR worker's clock.
const clock = () => performance.timeOrigin + performance.now();

// When a chunk's last sample was captured, as epoch ms: its AudioContext time (the audio
// sample clock, from capture-worklet.js) mapped through getOutputTimestamp(). Microphone
// hardware latency before the AudioContext is not included. Falls back to arrival time.
function captureTime(ctxTime) {
  const ts = state.ctx?.getOutputTimestamp?.();
  if (ctxTime == null || !ts?.performanceTime) return clock();
  return performance.timeOrigin + ts.performanceTime + (ctxTime - ts.contextTime) * 1000;
}

// -- engines ----------------------------------------------------------------
// The server engine (app/server.py /ws/ear): Whisper and the CTC model run on the server, on the
// audio the page streams to it, and everything that follows the recitation runs in the page, the
// same DeviceEngine code as on the phone. Its two ports stand in for asr-worker.js and ctc-worker.js
// and take their messages, except that a window is named by the sample it ends at instead of sent:
// the server keeps the audio. Replies from an earlier session (`gen`) are dropped.
class RemoteEar {
  asr = { postMessage: (m) => this._send("asr", m), onmessage: null };
  ctc = { postMessage: (m) => this._send("ctc", m), onmessage: null };
  total = 0; // samples streamed this session
  gen = 0;
  pending = {}; // channel -> the request awaiting its answer
  connect() {
    if (this.open) return this.open;
    const ws = new WebSocket(`${wsBase()}/ws/ear`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => this._receive(e.data);
    // Audio sent while closed is lost: the server is told where the stream is now, and hears silence before it.
    this.open = new Promise((ok, err) => {
      ws.onopen = () => {
        ws.send(JSON.stringify({ ch: "ear", type: "at", total: this.total }));
        ok(ws);
      };
      ws.onerror = () => err(new Error("no connection to the server"));
    });
    ws.onclose = () => {
      if (this.ws !== ws) return;
      Object.assign(this, { ws: null, open: null });
      for (const [ch, m] of Object.entries(this.pending)) this._fail(ch, m, "the connection to the server closed");
    };
    this.ws = ws;
    return this.open;
  }
  // A new session: sample numbers start again from 0.
  restart() {
    this.total = 0;
    this.gen += 1;
    this.pending = {};
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ ch: "ear", type: "at", total: 0 }));
  }
  push(chunk) {
    this.total += chunk.length;
    if (this.ws?.readyState !== 1) return;
    const pcm = new Int16Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(chunk[i] * 32768)));
    this.ws.send(pcm.buffer);
  }
  async _send(ch, { audio, ...m }) {
    const req = { ch, ...m, sent: clock() };
    if (m.type !== "load") { // (a load outlives sessions: it carries none)
      req.gen = this.gen;
      this.pending[ch] = req;
    }
    try {
      (await this.connect()).send(JSON.stringify(req));
    } catch (e) {
      this._fail(ch, req, e.message);
    }
  }
  // As the workers answer when inference fails: the page carries on (no transcript, no frames).
  _fail(ch, req, message) {
    if (this.pending[ch] === req) delete this.pending[ch];
    if (req.gen != null && req.gen !== this.gen) return;
    this[ch].onmessage?.({ data: { type: "error", message } });
    if (req.type === "transcribe") {
      this.asr.onmessage?.({ data: { type: "text", id: req.id, text: "", quiet: null, paused: null, voiceDb: req.voiceDb,
        ms: clock() - req.sent, gate: null, skip: "error" } });
    }
  }
  _receive(data) {
    if (typeof data !== "string") { // the frames whose header came just before
      const m = this.frames;
      this.frames = null;
      if (m?.gen === this.gen) this.ctc.onmessage?.({ data: { ...m, frames: halfToFloat(new Uint16Array(data)) } });
      return;
    }
    const m = JSON.parse(data);
    const req = this.pending[m.ch];
    const answers = m.gen === this.gen && req && m.id === req.id;
    if (answers) delete this.pending[m.ch];
    // As the workers time a step: all of it, here the round trip (the server's own part is server_ms).
    if (answers && m.type !== "error") {
      const now = clock();
      Object.assign(m, { ms: now - req.sent, recv: now - (m.server_ms ?? 0), done: now });
    }
    if (m.type === "frames") return void (this.frames = m); // its frames come next
    if (m.gen == null || m.gen === this.gen) this[m.ch]?.onmessage?.({ data: m });
  }
}

// float16 to float32 (the server's CTC frames), through a table of all 65536 values.
let HALF = null;
function halfToFloat(h) {
  if (!HALF) {
    HALF = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) {
      const e = (i >> 10) & 31;
      const f = i & 1023;
      const v = e === 0 ? f * 2 ** -24 : e === 31 ? (f ? NaN : Infinity) : (1024 + f) * 2 ** (e - 25);
      HALF[i] = i & 0x8000 ? -v : v;
    }
  }
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = HALF[h[i]];
  return out;
}

// Audio reaches the engine in chunks of this many 16 kHz samples (capture-worklet.js): 50 ms, so a
// word waits ~25 ms for its chunk instead of ~125 ms (250 ms chunks until 2026-10-01). ?chunk=4000: as before.
const CHUNK = Number(params.get("chunk") || 800);

// The streaming student since 2026-10-07 (stream_ctc.py, ctc-stream.js; docs/results/phone_engine.md):
// each 0.1 s step encodes only the new audio, a quarter of the 2 s window model's work (on a phone the
// compute is most of the follower's delay), so no phone needs the small fallback for it. The window
// students (?ctc=ctc-student-base-v6-w2) keep their slow-phone fallback.
const CTC_MODEL = "ctc-stream-base-v1";
const CTC_FALLBACK = "ctc-student-tiny-v6-w2";
const CTC_STORED = "ctc-model-2"; // (a fallback stored under "ctc-model" chose between window models)
const CTC_SLOW_MS = 400;

class DeviceEngine {
  kind = "device";
  // remote: a RemoteEar, the server's models in place of the workers (the server engine), with the
  // name of the server's CTC model and its profile: the stream decoder and tracker settings and the
  // CTC step the server's models were tuned for (/api/mode; ?sc=, ?tc= and ?ctchop= still win).
  constructor(corpus, model, remote = null, remoteCtc = null, profile = null) {
    this.profile = profile || {};
    this.corpus = corpus;
    this.tracker = new Tracker(new CorpusIndex(corpus));
    this.model = model;
    this.remote = remote;
    if (remote) this.kind = "server";
    this.window = 6 * SR;
    this.buf = new Float32Array(this.window);
    // Speech gate policy (web/gate.js; docs/results/browser_gate.md): legacy unless ?gate=...
    this.gate = params.get("gate") || "legacy";
    this.filter = params.get("filter") === "1"; // the hallucination filter, a separate ablation
    // The word follower (follower.js) on the phone CTC model (ctc-worker.js): timed letter evidence
    // several times a second places the word; Whisper and the tracker find the du'a and anchor it
    // (docs/results/phone_follower.md). ?words=off: Whisper's lead and glide alone, as before.
    this.words = params.get("words") !== "off";
    // A phone too slow for it (the model's step over 400 ms) gets the whisper-tiny one from its next
    // session on: fewer, later steps cost more than a weaker ear (docs/results/phone_follower.md).
    const kept = stored(CTC_STORED); // a fallback chosen in an earlier session (older names don't count)
    this.ctcModel = remoteCtc || params.get("ctc") || ([CTC_MODEL, CTC_FALLBACK].includes(kept) ? kept : null) || CTC_MODEL;
    // A step as soon as the last is done and 0.1 s of audio has come in: the step's wait is part of
    // the delay too (0.2 s until 2026-10-01).
    this.ctcHop = Number(params.get("ctchop") || this.profile.ctcHop || 0.1) * SR;
    // While the follower places the words, Whisper only anchors it: every 2 s leaves the phone's
    // CPU to the CTC model. A server has the time for every second, as on the bench.
    this.anchorHop = Number(params.get("anchorhop") || (remote ? 1 : 2)) * SR;
    this.followCfg = { lapseHold: Number(params.get("lapse") ?? 0) };
    // The stream decoder (stream-follower.js: one belief over the whole du'a, reading moves as its
    // transitions; docs/results/bench.md) since 2026-10-03; ?follower=rules: the rule-based follower.js.
    this.followerKind = params.get("follower") === "rules" ? "rules" : "stream";
    this.streamCfg = { ...this.profile.sc, ...urlCfg("sc") };
  }
  // The follower is placing the words (its last result under a second old).
  get following() {
    return this.words && this.follower?.word != null && this.total - (this.framesAt ?? -Infinity) < SR;
  }
  prepare(onProgress) {
    if (this.ready) return this.ready;
    this.worker = this.remote?.asr ?? new Worker("asr-worker.js", { type: "module" });
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
    // The CTC model loads once Whisper has: compiling both at once slowed Whisper's first windows.
    if (this.words) this.ready.then(() => this._prepareCtc(), () => {});
    return this.ready;
  }
  _prepareCtc() {
    this.ctc = this.remote?.ctc ?? new Worker("ctc-worker.js", { type: "module" });
    this.ctcReady = false;
    this.ctc.onmessage = ({ data }) => {
      if (data.type === "ready") {
        this.ctcReady = true;
        this.ctcWindow = Math.round(data.meta.window_s * SR);
        this.ctcStream = data.meta.arch === "stream"; // a streaming model: each step encodes only new audio
      } else if (data.type === "error") {
        log.event("error", { where: "ctc", message: data.message });
        this.ctcBusy = false;
      } else if (data.type === "frames") this._onFrames(data);
    };
    this.ctc.postMessage({ type: "load", model: this.ctcModel, threads: Number(params.get("ctcthreads") || 0) || undefined });
  }
  lock(duaId) {
    // Follow only this du'a: identification is skipped entirely.
    this.tracker = new Tracker(new CorpusIndex(this.corpus.filter((d) => d.id === duaId)), followConfig());
    Object.assign(this, { anchor: null, follower: null }); // word indices of the old index mean nothing now
  }
  seek(duaId, segment) {
    const p = this.tracker.seek(duaId, segment);
    this.follower?.reset();
    this.anchor = p?.word ?? null;
    return p;
  }
  async start(onUpdate) {
    this.onUpdate = onUpdate;
    this.remote?.restart();
    this.tracker = new Tracker(new CorpusIndex(this.corpus), followConfig());
    this.streamCfg = { ...this.profile.sc, ...urlCfg("sc") };
    this.practice = null;
    Object.assign(this, { filled: 0, total: 0, lastSent: 0, lastUpdate: 0, busy: false, live: true, marks: [],
      firstText: true, voiceDb: null });
    // The stop detector on the audio as it arrives (gate.js LiveQuiet): an update is shown a second
    // or so after its window ends, and the reciter may have stopped since (docs/results/stops.md).
    this.ear = new LiveQuiet();
    this.noPauses = params.get("pauses") === "0";
    Object.assign(this, { ctcBusy: false, ctcSent: 0, anchor: null, follower: null, previewWord: null, ctcTimes: [] });
  }
  // Capture time (epoch ms) of sample `id` of the stream, from the chunk marks.
  capturedAt(id) {
    const m = this.marks.find(([n]) => n >= id);
    return m ? m[1] - ((m[0] - id) / SR) * 1000 : null;
  }
  push(chunk, at = clock()) {
    this.marks.push([this.total + chunk.length, at]);
    if (this.marks.length > 256) this.marks.shift();
    const { buf, window: W } = this;
    if (chunk.length >= W) buf.set(chunk.subarray(chunk.length - W));
    else {
      buf.copyWithin(0, chunk.length);
      buf.set(chunk, W - chunk.length);
    }
    this.filled = Math.min(W, this.filled + chunk.length);
    this.total += chunk.length;
    this.ear.push(chunk);
    this.remote?.push(chunk); // before anything asks the server about it
    this._maybeSend();
    this._maybeCtc();
  }
  get _ctcIdle() {
    return this.anchor == null && this.follower?.word == null;
  }
  _maybeCtc() {
    // Nothing to follow until the tracker has found the du'a (through a lapse, the follower's own word).
    // A streaming model keeps listening meanwhile, in bigger, cheaper steps, so its 3 s of context are
    // there when the du'a is found (and the frames match the bench's, which stream from the start).
    const idle = this._ctcIdle;
    if (idle && !this.ctcStream) return;
    const hop = idle ? Math.max(this.ctcHop, 0.3 * SR) : this.ctcHop;
    if (!this.live || !this.ctcReady || this.ctcBusy || this.total - this.ctcSent < hop) return;
    this.ctcBusy = true;
    this.ctcSent = this.total;
    // The latest window, silence before the session's start (buf starts zeroed).
    const audio = this.buf.slice(this.window - this.ctcWindow);
    this.ctc.postMessage({ type: "frames", id: this.total, audio }, [audio.buffer]);
  }
  _onFrames(data) {
    this.ctcBusy = false;
    if (!this.live) return;
    if (this._ctcIdle) return this._maybeCtc(); // (a streaming model listening ahead: nothing to place yet)
    this.ctcTimes.push(data.ms);
    if (this.ctcTimes.length === 20 && !params.get("ctc") && !this.remote && !this.ctcStream) {
      // Too slow for the full model: the small one from the next session. Fast on the small one
      // (the full one takes ~2.5x as long): back to the full one.
      const p50 = [...this.ctcTimes].sort((a, b) => a - b)[10];
      const next = this.ctcModel === CTC_MODEL && p50 > CTC_SLOW_MS ? CTC_FALLBACK
        : this.ctcModel === CTC_FALLBACK && p50 < CTC_SLOW_MS * 0.3 ? CTC_MODEL : null;
      if (next) {
        stored(CTC_STORED, next);
        log.event("ctc_model_next", { p50, next });
      }
    }
    const ix = this.tracker.ix;
    const stream = this.followerKind === "stream";
    if (this.follower?.ix !== ix) {
      this.follower = stream ? new StreamFollower(ix, this.streamCfg) : new LocalFollower(ix, this.followCfg);
    }
    const lineMass = (word) => this.tracker.lineMass(word);
    // The stop detector hears now; the window ended (total - id) ago: its quiet then, if still quiet now.
    const quietAtEnd = this.ear.quiet == null ? null : Math.max(0, this.ear.quiet - (this.total - data.id) / SR);
    const f0 = performance.now();
    const w = stream
      ? this.follower.step(data.frames, data.T, data.C, data.id / SR, this.anchor, quietAtEnd, lineMass, this.anchorAt)
      : this.follower.step(data.frames, data.T, data.C, data.id / SR, this.anchor, this.ear.quiet, lineMass);
    const followMs = performance.now() - f0; // on the page's own thread: the phone's budget is ~100 ms a step
    if (w != null) this.framesAt = data.id;
    if (this.practice) this._practiceStep(data, w);
    if (log.live) log.event("ctc", { end: data.id / SR, ms: data.ms, delay: (this.total - data.id) / SR,
      word: w, anchor: this.anchor, follow_ms: Number(followMs.toFixed(1)) });
    if (w != null) {
      const wd = ix.words[w];
      // On a line's last word with the reciter silent: the next line is previewed, and stays
      // previewed until the word changes (a breath in the pause doesn't switch it off and on).
      const [, hi] = ix.duaWordSpan[wd.dua];
      const eol = w + 1 >= hi || ix.wordSegment[w + 1] !== ix.wordSegment[w];
      if (eol && (this.ear.quiet ?? 0) > this.tracker.cfg.stillAfter) this.previewWord = w;
      const pause = eol && this.previewWord === w;
      this.onUpdate({ word: true, dua: ix.duaIds[wd.dua], segment: wd.segment, token: wd.token, ms: data.ms, pause });
    }
    this._maybeCtc();
  }
  _maybeSend() {
    // Skip stale hops rather than queue them: only the newest window matters.
    if (!this.live || this.busy) return;
    // While following, a stop is when the anchor matters most (a last word the CTC model didn't hear:
    // the follower catches up to it, follower.js quietAfter): Whisper runs as soon as the reciter has
    // been quiet a moment, on a window that holds everything up to the stop, instead of up to 2 s later.
    const quiet = this.ear.quiet ?? 0;
    const stopped = this.following && quiet > this.tracker.cfg.stillAfter && quiet < 2
      && this.lastSent < this.total - quiet * SR;
    if (!stopped && this.total - this.lastSent < (this.following ? this.anchorHop : SR)) return;
    this.busy = true;
    this.lastSent = this.total;
    const audio = this.buf.slice(this.window - this.filled);
    this.sentAt = clock();
    // The voice level the stop detector compares with, remembered from window to window, and the
    // seconds since the last update, for how much of them went on pauses (gate.js stopMeasures).
    this.worker.postMessage({ type: "transcribe", id: this.total, audio, gate: this.gate, filter: this.filter,
      voiceDb: this.voiceDb, dt: (this.total - this.lastUpdate) / SR }, [audio.buffer]);
  }
  _onText(data) {
    this.busy = false;
    if (!this.live) return;
    const dt = (data.id - this.lastUpdate) / SR;
    this.lastUpdate = data.id;
    // Audio captured while Whisper ran is how far the reciter has moved on: show where they are now.
    const delay = (this.total - data.id) / SR;
    // With the word follower, the tracker's own display shows the evidence, not a prediction: it is
    // only on screen until the follower's first step (or through a lapse), and a guess ahead that
    // the follower then took back would be the very jump it is there to stop.
    const lead = params.get("lead") === "0" || (this.words && this.ctcReady) ? 0 : delay + this.tracker.cfg.displayLead;
    const cfg = this.tracker.cfg;
    const quiet = this.noPauses ? 0 : data.quiet ?? 0; // seconds the reciter has been silent (asr-worker.js)
    if (data.voiceDb != null) this.voiceDb = this.ear.voiceDb = data.voiceDb;
    // ...and still now, in the audio that arrived while Whisper ran; the pauses in the window's last dt.
    const quietNow = this.noPauses ? null : this.ear.quiet;
    const paused = this.noPauses ? null : data.paused ?? null;
    const p = this.tracker.update(data.text, dt, lead, quiet, quietNow, paused);
    if (this.words) {
      // The follower's anchor: where the evidence alone puts them (no lead), once the du'a is found.
      const now = this.tracker.position();
      this.anchor = now.dua != null ? now.word : null;
      this.anchorAt = data.id / SR; // a new tracker result: the stream decoder takes its belief once
    }
    // They've stopped: so does the gliding highlight; a while ago: it may go back to where they stopped.
    const still = quiet > cfg.stillAfter || quietNow > cfg.stillAfter;
    const back = Math.max(quiet, quietNow ?? 0) > cfg.retreatAfter;
    if (log.live) {
      // `now_*`: where the evidence alone puts them (no lead, no holding through
      // pauses), to tell recognition errors from display ones.
      const now = this.tracker.position();
      // Gate and timing (docs/results/browser_gate.md): why Whisper did or didn't run, and
      // when the window's audio was captured, sent, received, done, and (event "visible") shown.
      const captured = this.capturedAt(data.id);
      const g = data.gate;
      log.event("hop", { end: data.id / SR, asr_ms: data.ms, text: data.text, quiet, quiet_now: quietNow, paused,
        voice_db: this.voiceDb, dt, lead,
        dua: p.dua, seg: p.segment, token: p.token, eol: p.atLineEnd, dua_p: p.duaConfidence, seg_p: p.segmentConfidence,
        now_seg: now.segment, now_token: now.token, unknown: this.tracker.null, speed: this.tracker.speed,
        cand: cands(p.candidates), gate: g?.policy ?? this.gate, ran: g ? g.run : null, skip: data.skip ?? null,
        level: g?.level, vad_run: g?.vad?.run, vad_peak: g?.vad?.peak, energy_run: g?.energy?.run,
        energy_floor: g?.energy?.floor, via: g?.via, halluc: data.hallucinated, infer_ms: data.infer_ms,
        cold: this.firstText, captured_ms: captured, sent_ms: this.sentAt, recv_ms: data.recv, done_ms: data.done,
        queue_ms: data.recv != null && this.sentAt != null ? data.recv - this.sentAt : null });
      const id = data.id;
      requestAnimationFrame(() => {
        const shown = clock();
        log.event("visible", { end: id / SR, shown_ms: shown, age_ms: captured != null ? shown - captured : null });
      });
    }
    this.firstText = false;
    this.onUpdate({ dua: p.dua, segment: p.segment, token: p.token, speed: still ? 0 : this.tracker.speed, back, unknown: this.tracker.null, pause: p.atLineEnd && (!data.text || still), heard: data.text, ms: data.ms,
      candidates: p.candidates.map(([id, prob]) => ({ id, p: prob })), sameAs: p.sameAs || [] });
    this._maybeSend();
  }
  follow() {
    const { leadCrossQuiet, retreatAfter } = DEFAULTS;
    Object.assign(this.tracker.cfg, { leadCrossQuiet, retreatAfter }, followConfig());
  }
  stop() {
    this.live = false;
  }
  // Practice mode (practice.js, docs/results/practice_bench.md): the du'a the reader chose (locked),
  // read on from where they start. No jumps across the du'a (cFar), so a repeated block can't pull
  // the highlight to its other copy; the checker judges each line from the frames the follower
  // commits, once the highlight has stayed two lines past it.
  startPractice(duaId, onDecide) {
    this.streamCfg = { ...this.streamCfg, cFar: -40 };
    this.follower = null;
    this.practice = { dua: duaId, onDecide, committer: new FrameCommitter(), checker: null, lo: 0, decided: 0, t: 0 };
  }
  _practiceLines() {
    const pr = this.practice;
    const ix = this.tracker.ix;
    const d = ix.duaIds.indexOf(pr.dua);
    if (d < 0) return null;
    if (pr.ix !== ix) Object.assign(pr, { ix, d, lines: linesOf(ix, d), checker: null });
    return pr.lines;
  }
  _practiceStep(data, w) {
    const pr = this.practice;
    const lines = this._practiceLines();
    if (!lines) return;
    pr.checker ??= new PracticeChecker(lines.letters, pr.lo, lines.letters.length - 1, {}, lines.firstWords);
    const t = data.id / SR;
    const { rows, times } = pr.committer.add(data.frames, data.T, data.C, t);
    pr.checker.addFrames(rows, data.C, times);
    const li = w == null || pr.ix.wordDua[w] !== pr.d ? null : lines.lineOf.get(pr.ix.wordSegment[w]) ?? null;
    pr.t = t;
    const t0 = performance.now();
    const out = pr.checker.step(t, li);
    if (out.length) {
      const ms = (performance.now() - t0) / out.length; // on the page's own thread, like follow_ms
      pr.decided += out.length;
      pr.onDecide(out.map((x) => ({ ...x, segment: lines.segs[x.line], ms })));
    }
  }
  // The reader starts at this line (the first, or one they tap before any line is judged); once lines
  // are judged, a tap only moves the highlight.
  practiceFrom(segment) {
    const pr = this.practice;
    if (!pr || pr.decided) return;
    const lines = this._practiceLines();
    pr.lo = lines?.lineOf.get(segment) ?? 0;
    pr.checker = null;
  }
  // The session is over: the lines not judged yet, through the furthest one the highlight reached.
  finishPractice() {
    const pr = this.practice;
    if (!pr?.checker) return [];
    return pr.checker.finish(pr.t, pr.checker.furthest).map((x) => ({ ...x, segment: pr.lines.segs[x.line] }));
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
  kids.init(state.duas, {
    openPicker,
    pick: (id) => {
      closePicker();
      state.chosen = id;
      begin(micSource);
    },
    again: (id) => {
      end("again");
      state.chosen = id;
      begin(micSource);
    },
    home: () => end("done"),
  });
  const mode = await fetch("api/mode").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  state.mode = mode;
  state.engine = mode?.mode !== "server"
    // Trained with synthetic ordinary voices, at an 8 s context (docs/results/synthetic_voices.md, phone_speed.md).
    ? new DeviceEngine(corpus, params.get("model") || "whisper-base-ph-kd2")
    : new DeviceEngine(corpus, mode.model, new RemoteEar(), mode.ctc, mode.profile);
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
  $("choose").onclick = () => openPicker();
  // Practice mode checks each line against the CTC frames (the phone's own, or the server's). Chosen
  // beneath the seal, as a camera chooses photo or video: the seal, the picker and the chips then
  // practise (setPractising). Each visit starts in follow: a Thursday night's Kumayl isn't a test.
  $("modes").hidden = !state.engine.startPractice;
  $("mode-follow").onclick = () => setPractising(false);
  $("mode-practise").onclick = () => setPractising(true);
  setPractising(params.get("mode") === "practise" && !!state.engine.startPractice, false);
  setHideText(stored("practice-hide") === "1", false);
  for (const id of ["opt-hide", "home-hide"]) $(id).onchange = (e) => setHideText(e.target.checked);
  $("hint-btn").onclick = hintWord;
  $("review-again").onclick = practiseAgain;
  $("review-done").onclick = closeReview;
  $("picker").onclick = (e) => e.target === $("picker") && closePicker();
  $("search").oninput = () => fillPicker($("search").value);
  addEventListener("keydown", (e) => e.key === "Escape" && closePicker());
  $("file").onchange = (e) => {
    const f = e.target.files[0];
    if (f) begin(fileSource(URL.createObjectURL(f), f.name));
  };
  $("stop").onclick = () => (state.review ? closeReview() : end("stop"));
  $("menu-btn").onclick = () => ($("menu").hidden = !$("menu").hidden);
  $("opt-en").onchange = (e) => {
    document.body.classList.toggle("no-en", !e.target.checked);
    log.event("option", { translation: e.target.checked });
  };
  $("opt-tl").onchange = (e) => {
    document.body.classList.toggle("no-tl", !e.target.checked);
    log.event("option", { transliteration: e.target.checked });
  };
  $("opt-focus").checked = stored("focus") === "1";
  document.body.classList.toggle("focus", $("opt-focus").checked);
  $("opt-focus").onchange = (e) => {
    document.body.classList.toggle("focus", e.target.checked);
    stored("focus", e.target.checked ? "1" : "0");
    log.event("option", { focus: e.target.checked });
  };
  $("opt-font").onchange = (e) => setFont(e.target.value);
  setFont(stored("font") || "quran", false);
  showToday();
  watchForSharing();
  $("mode-page").onclick = () => setMajlis(false);
  $("mode-majlis").onclick = () => setMajlis(true);
  setMajlis(params.has("watch") || params.has("majlis") || stored("majlis") === "1", false);
  $("size-down").onclick = () => setTextSize(state.textSize - 0.1);
  $("size-up").onclick = () => setTextSize(state.textSize + 0.1);
  setTextSize(Number(stored("text-size")) || 1, false);
  $("back").onclick = () => {
    $("back").hidden = true;
    scrollToNow();
  };
  addEventListener("scroll", showBack, { passive: true });
  $("resume").onclick = resume;
  showResume();
  document.addEventListener("visibilitychange", () => document.visibilityState === "visible" && keepAwake());
  $("unit-phrase").onclick = () => setPerLine(false);
  $("unit-line").onclick = () => setPerLine(true);
  setPerLine(stored("per-line") === "1", false);
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
    if (state.pressed) return (state.pressed = false); // that was a long-press (sharing)
    // Practice, stopped: a line marked left out, tapped, is "I did say it" (and tapped again, undone).
    if (state.review) return ln?.matches(".v-left, .dismissed") && dismiss(ln);
    if (!ln || !state.dua) return;
    // Practice: a red rosette tapped is "I did say that line" (the checker was wrong).
    if (state.practice && ln.classList.contains("v-left") && e.target.closest(".mark")) return dismiss(ln);
    const i = Number(ln.dataset.i);
    log.event("tap", { seg: state.duas[state.dua].segments[i].id, shown: state.segment });
    if (state.source && document.body.dataset.state === "following") seekTo(state.dua, state.duas[state.dua].segments[i].id);
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
  return { ...(following() === "reciter" ? RECITER : {}), ...(state.engine?.profile?.tc ?? {}), ...urlCfg("tc") };
}

// ?sc=nextMargin:0 / ?tc=switchConfirm:3,switchSure:0.9 (a;b;c for lists): stream decoder and
// tracker settings for page runs (scripts/bench_page.py variants); the session log echoes both.
function urlCfg(key) {
  const val = (x) => (x === "true" ? true : x === "false" ? false : Number.isNaN(Number(x)) ? x : Number(x));
  return Object.fromEntries((params.get(key) || "").split(",").filter((kv) => kv.includes(":"))
    .map((kv) => { const [k, v] = kv.split(":"); return [k, v.includes(";") ? v.split(";").map(val) : val(v)]; }));
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

// Word by word: the recited words light up one by one as they're said. Line by line: the whole
// line lights up at once and the words inside it aren't followed.
function setPerLine(on, remember = true) {
  state.perLine = on;
  $("unit-phrase").setAttribute("aria-checked", String(!on));
  $("unit-line").setAttribute("aria-checked", String(on));
  if (remember) {
    stored("per-line", on ? "1" : "0");
    log.event("option", { perLine: on });
  }
  const ln = state.lines?.get(state.segment);
  if (!ln) return;
  state.hl = null; // stops the glide loop; the next update restarts it
  state.token = null;
  setWords(ln, !on);
}

const TEXT_SIZES = [0.8, 1.6];

function setTextSize(size, remember = true) {
  const [lo, hi] = TEXT_SIZES;
  size = Math.round(Math.min(hi, Math.max(lo, size)) * 10) / 10;
  state.textSize = size;
  document.body.style.setProperty("--text", size);
  $("size-down").disabled = size <= lo;
  $("size-up").disabled = size >= hi;
  if (!remember) return;
  stored("text-size", String(size));
  log.event("option", { textSize: size });
  scrollToNow();
}

// Phones dim and lock in the middle of a long du'a; hold the screen on while
// following. The browser drops the lock when the tab is hidden, so it's taken
// again on coming back (init).
async function keepAwake() {
  if (!state.source || state.wake || !navigator.wakeLock) return;
  try {
    state.wake = await navigator.wakeLock.request("screen");
    state.wake.onrelease = () => (state.wake = null);
  } catch {} // refused (battery saver, no permission): carry on
}

// "Back to reciter" once the line being recited is out of sight.
function showBack() {
  const ln = state.lines?.get(state.segment);
  const r = ln?.isConnected && ln.getBoundingClientRect();
  $("back").hidden = !r || (r.bottom > 0 && r.top < innerHeight);
}

// Where the last recitation from the microphone reached, offered on the home
// screen: long du'as are often finished in more than one sitting.
const RESUME_DAYS = 7;

function saveProgress(dua, idx) {
  if (!state.stream) return; // a recording played back, or another screen's room
  const done = idx >= dua.segments.length - 3;
  stored("resume", done ? "" : JSON.stringify({ dua: dua.id, line: idx + 1, at: Date.now() }));
}

function savedProgress() {
  try {
    const p = JSON.parse(stored("resume") || "null");
    return p && state.duas[p.dua] && Date.now() - p.at < RESUME_DAYS * 864e5 ? p : null;
  } catch {
    return null;
  }
}

function showResume() {
  const p = savedProgress();
  $("resume").hidden = !p;
  if (p) $("resume").textContent = `continue ${state.duas[p.dua].name_en} from line ${p.line}`;
}

// Follows only that du'a; the tracker finds the line itself within a few seconds.
function resume() {
  const p = savedProgress();
  if (!p) return showResume();
  log.event("resume", { dua: p.dua, line: p.line });
  state.chosen = p.dua;
  if (state.practising) state.practiceFrom = p.line - 1; // practice can't find the line itself: it starts there
  begin(micSource);
}

// Audio seconds received per second of real time. A microphone whose sound arrives at the wrong
// speed breaks everything after it: on 2026-10-02 Firefox 157 with a USB headset delivered 2.00 s of
// audio per second (chunks heard twice), and the page followed a slowed-down, stuttering recitation.
// The same 2.0x on 2026-10-04 was two captures started by a second tap (see begin()); the headset
// itself ran at 1.00 that day, so the Firefox case may have been the same.
// Logged every 30 s; past 15% off, the reader is told instead of left with a page that wanders.
function checkClock(n) {
  const now = performance.now();
  const c = (state.clock ??= { t0: now, samples: -n, logged: 0, warned: false });
  c.samples += n; // counted from the first chunk's arrival
  const wall = (now - c.t0) / 1000;
  if (wall < 5) return;
  const ratio = c.samples / SR / wall;
  if (wall - c.logged >= 30) {
    c.logged = wall;
    log.event("clock", { ratio: Number(ratio.toFixed(3)) });
  }
  if (!c.warned && wall >= 8 && Math.abs(ratio - 1) > 0.15) {
    c.warned = true;
    log.event("clock_bad", { ratio: Number(ratio.toFixed(3)) });
    const text = `This microphone's sound is arriving at ${ratio.toFixed(1)}× speed, so I can't follow it. Try another browser or microphone.`;
    listenMsg(text);
    $("dua-also").textContent = text;
  }
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

// The host's phone sleeps and changes networks too: once a room is open, reopen
// it under the same code whenever the socket drops, and catch viewers up.
function openRoom(code) {
  const ws = new WebSocket(`${wsBase()}/ws/room/${code}?role=host`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if ("viewers" in m) $("share-viewers").textContent = m.viewers ? `${m.viewers} following` : "No one following yet";
  };
  ws.onclose = () => state.roomCode === code && setTimeout(() => openRoom(code).catch(() => {}), 2000);
  return new Promise((ok, err) => ((ws.onopen = ok), (ws.onerror = err))).then(() => {
    Object.assign(state, { room: ws, roomCode: code });
    if (state.last) ws.send(JSON.stringify(state.last));
  });
}

async function share() {
  $("menu").hidden = true;
  if (!state.room) {
    // Unambiguous letters only: people read this aloud across a room.
    await openRoom(Array.from({ length: 5 }, () => "ACDEFHJKMNPRTUVWXY"[Math.floor(Math.random() * 18)]).join(""));
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

// One capture at a time. Starting waits on the models, the audio context and the microphone, and every
// tap in that time (Start again, a du'a, a chip) used to start another capture beside the first: the
// engine then heard each chunk once per capture. On 2026-10-04 (session uuc5, Android) the even and odd
// chunks were the same audio 10 ms apart, the clock read 2.0x, and Munajat Kha'ifeen took 103 s to
// find and stuck at a line; repeated taps in headless Chrome gave 3.0x. A du'a picked while already
// listening locks onto it instead.
async function begin(makeSource) {
  if (state.node || state.starting) {
    if (state.node && state.chosen && makeSource === micSource) state.engine.lock(state.chosen);
    return;
  }
  state.starting = true;
  try {
    await startCapture(makeSource);
  } finally {
    state.starting = false;
  }
}

async function startCapture(makeSource) {
  $("samples").hidden = true;
  setState("loading");
  $("start").disabled = true;
  $("load").hidden = false;
  $("hint").textContent = "Preparing… (only the first time)";
  try {
    await state.engine.prepare((p) => ($("load-bar").style.width = `${Math.round(p)}%`));
  } catch (e) {
    $("hint").textContent = state.engine.remote ? "Couldn't reach the server." : "Couldn't load the speech model on this device.";
    $("start").disabled = false;
    return setState("idle");
  }
  $("load").hidden = true;
  $("start").disabled = false;
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
  // ?practice=<du'a id>: practise that du'a; ?practice=1: practise whichever is recited (replays, a link).
  if (params.has("practice")) state.practiceNext = true;
  if (!state.chosen && state.duas[params.get("practice")]) state.chosen = params.get("practice");
  await state.engine.start(update);
  if (state.chosen) state.engine.lock(state.chosen);
  state.practice = !!((state.practiceNext || (state.practising && !kids.on)) && state.engine.startPractice);
  state.practiceNext = false;
  document.body.classList.toggle("practice", state.practice);
  const track = state.stream?.getAudioTracks()[0];
  const mic = track?.getSettings() ?? {};
  log.start({
    engine: state.engine.kind, model: state.engine.model ?? state.mode?.model, source: makeSource.label ?? "mic",
    chosen: state.chosen, follow: following(), params: location.search, ua: navigator.userAgent,
    screen: [innerWidth, innerHeight, devicePixelRatio], audio_rate: state.ctx.sampleRate, load_ms: state.engine.loadMs,
    mic: track && { label: track.label, rate: mic.sampleRate, echo: mic.echoCancellation, noise: mic.noiseSuppression,
      agc: mic.autoGainControl },
    tracker: state.engine.tracker?.cfg, chunk: CHUNK,
    cores: navigator.hardwareConcurrency, isolated: self.crossOriginIsolated,
    words: state.engine.words ? { model: state.engine.ctcModel, hop: state.engine.ctcHop / SR,
      anchor_hop: state.engine.anchorHop / SR, follower: state.engine.followerKind,
      ...(state.engine.followerKind === "stream" ? { ...STREAM_DEFAULTS, ...state.engine.streamCfg }
        : { ...FOLLOW_DEFAULTS, ...state.engine.followCfg }) } : null,
  });
  const node = new AudioWorkletNode(state.ctx, "capture", { processorOptions: { chunk: CHUNK } });
  node.port.onmessage = (e) => {
    const x = e.data.x ?? e.data;
    log.audio(x);
    checkClock(x.length);
    state.engine.push(x, captureTime(e.data.t));
  };
  source.connect(node);
  // Feeds the listening star (meter below); a dead end, like the capture node.
  state.analyser ??= new AnalyserNode(state.ctx, { fftSize: 2048, smoothingTimeConstant: 0 });
  source.connect(state.analyser);
  Object.assign(state, { node, source, dua: null, segment: null, token: null, listeningSince: Date.now() });
  keepAwake();
  showListening();
  meter();
  if (state.practice && state.chosen) practiceBegin(state.chosen); // else once the du'a is found (render)
}

// While the du'a is being found, the star answers the reciter's voice: it glows
// and swells as they recite, turns a little faster, and breathes when they pause.
// Once it's found, the small seal in the top bar carries on the same way, so the
// page still shows that it hears. voice.js turns the microphone into a calm 0..1
// level; here a soft spring sits between that level and the size, and only
// transform and opacity change, which the compositor animates without repainting.
function meter() {
  const girih = document.querySelector(".girih");
  const glow = document.querySelector(".voice-glow");
  const seal = document.querySelector(".live svg");
  const { analyser } = state;
  const spectrum = new Float32Array(analyser.frequencyBinCount);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const voice = new VoiceLevel();
  const debug = params.has("debug") && $("latency");
  let size = 0, sizeV = 0; // spring toward the level
  let spin = 3, angle = 0; // deg/s and deg
  let last = performance.now();
  const frame = (now) => {
    const listening = document.body.dataset.state === "listening";
    if ((!listening && document.body.dataset.state !== "following") || !state.source) {
      girih.style.transform = glow.style.opacity = seal.style.transform = seal.style.opacity = "";
      kids.voice(null);
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

    const breath = 0.015 * Math.sin((now / 1000) * (2 * Math.PI / 4.5)) * (1 - level);
    if (kids.on) kids.voice(listening, level, size, now, still); // Noor, instead of the star and the seal
    else if (listening) {
      glow.style.opacity = Math.min(1, level * 1.15).toFixed(3);
      if (!still) girih.style.transform = `rotate(${angle.toFixed(2)}deg) scale(${(1 + 0.12 * size + breath).toFixed(4)})`;
    } else {
      seal.style.opacity = (0.45 + 0.55 * Math.min(1, level * 1.15)).toFixed(3);
      if (!still) seal.style.transform = `rotate(${angle.toFixed(2)}deg) scale(${(1 + 0.22 * size + breath).toFixed(4)})`;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function end(reason) {
  // Practice stopped by the reader: the text stays, with what was checked (a recording that ends, as
  // in page_replay.mjs, goes home as before).
  const review = state.practice && practiceEnd(reason === "stop");
  const { node, source, stream } = state;
  if (source && node) source.disconnect(node);
  if (source && state.analyser) source.disconnect(state.analyser);
  if (source === state.mediaSource) $("player").pause();
  stream?.getTracks().forEach((t) => t.stop());
  state.engine?.stop();
  state.wake?.release().catch(() => {});
  state.wake = null;
  if (state.room?.readyState === 1) state.room.send(JSON.stringify({ ended: true }));
  log.end(reason).then(showSessions);
  Object.assign(state, { node: null, source: null, stream: null, clock: null, dua: null, segment: null, chosen: null, hl: null,
    preview: false, guessesShown: "", guessSeen: new Map(), lapse: 0 });
  document.body.classList.remove("holding");
  kids.reset();
  $("guesses").hidden = true;
  $("menu").hidden = true;
  $("hint").textContent = "Tap to begin";
  $("back").hidden = true;
  showResume();
  showToday();
  setState(review ? "review" : "idle");
}

// -- rendering ------------------------------------------------------------------
function showListening() {
  setState("listening");
  $("dua-ar").textContent = "";
  $("dua-en").textContent = state.practice ? "Practice" : "Listening";
  $("dua-also").textContent = "";
  $("progress").style.width = "0";
  $("text").replaceChildren();
  $("folio-head").hidden = true;
  $("listen-msg").textContent = state.practice ? "Begin reciting the du'a you're practising" : "Begin reciting";
}

// The word follower: its messages place the line and word directly (no glide) while they
// keep coming; the tracker's updates still identify the du'a and drive the rest.
function renderWord(u) {
  state.wordAt = performance.now();
  state.wordLive = !!u.dua && u.dua === state.dua;
  if (!state.wordLive) return; // not locked yet, or the tracker changed du'a: its updates lead
  state.hl = null; // stops the glide loop
  if (u.segment !== state.segment) moveTo(state.duas[u.dua], u.segment);
  catchUp(u.token);
  if (u.pause != null) preview(u.pause);
}

// The follower catching up several words at once (it fell behind on words it didn't hear): run
// the highlight through them quickly rather than snapping over them.
function catchUp(token) {
  clearTimeout(state.catchTimer);
  const from = state.token;
  if (from == null || token == null || token <= from + 1 || state.perLine) return paintWords(token);
  let t = from;
  const next = () => {
    paintWords(++t);
    if (t < token) state.catchTimer = setTimeout(next, 70);
  };
  next();
}

function preview(on) {
  state.lines.get(state.segment + 1)?.classList.toggle("coming", on);
  if (on !== !!state.preview) log.event("preview", { on });
  state.preview = on;
}

// "I'm here": the tapped line is where the reciter is. The tracker restarts there
// (Tracker.seek) and the display goes there now, at the line's first word.
function seekTo(duaId, segment) {
  const p = state.engine.seek?.(duaId, segment);
  if (state.practice) {
    state.engine.practiceFrom?.(segment); // the start line, until the first line is judged
    if (!state.verdicts?.size) state.practiceStart = state.duas[duaId].segments.findIndex((s) => s.id === segment);
  }
  log.event("seek", { dua: duaId, seg: segment });
  render({ dua: duaId, segment, token: p?.token ?? 0, speed: 0, unknown: 0, pause: false, heard: "", ms: 0,
    candidates: state.last?.candidates || [], sameAs: [] });
}

function render(u) {
  if (u.word) return renderWord(u);
  state.last = u;
  if (params.has("debug")) {
    $("heard").textContent = u.heard || "…";
    $("latency").textContent = `${Math.round(u.ms)} ms`;
  }
  // Through a lapse the page holds still; from the second update the top seal
  // redraws itself, as on the listening screen, so the stillness reads as searching.
  state.lapse = u.dua ? 0 : (state.lapse || 0) + 1;
  document.body.classList.toggle("holding", !!state.dua && state.lapse >= 2);
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
    if (state.stream) addRecent(u.dua);
    state.dua = u.dua;
    state.segment = null;
    $("dua-ar").textContent = dua.name_ar;
    $("dua-en").textContent = dua.name_en;
    buildText(dua);
    reveal(document.body.dataset.state === "listening");
    setState("following");
    if (state.practice && !state.practiceDua) practiceBegin(u.dua, u.segment);
  }
  const words = state.wordLive && performance.now() - state.wordAt < 1000;
  if (!words) {
    if (u.segment !== state.segment) moveTo(dua, u.segment);
    glide(u);
  }
  showSameAs(u.sameAs || []);
  if (!words || u.pause == null) preview(!!u.pause); // while the follower places words, it previews too
}

// The du'a is found: the listening star opens out while the text rises in
// (style.css, body.revealing), and the first line is placed without a scroll.
function reveal(fromListening) {
  const body = document.body;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  body.classList.add("revealing");
  body.classList.toggle("from-listen", fromListening && !still);
  state.placeNow = true;
  clearTimeout(state.revealTimer);
  state.revealTimer = setTimeout(() => body.classList.remove("revealing", "from-listen"), 1700);
}

// Inside a passage another text shares word for word (Ayat al-Kursi in Sahifa 54),
// the recitation could be either: say so under the title until it's told apart.
function showSameAs(ids) {
  const names = ids.map((id) => state.duas[id]?.name_en).filter(Boolean);
  const text = names.length ? `Also in ${names.slice(0, 2).join(", ")}${names.length > 2 ? "…" : ""}`
    : state.practice && state.practiceStart != null ? `Practice · from line ${state.practiceStart + 1}` : "";
  if ($("dua-also").textContent === text) return;
  $("dua-also").textContent = text;
  log.event("same_as", { ids });
}

function listenMsg(text) {
  if ($("listen-msg").textContent === text) return;
  $("listen-msg").textContent = text;
  log.event("msg", { text });
}

// -- practice mode (practice.js) ------------------------------------------------------
// "practise" beneath the seal (setPractising), then recite. A du'a chosen (the picker, a chip) is
// checked from its first line, or from a line tapped before the first verdict; otherwise the du'a is
// found from the recitation, checked from the line it was found at, and the page follows only it.
// Each line's rosette then says what the checker heard: gilded once heard, red if left out (tapped, "I
// did say it"), its outline broken if the sound left it unsure. Nothing is gilded just for being
// passed. Stopped, the text stays with the verdicts (showReview). Lines left out are remembered per
// du'a: marked the next time, and offered on the home screen and in the picker.
function setPractising(on, remember = true) {
  state.practising = on;
  document.body.classList.toggle("practising", on);
  $("mode-follow").setAttribute("aria-checked", String(!on));
  $("mode-practise").setAttribute("aria-checked", String(on));
  $("tagline").textContent = on ? "Recite, and each line is checked." : "Recite, and the words follow you.";
  $("start").setAttribute("aria-label", on ? "Start practising" : "Start listening");
  $("play").hidden = $("play-dot").hidden = on;
  $("hide-opt").hidden = $("hide-dot").hidden = !on;
  showToday();
  if (remember) log.event("option", { practising: on });
}

// "Hide the text until it's heard": on the home screen while practising, and in the menu (practice.css).
function setHideText(on, remember = true) {
  document.body.classList.toggle("hide-text", on);
  $("opt-hide").checked = $("home-hide").checked = on;
  if (!remember) return;
  stored("practice-hide", on ? "1" : "0");
  log.event("option", { hideText: on });
}

function practiceBegin(duaId, segment = null) {
  const dua = state.duas[duaId];
  Object.assign(state, { practiceDua: duaId, verdicts: new Map(), unsureRun: 0, unclearShown: false, hints: new Map(),
    hintCount: 0 });
  if (segment != null) state.engine.lock(duaId); // found from the recitation: follow only it from here on
  state.engine.startPractice(dua.id, onVerdicts);
  // Where it was found; or, a du'a chosen, the line asked for (practise again, a chip, ?from=<line
  // number>) or the top.
  const found = dua.segments.findIndex((s) => s.id === segment);
  const from = state.practiceFrom ?? (Number(params.get("from")) || 1) - 1;
  state.practiceFrom = null;
  const i = found >= 0 ? found : Math.max(0, Math.min(dua.segments.length - 1, from));
  log.event("practice_start", { dua: dua.id, line: i + 1, found: found >= 0, c_far: state.engine.streamCfg?.cFar });
  seekTo(dua.id, dua.segments[i].id);
  const weak = practiceWeights(dua.id);
  for (const [id, ln] of state.lines) ln.classList.toggle("weak", (weak[id] || 0) >= 0.5);
  if (found < 0 && state.stream) toast("Tap a line to start there instead", 3500);
}

const storedObject = (key) => {
  try {
    const v = JSON.parse(stored(key) || "{}");
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
};
// Per du'a: segment id -> how often left out lately (savePractice); du'a id -> when last practised.
const practiceWeights = (id) => storedObject(`practice:${id}`);
const practiceSeen = () => storedObject("practice-seen");

// The line numbers to work on: left out lately (the rosette's heavier outline, practice.css .weak).
function weakLines(id) {
  const w = practiceWeights(id);
  return state.duas[id].segments.map((s, i) => ((w[s.id] || 0) >= 0.5 ? i + 1 : 0)).filter(Boolean);
}

// Under a du'a's name in the picker while practising.
function practiceNote(id) {
  const at = practiceSeen()[id];
  if (!at) return "";
  const day = (t) => new Date(t).setHours(0, 0, 0, 0);
  const days = Math.round((day(Date.now()) - day(at)) / 864e5);
  const n = weakLines(id).length;
  return `Practised ${days <= 0 ? "today" : days === 1 ? "yesterday" : `${days} days ago`}`
    + (n ? ` · ${n} line${n === 1 ? "" : "s"} to work on` : "");
}

// Practising, on the home screen in place of today's du'as: those with lines to work on, the most
// recently practised first, each from the line before the first of them.
function showWorkOn() {
  const seen = practiceSeen();
  const ids = Object.keys(seen).filter((id) => state.duas[id] && weakLines(id).length)
    .sort((a, b) => seen[b] - seen[a]).slice(0, 3);
  if (!ids.length) return false;
  $("today").hidden = false;
  $("today-label").textContent = "To work on";
  $("today-chips").replaceChildren(...ids.map((id) => {
    const weak = weakLines(id);
    const b = Object.assign(document.createElement("button"), { className: "chip", textContent: state.duas[id].name_en });
    b.append(Object.assign(document.createElement("span"), { className: "note",
      textContent: `${weak.length} line${weak.length === 1 ? "" : "s"} · from line ${Math.max(1, weak[0] - 1)}` }));
    b.onclick = () => {
      log.event("today", { dua: id, weak });
      Object.assign(state, { chosen: id, practiceFrom: Math.max(0, weak[0] - 2) });
      begin(micSource);
    };
    return b;
  }));
  return true;
}

const VERDICT_CLASS = { heard: "v-heard", "left out": "v-left", "not sure": "v-unsure" };

function onVerdicts(list) {
  for (const v of list) {
    state.verdicts.set(v.segment, v.verdict);
    const ln = state.lines?.get(v.segment);
    ln?.classList.remove("v-heard", "v-left", "v-unsure", "dismissed");
    ln?.classList.add(VERDICT_CLASS[v.verdict]);
    log.event("practice", { seg: v.segment, t: Number(v.t.toFixed(2)), score: Number(v.score.toFixed(1)), shown: v.shown,
      verdict: v.verdict, check_ms: v.ms == null ? null : Number(v.ms.toFixed(1)) });
    // Several lines in a row it couldn't vouch for: the sound (echo, a phone far away), not the reader.
    state.unsureRun = v.verdict === "not sure" ? state.unsureRun + 1 : 0;
    if (state.unsureRun >= 3 && !state.unclearShown) {
      state.unclearShown = true;
      toast("I can't hear clearly enough to check every line", 5000);
    }
  }
}

// "I did say it": a line marked left out counts as heard (logged while listening). Once stopped, a
// second tap takes it back.
function dismiss(ln) {
  const dua = state.review?.dua ?? state.duas[state.dua];
  const seg = dua.segments[ln.dataset.i].id;
  const undo = ln.classList.contains("dismissed");
  ln.classList.toggle("v-left", undo);
  ln.classList.toggle("v-heard", !undo);
  ln.classList.toggle("dismissed", !undo);
  state.verdicts.set(seg, undo ? "left out" : "heard");
  log.event("dismiss", { seg, undo });
  if (!state.review) return;
  savePractice();
  showReview();
  toast(undo ? "Marked as left out" : "Marked as said");
}

// The line numbers of each verdict.
function verdictLines(dua) {
  const no = new Map(dua.segments.map((s, i) => [s.id, i + 1]));
  const lines = (kind) => [...state.verdicts].filter(([, v]) => v === kind).map(([seg]) => no.get(seg)).sort((a, b) => a - b);
  return { heard: lines("heard"), left: lines("left out"), unsure: lines("not sure") };
}

// The session is over: the lines not judged yet are judged, through the furthest one reached. Stopped
// by the reader (review), the text stays with the verdicts and this returns true; otherwise (a
// recording ended) a summary in passing.
function practiceEnd(review) {
  // Stopping partway through a line isn't leaving it out. The furthest line reached is judged now on
  // the part said so far, and the display was on it, so the reader was in it: found left out, it
  // stays unjudged (logged). On 2026-10-04 a replay of Du'a Ahd stopped mid-line 6 marked it left out.
  const last = state.engine.finishPractice?.() || [];
  const furthest = Math.max(-1, ...last.map((v) => v.line));
  for (const v of last) {
    if (v.line === furthest && v.verdict === "left out") {
      log.event("practice_unfinished", { seg: v.segment, score: Number(v.score.toFixed(1)) });
    }
  }
  onVerdicts(last.filter((v) => v.line !== furthest || v.verdict !== "left out"));
  const begun = !!state.practiceDua;
  const dua = state.duas[state.practiceDua];
  Object.assign(state, { practice: false, practiceDua: null });
  if (!begun || !state.verdicts.size) {
    document.body.classList.remove("practice");
    if (begun) toast("Stopped before any line was checked", 4000);
    return false;
  }
  state.review = { dua, base: practiceWeights(dua.id), from: state.practiceStart, hints: state.hintCount };
  const { heard, left, unsure } = verdictLines(dua);
  log.event("practice_end", { heard: heard.length, left, unsure, hints: state.hintCount });
  savePractice();
  if (review) {
    showReview();
    return true;
  }
  state.review = null;
  document.body.classList.remove("practice");
  const parts = [`${heard.length} line${heard.length === 1 ? "" : "s"} heard`];
  if (left.length) parts.push(`left out: ${left.join(", ")}`);
  if (unsure.length) parts.push(`${unsure.length} not sure`); // the sound's doing, not the reader's: no list
  toast(parts.join(" · "), 8000);
  return false;
}

// Remembered per line: up one for each time left out, halved each time heard. Counted from the
// weights before this session, so a line marked as said after stopping counts as heard.
function savePractice() {
  const { dua, base } = state.review;
  const w = { ...base };
  for (const [seg, v] of state.verdicts) {
    const x = (w[seg] || 0) * (v === "heard" ? 0.5 : 1) + (v === "left out" ? 1 : 0);
    if (x >= 0.1) w[seg] = Number(x.toFixed(2));
    else delete w[seg];
  }
  stored(`practice:${dua.id}`, JSON.stringify(w));
  stored("practice-seen", JSON.stringify({ ...practiceSeen(), [dua.id]: Date.now() }));
}

// Stopped: the whole text, every line shown with its verdict (practice.css, data-state review), and a
// card: the counts, the lines left out (a tap goes to the line), practise again from the same line.
function showReview() {
  const { dua, hints } = state.review;
  const { heard, left, unsure } = verdictLines(dua);
  const n = heard.length + left.length + unsure.length;
  const sum = [`${n} line${n === 1 ? "" : "s"} checked`, `${heard.length} heard`];
  if (left.length) sum.push(`${left.length} left out`);
  if (unsure.length) sum.push(`${unsure.length} not sure`);
  $("review-sum").textContent = sum.join(" · ");
  $("review-left").replaceChildren(...left.map((k) => {
    const b = Object.assign(document.createElement("button"), { className: "review-line" });
    b.setAttribute("aria-label", `Line ${k}`);
    b.append(marker(k));
    b.onclick = () => state.lines.get(dua.segments[k - 1].id)?.scrollIntoView({ behavior: "smooth", block: "center" });
    return b;
  }));
  const note = [];
  if (hints) note.push(hints === 1 ? "1 word shown as a hint." : `${hints} words shown as hints.`);
  if (left.length) note.push("Tap a line in red if you did say it.");
  note.push("This checks whether each line was said, not its words, vowels or tajweed.");
  $("review-note").textContent = note.join(" ");
  for (const ln of state.lines.values()) ln.classList.remove("now", "next", "coming");
  $("review").hidden = false;
}

function closeReview() {
  if (!state.review) return;
  state.review = null;
  $("review").hidden = true;
  document.body.classList.remove("practice");
  setState("idle");
  scrollTo({ top: 0, behavior: "instant" });
}

function practiseAgain() {
  const { dua, from } = state.review;
  closeReview();
  Object.assign(state, { chosen: dua.id, practiceFrom: from, practiceNext: true });
  begin(micSource).finally(() => (state.practiceNext = false)); // a refused microphone: not the next start
}

// Hidden text: one more word each tap, logged as a hint, not a mistake. The first word of the line
// being read that isn't shown yet; once it's all shown, the first of the next line.
function hintWord() {
  if (!state.practice || !state.dua) return;
  for (const seg of [state.segment, state.segment + 1]) {
    const ln = state.lines.get(seg);
    if (!ln || ln.matches(".v-heard, .v-left, .passed")) continue;
    if (!ln.querySelector(".wd")) setWords(ln, true);
    const next = [...ln.querySelectorAll(".wd")].find((w) => !w.matches(".said, .w, .hinted"));
    if (!next) continue;
    const i = Number(next.dataset.i);
    next.classList.add("hinted");
    if (!state.hints.has(seg)) state.hints.set(seg, new Set());
    state.hints.get(seg).add(i);
    state.hintCount += 1;
    log.event("hint", { seg, word: i });
    return;
  }
}

// -- choosing a du'a --------------------------------------------------------------
function openPicker() {
  $("search").value = "";
  fillPicker("");
  kids.fillShelf();
  $("picker").hidden = false;
  $("search").focus();
}

function closePicker() {
  $("picker").hidden = true;
}

// Favourites (starred in the picker) and recents (du'as followed from the
// microphone) head the list until a search is typed.
const RECENTS = 5;

function storedList(key) {
  try {
    const v = JSON.parse(stored(key) || "[]");
    return Array.isArray(v) ? v.filter((id) => state.duas[id]) : [];
  } catch {
    return [];
  }
}

function addRecent(id) {
  stored("recent", JSON.stringify([id, ...storedList("recent").filter((r) => r !== id)].slice(0, RECENTS)));
}

function toggleFavourite(id) {
  const favs = storedList("favourites");
  const on = !favs.includes(id);
  stored("favourites", JSON.stringify(on ? [...favs, id] : favs.filter((f) => f !== id)));
  log.event("favourite", { dua: id, on });
  return on;
}

function pickerItem(d, favs) {
  const li = document.createElement("li");
  const b = document.createElement("button");
  b.className = "pick";
  const en = document.createElement("span");
  en.textContent = d.name_en;
  const note = (state.practising && practiceNote(d.id)) || noteFor(d.id);
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
  const star = Object.assign(document.createElement("button"), { className: "fav", textContent: "★" });
  const mark = (on) => {
    star.setAttribute("aria-pressed", String(on));
    star.setAttribute("aria-label", on ? `Remove ${d.name_en} from favourites` : `Add ${d.name_en} to favourites`);
  };
  mark(favs.includes(d.id));
  star.onclick = () => {
    mark(toggleFavourite(d.id));
    if (!$("search").value.trim()) fillPicker(""); // keep the Favourites section in step
  };
  li.append(b, star);
  return li;
}

function fillPicker(query) {
  const q = query.trim().toLowerCase();
  const favs = storedList("favourites");
  const all = Object.values(state.duas)
    .filter((d) => !q || d.name_en.toLowerCase().includes(q) || d.name_ar.includes(query.trim()))
    .sort((a, b) => a.name_en.localeCompare(b.name_en));
  const items = [];
  const section = (title, duas) => {
    if (!duas.length) return;
    items.push(Object.assign(document.createElement("li"), { className: "section", textContent: title }));
    items.push(...duas.map((d) => pickerItem(d, favs)));
  };
  if (q) section("", all);
  else {
    // Practising: the du'as practised before head the list, the most recent first (practiceNote).
    const seen = state.practising ? practiceSeen() : {};
    const practised = Object.keys(seen).filter((id) => state.duas[id]).sort((a, b) => seen[b] - seen[a]);
    const recent = storedList("recent").filter((id) => !favs.includes(id) && !practised.includes(id));
    section("Practised", practised.map((id) => state.duas[id]));
    section("Favourites", favs.map((id) => state.duas[id]));
    section("Recent", recent.map((id) => state.duas[id]));
    section(favs.length || recent.length || practised.length ? "All" : "", all);
  }
  $("picker-list").replaceChildren(...items.filter((li) => li.className !== "section" || li.textContent));
}

// -- what's customarily recited now (NOTES above), offered on the home screen ------
// The Islamic day begins at sunset, taken here as 6 pm: a Thursday evening is
// already the night of Friday.
const hijri = (date) => {
  try {
    const parts = new Intl.DateTimeFormat("en-u-ca-islamic-umalqura", { day: "numeric", month: "numeric" })
      .formatToParts(date);
    const get = (t) => Number(parts.find((p) => p.type === t)?.value);
    return { day: get("day"), month: get("month") };
  } catch {
    return null;
  }
};
const DAY_IDS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];

function duasForNow(now = new Date()) {
  const h = now.getHours();
  const evening = h >= 18;
  const predawn = h < 5;
  const wd = now.getDay();
  const today = hijri(now);
  const night = evening ? hijri(new Date(now.getTime() + 864e5)) : predawn ? today : null; // whose night it is
  const ids = [];
  if (night?.month === 9) {
    ids.push(`dua-ramadan-${night.day}-night`, "dua-iftitah");
    if ([19, 21, 23].includes(night.day)) ids.push("dua-jawshan-kabir");
    if ([13, 14, 15].includes(night.day)) ids.push("dua-mujeer");
  }
  if (predawn && today?.month === 9) ids.push("dua-abu-hamza-thumali", "dua-baha", "dua-tasbih-suhoor");
  if (!evening && !predawn && today?.month === 9) ids.push(`dua-ramadan-${today.day}`);
  if (!evening && today?.month === 12 && today.day === 9) ids.push("dua-arafat");
  if (!evening && today?.month === 1 && today.day === 10) ids.push("ziyarat-ashura");
  const eid = (today?.month === 10 && today.day === 1) || (today?.month === 12 && today.day === 10);
  if ((wd === 4 && evening) || (wd === 5 && predawn)) ids.push("dua-kumayl");
  if ((wd === 2 && evening) || (wd === 3 && predawn)) ids.push("dua-tawassul");
  if (!evening && !predawn && h < 12 && (wd === 5 || eid)) ids.push("dua-nudbah");
  if (wd === 5 && h >= 15 && !evening) ids.push("dua-simaat");
  if (!evening && !predawn && h < 12) ids.push("dua-aahad");
  if (!evening && !predawn) ids.push(`dua-${DAY_IDS[wd]}`, `ziyarat-${DAY_IDS[wd]}`);
  return [...new Set(ids)].filter((id) => state.duas[id]).slice(0, 3);
}

function showToday() {
  if (state.practising && !kids.on && showWorkOn()) return;
  const ids = duasForNow();
  $("today").hidden = !ids.length;
  const h = new Date().getHours();
  $("today-label").textContent = h >= 18 || h < 5 ? "For tonight" : "For today";
  $("today-chips").replaceChildren(...ids.map((id) => {
    const d = state.duas[id];
    const b = Object.assign(document.createElement("button"), { className: "chip", textContent: d.name_en });
    b.onclick = () => {
      log.event("today", { dua: id });
      state.chosen = id;
      begin(micSource);
    };
    kids.chip(b, id);
    return b;
  }));
}

// -- sharing a line: press and hold it -----------------------------------------------
function watchForSharing() {
  let timer = null, start = null;
  const cancel = () => clearTimeout(timer);
  $("text").addEventListener("pointerdown", (e) => {
    const ln = e.target.closest(".ln");
    if (!ln || !state.dua) return;
    start = [e.clientX, e.clientY];
    timer = setTimeout(() => {
      state.pressed = true; // the click that follows isn't a tap on the line
      shareLine(Number(ln.dataset.i));
    }, 550);
  });
  $("text").addEventListener("pointermove", (e) => {
    if (start && Math.hypot(e.clientX - start[0], e.clientY - start[1]) > 10) cancel();
  });
  for (const ev of ["pointerup", "pointercancel", "pointerleave"]) $("text").addEventListener(ev, cancel);
  $("text").addEventListener("contextmenu", (e) => e.target.closest(".ln") && e.preventDefault());
}

async function shareLine(i) {
  const dua = state.duas[state.dua];
  const s = dua.segments[i];
  const text = [s.ar, tidyTl(s.tl), tidyEn(s.en), `${dua.name_en}, line ${i + 1}`].filter(Boolean).join("\n\n");
  log.event("share_line", { seg: s.id });
  navigator.vibrate?.(15);
  if (navigator.share) {
    try {
      return await navigator.share({ text });
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  try {
    await navigator.clipboard.writeText(text);
    toast("Line copied");
  } catch {
    toast("Couldn't copy the line");
  }
}

function toast(text, ms = 2000) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => ($("toast").hidden = true), ms);
}

// -- Arabic font -----------------------------------------------------------------------
// Amiri Quran loads with the page; the others only once chosen.
const FONTS = {
  quran: { family: '"Amiri Quran", Amiri, serif' },
  amiri: { family: "Amiri, serif" },
  scheherazade: { family: '"Scheherazade New", Amiri, serif', css: "Scheherazade+New:wght@400;700" },
  noto: { family: '"Noto Naskh Arabic", Amiri, serif', css: "Noto+Naskh+Arabic:wght@400;600" },
  nastaliq: { family: '"Noto Nastaliq Urdu", Amiri, serif', css: "Noto+Nastaliq+Urdu:wght@400;600" },
};

function setFont(key, remember = true) {
  const font = FONTS[key] || FONTS.quran;
  key = FONTS[key] ? key : "quran";
  if (font.css && !document.querySelector(`link[data-font="${key}"]`)) {
    document.head.append(Object.assign(document.createElement("link"), {
      rel: "stylesheet", href: `https://fonts.googleapis.com/css2?family=${font.css}&display=swap`,
    }));
    document.head.lastChild.dataset.font = key;
  }
  document.documentElement.style.setProperty("--naskh", font.family);
  document.body.classList.toggle("nastaliq", key === "nastaliq");
  $("opt-font").value = key;
  if (!remember) return;
  stored("font", key);
  log.event("option", { font: key });
  scrollToNow();
}

// A guess stays offered for GUESS_KEEP_MS after its likelihood falls, faded: an ordinary voice
// gives the phone's Whisper a few good windows and then poor ones, and the chip was gone before
// it could be tapped (l8s3: Dua Kumayl at 0.70 for 2 s, then nothing for the rest of the session).
const GUESS_KEEP_MS = 12000;

function showGuesses(candidates) {
  const now = Date.now();
  const seen = (state.guessSeen ??= new Map());
  for (const c of candidates) if (c.p >= 0.08) seen.set(c.id, { p: Math.max(c.p, seen.get(c.id)?.p ?? 0), at: now });
  for (const [id, g] of seen) if (now - g.at > GUESS_KEEP_MS) seen.delete(id);
  const current = candidates.filter((c) => c.p >= 0.08);
  const kept = [...seen].filter(([id]) => !current.some((c) => c.id === id))
    .sort((a, b) => b[1].p - a[1].p).map(([id]) => ({ id, p: 0, kept: true }));
  const likely = [...current, ...kept].slice(0, 3);
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
      b.style.opacity = (0.45 + 0.55 * (c.p / top)).toFixed(2); // the likeliest stands out as it firms up; kept ones fade
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
      kids.chip(b, c.id);
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
        Object.assign(document.createElement("p"), { className: "tl", textContent: tidyTl(s.tl) }),
        Object.assign(document.createElement("p"), { className: "en", textContent: tidyEn(s.en) }),
      );
      const gloss = Object.assign(document.createElement("div"), { className: "gloss" });
      gloss.append(inner);
      ln.append(ar, gloss);
      state.lines.set(s.id, ln);
      return ln;
    }),
    closing(),
  );
  kids.hang(dua);
}

// Under the last line, the seal the recitation began with; it draws itself
// once the last line is recited (finishing() below).
function closing() {
  const end = Object.assign(document.createElement("div"), { className: "end", id: "end" });
  end.setAttribute("aria-hidden", "true");
  end.innerHTML = '<svg viewBox="-50 -50 100 100"><use href="#khatam" x="-50" y="-50" width="100" height="100"/></svg>';
  return end;
}

function finishing(on) {
  $("end")?.classList.toggle("done", on);
  state.lines?.get(state.duas[state.dua]?.segments.at(-1).id)?.classList.toggle("sealed", on); // its rosette gilded too
  kids.finished(on, state.dua, !!state.stream);
}

// The sources' transliterations come in several styles (ALL CAPS, backticks
// or curly quotes for the ayn, stray non-breaking spaces): one look for all.
function tidy(text) {
  return (text || "").replace(/ /g, " ").replace(/\s+/g, " ").replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([,;:])(?=[^\s\d])/g, "$1 ").trim();
}
const capital = (t) => t.charAt(0).toUpperCase() + t.slice(1);

function tidyTl(text) {
  let t = tidy(text).replace(/[`‘]/g, "‘").replace(/['’´]/g, "’");
  if (!/[a-z]/.test(t)) t = t.toLowerCase(); // shouted in capitals
  return capital(t);
}

function tidyEn(text) {
  return capital(tidy(text));
}

// Only the line being recited is split into words; the rest stay plain text. Practising, words shown
// as hints (hintWord) stay shown.
function setWords(ln, split) {
  const ar = ln.querySelector(".ar");
  const seg = state.duas[state.dua].segments[ln.dataset.i];
  const hinted = (split && state.practice && state.hints?.get(seg.id)) || new Set();
  const words = split
    ? seg.ar.split(/\s+/).filter(Boolean).flatMap((w, i) => {
      const span = Object.assign(document.createElement("span"), { className: hinted.has(i) ? "wd hinted" : "wd", textContent: w });
      span.dataset.i = i;
      return [span, " "];
    })
    : [seg.ar, " "];
  ar.replaceChildren(...words, ar.querySelector(".mark"));
}

function moveTo(dua, segment) {
  clearTimeout(state.catchTimer); // a catch-up run belongs to the line it started on
  const prev = state.lines.get(state.segment);
  // Practice, text hidden: a line the reader went on from, having said at least half of it word by
  // word, shows whole (practice.css). A line skipped, or only passed through, waits for its verdict.
  if (state.practice && prev && Number(state.lines.get(segment)?.dataset.i) > Number(prev.dataset.i)) {
    const wds = prev.querySelectorAll(".wd");
    if (state.perLine || prev.querySelectorAll(".wd.said, .wd.w").length * 2 >= wds.length) prev.classList.add("passed");
  }
  if (prev) setWords(prev, false);
  state.segment = segment;
  state.token = null;
  const idx = dua.segments.findIndex((s) => s.id === segment);
  log.event("line", { seg: segment, idx });
  $("progress").style.width = `${((idx + 1) / dua.segments.length) * 100}%`;
  saveProgress(dua, idx);
  kids.progress(idx);
  finishing(state.perLine && idx === dua.segments.length - 1); // word by word: at the last word (paintWords)
  for (const [id, ln] of state.lines) {
    ln.classList.toggle("now", id === segment);
    ln.classList.toggle("past", id < segment);
    ln.classList.toggle("next", id === segment + 1);
    ln.classList.remove("coming");
  }
  setWords(state.lines.get(segment), !state.perLine);
  scrollToNow();
}

// Between updates the highlight glides at the reciter's pace (display.js)
// instead of hopping a word or two once a second.
function glide(u) {
  clearTimeout(state.catchTimer);
  const n = state.lines.get(state.segment).querySelectorAll(".wd").length;
  if (u.token == null || !n || params.get("glide") === "0") return paintWords(u.token);
  state.hl ??= new Highlight();
  state.pace = u.speed ?? 1;
  state.hl.update(performance.now() / 1000, `${u.dua}:${u.segment}`, u.token, 0, n, state.hushed ? 0 : state.pace,
    !!u.back);
  if (state.gliding) return;
  state.gliding = true;
  const frame = () => {
    if (!state.hl?.line || !state.lines?.get(state.segment)) return (state.gliding = false);
    const t = performance.now() / 1000;
    // Between updates, the page's own stop detector: the reciter stopped (or went on) since the last one.
    const q = state.engine?.ear?.quiet;
    const hushed = q != null && q > DEFAULTS.stillAfter;
    if (hushed !== !!state.hushed) {
      state.hushed = hushed;
      state.hl.pace(t, hushed ? 0 : state.pace);
      log.event("hush", { on: hushed });
    }
    paintWords(state.hl.word(t));
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function paintWords(token) {
  if (state.perLine || token === state.token) return;
  state.token = token;
  log.event("word", { token });
  const spans = state.lines.get(state.segment).querySelectorAll(".wd");
  for (const span of spans) {
    const i = Number(span.dataset.i);
    span.classList.toggle("said", i < token);
    span.classList.toggle("w", i === token);
  }
  if (state.segment === state.duas[state.dua].segments.at(-1).id) finishing(token >= spans.length - 1);
}

// Keeps the recited line centred. The old line shrinks and the new one grows over
// about half a second, moving the line down the page as it goes, so scrolling to
// where it is now and correcting later jumps down then back up. Instead the spring
// works on where the line sits on screen: every frame the page scrolls so the line's
// centre is where a critically damped spring says it should be, and the line glides
// to the middle however the text around it is resizing. A hand on the screen stops
// it until the next line.
function scrollToNow() {
  const ln = state.lines?.get(state.segment);
  if (!ln) return;
  const instant = state.placeNow || matchMedia("(prefers-reduced-motion: reduce)").matches;
  state.placeNow = false;
  const centre = () => {
    const r = ln.getBoundingClientRect();
    return r.top + r.height / 2;
  };
  const run = { p: centre(), v: 0, t0: performance.now(), last: performance.now() };
  state.follow = run;
  const w = 9; // rad/s: settles in about half a second
  const frame = (now) => {
    if (state.follow !== run || !ln.isConnected) return;
    const dt = Math.min(0.05, (now - run.last) / 1000);
    run.last = now;
    const goal = innerHeight * 0.48;
    if (instant) run.p = goal;
    else {
      run.v += (w * w * (goal - run.p) - 2 * w * run.v) * dt;
      run.p += run.v * dt;
    }
    scrollTo({ top: scrollY + centre() - run.p, behavior: "instant" });
    const settled = Math.abs(goal - run.p) < 0.5 && Math.abs(run.v) < 5;
    if (now - run.t0 < 700 || !settled) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}
// Scrolling by hand wins over the follower.
for (const ev of ["wheel", "touchstart"]) addEventListener(ev, () => (state.follow = null), { passive: true });

init();
