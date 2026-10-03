// Whether the phone runs Whisper on a window, and why not: the browser's speech gate.
// Pure functions over 16 kHz Float32Array audio; Silero runs through `vadProbs`,
// supplied by the caller (asr-worker.js in the page, scripts/gate_replay.mjs in Node),
// so the replay harness exercises this exact code rather than a Python copy.
//
// Policies (?gate=... on the page; the default stays "legacy"):
//   legacy           Silero speech run in the last 1.5 s (asr.speech_in_tail), as before
//   energy_assisted  legacy, OR a run of >= 7 frames (32 ms) at >= floor + 6 dB in the last
//                    1.5 s, floor = max(-70 dBFS, 10th-percentile frame energy of the window):
//                    quietAtEnd's energy evidence with the speech rule's run length. For
//                    melodic, drawn-out recitation that Silero scores as non-speech.
//   ungated          no Silero gate; only the -45 dBFS floor
// The -45 dBFS tail floor applies to all three (a level threshold, not digital silence).

export const SR = 16000;
export const FRAME = 512;
export const SILENCE_DBFS = -45;
export const POLICIES = ["legacy", "energy_assisted", "ungated"];

// Silero v6 probabilities for whole 512-sample frames, fresh state, framed like faster-whisper:
// each frame is [64 samples of the previous frame | 512 samples], and (a quirk of
// faster-whisper's, mirrored for parity) with zeroLast the last 64 samples of the last frame
// are zeroed. `run(feeds)` is an onnxruntime session's run; `Tensor` its tensor class.
export function makeVadProbs(run, Tensor) {
  return async (padded, zeroLast = false) => {
    const frames = padded.length / FRAME;
    const input = new Float32Array(frames * 576);
    for (let f = 0; f < frames; f++) {
      if (f > 0) input.set(padded.subarray(f * FRAME - 64, f * FRAME), f * 576);
      input.set(padded.subarray(f * FRAME, (f + 1) * FRAME), f * 576 + 64);
    }
    if (zeroLast) input.fill(0, frames * 576 - 64);
    const zeros = () => new Tensor("float32", new Float32Array(128), [1, 1, 128]);
    const out = await run({ input: new Tensor("float32", input, [frames, 576]), h: zeros(), c: zeros() });
    return out.speech_probs.data;
  };
}

export function rmsDb(x) {
  let e = 0;
  for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return x.length ? 20 * Math.log10(Math.sqrt(e / x.length) + 1e-12) : -240;
}

// Frame energies (dB) of whole 512-sample frames, aligned to the end of `x`.
export function frameDb(x) {
  const n = Math.floor(x.length / FRAME);
  const off = x.length - n * FRAME;
  const db = new Float64Array(n);
  for (let f = 0; f < n; f++) {
    let e = 0;
    for (let i = off + f * FRAME; i < off + (f + 1) * FRAME; i++) e += x[i] * x[i];
    db[f] = 10 * Math.log10(e / FRAME + 1e-12);
  }
  return db;
}

// numpy's default (linear) percentile
export function percentile(values, q) {
  if (!values.length) return -240;
  const s = Array.from(values).sort((a, b) => a - b);
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo);
}

// asr.speech_in_tail's run rule over Silero probabilities: >= 7 frames >= 0.35 peaking >= 0.5.
export function speechRun(probs) {
  let run = 0;
  let peak = 0;
  let best = 0;
  let bestPeak = 0;
  for (const p of probs) {
    if (p >= 0.35) {
      run += 1;
      peak = Math.max(peak, p);
      if (run > best) [best, bestPeak] = [run, peak];
      if (run >= 7 && peak >= 0.5) return { pass: true, run, peak };
    } else {
      run = 0;
      peak = 0;
    }
  }
  return { pass: false, run: best, peak: bestPeak };
}

// Silero over the last tailS (legacy speechInTail), with its evidence.
export async function speechInTail(audio, vadProbs, tailS = 1.5) {
  const n = Math.min(audio.length, Math.round(tailS * SR));
  const tail = audio.subarray(audio.length - n);
  const level = rmsDb(tail);
  if (!n || level < SILENCE_DBFS) return { pass: false, level, floor: true, run: 0, peak: 0 };
  const padded = new Float32Array(Math.ceil(n / FRAME) * FRAME);
  padded.set(tail);
  const r = speechRun(await vadProbs(padded));
  return { ...r, level, floor: false };
}

// Energy evidence in the last tailS: the longest run of frames at >= floor + 6 dB.
export function energyInTail(audio, tailS = 1.5) {
  const db = frameDb(audio);
  const floor = Math.max(-70, percentile(db, 10));
  const k = Math.min(db.length, Math.floor((tailS * SR) / FRAME));
  let run = 0;
  let best = 0;
  for (let i = db.length - k; i < db.length; i++) {
    run = db[i] >= floor + 6 ? run + 1 : 0;
    best = Math.max(best, run);
  }
  return { pass: best >= 7, run: best, floor };
}

// The gate decision for one window. `reason` is null when Whisper runs, else
// "below_floor" (tail under -45 dBFS) or "vad_reject" (sound, but no speech evidence).
export async function decide(audio, policy, vadProbs) {
  if (!POLICIES.includes(policy)) throw new Error(`unknown gate policy ${policy}`);
  const n = Math.min(audio.length, Math.round(1.5 * SR));
  const level = rmsDb(audio.subarray(audio.length - n));
  const out = { policy, level: Math.round(level * 10) / 10, run: true, reason: null, vad: null, energy: null };
  if (!n || level < SILENCE_DBFS) return { ...out, run: false, reason: "below_floor" };
  if (policy === "ungated") return out;
  const vad = await speechInTail(audio, vadProbs);
  out.vad = { pass: vad.pass, run: vad.run, peak: Math.round(vad.peak * 1000) / 1000 };
  if (vad.pass) return out;
  if (policy === "energy_assisted") {
    const e = energyInTail(audio);
    out.energy = { pass: e.pass, run: e.run, floor: Math.round(e.floor * 10) / 10 };
    if (e.pass) return { ...out, via: "energy" };
  }
  return { ...out, run: false, reason: "vad_reject" };
}

// -- The stop detector (asr.quiet_at_end, asr.QuietMeter, asr.LiveQuiet) -------------------
// A 32 ms frame of the window's last 3 s is sound if Silero says speech (>= 0.35), or it is
// 6 dB over the window's floor AND within QUIET_REL_DB of the reciter's voice (the median
// frame Silero is sure of, from at least VOICE_MIN_FRAMES, remembered from window to window
// by the caller); sound = a run of >= 3 frames. A phone's gain control turns the room up
// by 10-15 dB once the reciter stops, and against the floor alone that hiss read as someone
// still reciting (docs/results/stops.md). relDb null = the floor alone, as before.
export const QUIET_REL_DB = 15;
export const VOICE_MIN_FRAMES = 16;
export const LIVE_REL_DB = 15;
export const MIN_PAUSE = 0.3; // TrackerConfig.still_motion_after

// The reciter's voice in dBFS (asr.voice_level), or null.
export function voiceLevel(probs, db, minFrames = VOICE_MIN_FRAMES) {
  const voiced = [];
  for (let i = 0; i < probs.length; i++) if (probs[i] >= 0.5) voiced.push(db[i]);
  if (voiced.length < Math.max(1, minFrames)) return null;
  voiced.sort((x, y) => x - y);
  const m = voiced.length >> 1;
  return voiced.length % 2 ? voiced[m] : (voiced[m - 1] + voiced[m]) / 2;
}

// Frame spans [start, end) of sound (asr._sound_runs).
function soundRuns(probs, db, floor, relDb, voiceDb) {
  let bar = Math.max(floor, -70) + 6; // never below -70 dBFS: noise suppression outputs exact zeros
  if (relDb != null) {
    const voice = voiceDb ?? voiceLevel(probs, db);
    bar = voice != null ? Math.max(bar, voice - relDb) : Infinity;
  }
  const runs = [];
  let start = -1;
  for (let i = 0; i <= probs.length; i++) {
    const active = i < probs.length && (probs[i] >= 0.35 || db[i] >= bar);
    if (active && start < 0) start = i;
    else if (!active && start >= 0) {
      if (i - start >= 3) runs.push([start, i]);
      start = -1;
    }
  }
  return runs;
}

// Seconds of the last `seconds` of n frames spent in silences of at least minPause (asr._paused).
function pausedIn(runs, n, seconds, minPause) {
  const step = FRAME / SR;
  const lo = Math.max(0, n - Math.round(seconds / step));
  const edges = [0, ...runs.flat(), n];
  let out = 0;
  for (let i = 0; i < edges.length; i += 2) {
    const [a, b] = [edges[i], edges[i + 1]];
    if ((b - a) * step >= minPause) out += Math.max(0, b - Math.max(a, lo));
  }
  return out * step;
}

// One window's stop measures (asr.QuietMeter called on it): seconds since the reciter last
// made a sound, the seconds of the last dt spent in pauses (null without dt), and the voice
// level to remember (this window's if it had enough speech, else the one given).
export async function stopMeasures(audio, vadProbs, { voiceDb = null, dt = null, relDb = QUIET_REL_DB,
  minPause = MIN_PAUSE, tailS = 3.0 } = {}) {
  const x = audio.subarray(audio.length % FRAME);
  const nTail = Math.min(x.length, Math.floor((tailS * SR) / FRAME) * FRAME);
  if (!nTail || rmsDb(x.subarray(x.length - nTail)) < SILENCE_DBFS) { // below the floor
    return { quiet: tailS, paused: dt, voiceDb };
  }
  const all = frameDb(x);
  const floor = percentile(all, 10);
  const probs = await vadProbs(x.slice(x.length - nTail), true);
  const k = probs.length;
  const db = all.subarray(all.length - k);
  const v = voiceLevel(probs, db);
  const voice = v ?? voiceDb;
  const runs = soundRuns(probs, db, floor, relDb, voice);
  const quiet = (runs.length ? k - runs[runs.length - 1][1] : k) * (FRAME / SR);
  const paused = dt == null ? null : Math.min(dt, pausedIn(runs, k, dt, minPause));
  return { quiet, paused, voiceDb: voice };
}

// Seconds since the reciter last made a sound (asr.quiet_at_end, stateless: the window's own voice).
export async function quietAtEnd(audio, vadProbs, tailS = 3.0, relDb = QUIET_REL_DB) {
  return (await stopMeasures(audio, vadProbs, { tailS, relDb })).quiet;
}

// Seconds the reciter has been silent, on the audio the page holds right now (asr.LiveQuiet):
// the window's quiet is a second or two old by the time its update is shown. Loudness alone
// (Silero is busy in the worker): a frame is sound if it comes within relDb of the voice level
// the worker last measured; sound = a run of >= 3 frames. null until a voice level is known.
export class LiveQuiet {
  constructor(relDb = LIVE_REL_DB) {
    this.relDb = relDb;
    this.rest = new Float32Array(0);
    this.voiceDb = null;
    this.n = 0; // frames heard
    this.run = 0; // frames in the current run of sound
    this.lastEnd = 0; // frames heard when a run of 3+ was last sounding
  }

  push(chunk) {
    const x = new Float32Array(this.rest.length + chunk.length);
    x.set(this.rest);
    x.set(chunk, this.rest.length);
    const k = Math.floor(x.length / FRAME);
    for (let f = 0; f < k; f++) {
      let e = 0;
      for (let i = f * FRAME; i < (f + 1) * FRAME; i++) e += x[i] * x[i];
      const d = 10 * Math.log10(e / FRAME + 1e-12);
      this.n += 1;
      const sound = this.voiceDb != null && d >= this.voiceDb - this.relDb;
      this.run = sound ? this.run + 1 : 0;
      if (this.run >= 3) this.lastEnd = this.n;
    }
    this.rest = x.slice(k * FRAME);
  }

  get quiet() {
    return this.voiceDb == null ? null : ((this.n - this.lastEnd) * FRAME) / SR;
  }
}

// asr._looks_hallucinated, for logging (and ?filter=1). The Python version uses zlib's
// compression ratio; here deflate via CompressionStream, so ratios can differ slightly.
const HALLUCINATIONS = ["اشترك", "ترجمة", "نانسي", "المترجم", "للمشاهدة", "موسيقى"];
const HALLUCINATED_WHOLE = new Set(["شكرا", "شكرا لكم", "شكرا جزيلا", "شكرا لك"]);

export async function looksHallucinated(text) {
  const bare = text.replace(/^[ .!؟?،,]+|[ .!؟?،,]+$/g, "");
  if (HALLUCINATED_WHOLE.has(bare) || HALLUCINATIONS.some((h) => text.includes(h))) return true;
  const raw = new TextEncoder().encode(text);
  if (raw.length <= 40 || typeof CompressionStream === "undefined") return false;
  const stream = new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate"));
  const packed = await new Response(stream).arrayBuffer();
  return raw.length / packed.byteLength > 2.4;
}
