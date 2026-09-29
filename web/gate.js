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

// Seconds since the reciter last made a sound (asr.quiet_at_end): a frame is sound if
// Silero says speech (>= 0.35) or it's 6 dB over the window's floor; sound = >= 3 frames.
export async function quietAtEnd(audio, vadProbs, tailS = 3.0) {
  const x = audio.subarray(audio.length % FRAME);
  const nTail = Math.min(x.length, Math.floor((tailS * SR) / FRAME) * FRAME);
  if (!nTail) return tailS;
  const db = frameDb(x);
  if (rmsDb(x.subarray(x.length - nTail)) < SILENCE_DBFS) return tailS; // below the floor
  // Never below -70 dBFS: noise suppression outputs exact zeros between words (see asr.py).
  const floor = Math.max(-70, percentile(db, 10));
  const probs = await vadProbs(x.slice(x.length - nTail), true);
  const k = probs.length;
  let lastEnd = -1;
  let start = -1;
  for (let i = 0; i <= k; i++) {
    const active = i < k && (probs[i] >= 0.35 || db[db.length - k + i] >= floor + 6);
    if (active && start < 0) start = i;
    else if (!active && start >= 0) {
      if (i - start >= 3) lastEnd = i;
      start = -1;
    }
  }
  return (lastEnd < 0 ? k : k - lastEnd) * (FRAME / SR);
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
