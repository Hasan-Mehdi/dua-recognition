// Whisper in a Web Worker, so decoding never blocks the page.
// Processor, tokenizer and model are loaded explicitly: the browser build's
// ASR pipeline skipped loading the processor for local models.
import {
  AutoProcessor,
  AutoTokenizer,
  WhisperForConditionalGeneration,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

env.localModelPath = new URL("./models/", self.location.href).href;
env.allowLocalModels = true; // off by default in browsers
env.allowRemoteModels = false;

// -- VAD: Silero v6 (MIT), the same model and rule as asr.speech_in_tail ------
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
let vad = null;

// Silero probabilities for whole 512-sample frames, fresh state, framed like faster-whisper:
// each frame is [64 samples of the previous frame | 512 samples], and (a quirk of
// faster-whisper's, mirrored for parity) the last 64 samples of the last frame are zeroed.
async function vadProbs(padded, zeroLast = false) {
  const frames = padded.length / 512;
  const input = new Float32Array(frames * 576);
  for (let f = 0; f < frames; f++) {
    if (f > 0) input.set(padded.subarray(f * 512 - 64, f * 512), f * 576);
    input.set(padded.subarray(f * 512, (f + 1) * 512), f * 576 + 64);
  }
  if (zeroLast) input.fill(0, frames * 576 - 64);
  const zeros = () => new ort.Tensor("float32", new Float32Array(128), [1, 1, 128]);
  const out = await vad.run({ input: new ort.Tensor("float32", input, [frames, 576]), h: zeros(), c: zeros() });
  return out.speech_probs.data;
}

async function speechInTail(audio, tailS = 1.5) {
  const n = Math.min(audio.length, Math.round(tailS * 16000));
  let tail = audio.subarray(audio.length - n);
  let energy = 0;
  for (const x of tail) energy += x * x;
  if (!n || 10 * Math.log10(energy / n + 1e-24) < -45) return false; // digital silence
  const frames = Math.ceil(n / 512);
  const padded = new Float32Array(frames * 512);
  padded.set(tail);
  const probs = await vadProbs(padded);
  let run = 0;
  let peak = 0;
  for (const p of probs) {
    if (p >= 0.35) {
      run += 1;
      peak = Math.max(peak, p);
      if (run >= 7 && peak >= 0.5) return true;
    } else {
      run = 0;
      peak = 0;
    }
  }
  return false;
}

// Seconds since the reciter last made a sound, as asr.quiet_at_end: a frame
// is sound if Silero says speech (>= 0.35) or it's 6 dB over the window's
// floor (10th-percentile frame energy, at least -70 dBFS); sound = a run of >= 3 frames.
async function quietAtEnd(audio, tailS = 3.0) {
  const x = audio.subarray(audio.length % 512);
  const nTail = Math.min(x.length, Math.floor((tailS * 16000) / 512) * 512);
  if (!nTail) return tailS;
  const nFrames = x.length / 512;
  const db = new Float64Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    let e = 0;
    for (let i = f * 512; i < (f + 1) * 512; i++) e += x[i] * x[i];
    db[f] = 10 * Math.log10(e / 512 + 1e-12);
  }
  let tailEnergy = 0;
  for (let i = x.length - nTail; i < x.length; i++) tailEnergy += x[i] * x[i];
  if (20 * Math.log10(Math.sqrt(tailEnergy / nTail) + 1e-12) < -45) return tailS; // digital silence
  const sorted = Array.from(db).sort((a, b) => a - b);
  const pos = 0.1 * (sorted.length - 1); // numpy's default (linear) percentile
  const lo = Math.floor(pos);
  // Never below -70 dBFS: noise suppression outputs exact zeros between words (see asr.py).
  const floor = Math.max(-70, sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo));
  const probs = await vadProbs(x.slice(x.length - nTail), true);
  const k = probs.length;
  let lastEnd = -1;
  let start = -1;
  for (let i = 0; i <= k; i++) {
    const active = i < k && (probs[i] >= 0.35 || db[nFrames - k + i] >= floor + 6);
    if (active && start < 0) start = i;
    else if (!active && start >= 0) {
      if (i - start >= 3) lastEnd = i;
      start = -1;
    }
  }
  return (lastEnd < 0 ? k : k - lastEnd) * (512 / 16000);
}

let processor = null;
let tokenizer = null;
let model = null;

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try {
      const progress_callback = (p) =>
        p.status === "progress" && self.postMessage({ type: "progress", file: p.file, progress: p.progress });
      // Sequentially: loading these concurrently stalled in Chrome.
      vad = await ort.InferenceSession.create(new URL("./vad/silero_vad_v6.onnx", self.location.href).href);
      self.postMessage({ type: "stage", text: "Loading the audio processor…" });
      processor = await AutoProcessor.from_pretrained(data.model);
      self.postMessage({ type: "stage", text: "Loading the tokenizer…" });
      tokenizer = await AutoTokenizer.from_pretrained(data.model);
      self.postMessage({ type: "stage", text: "Downloading the speech model (about 100 MB, once)…" });
      model = await WhisperForConditionalGeneration.from_pretrained(data.model, {
        dtype: "q8",
        device: data.webgpu ? "webgpu" : "wasm",
        progress_callback,
      });
      self.postMessage({ type: "ready" });
    } catch (e) {
      self.postMessage({ type: "error", message: String(e) });
    }
    return;
  }
  if (data.type === "transcribe") {
    const t0 = performance.now();
    let text = "";
    let quiet = 0;
    try {
      quiet = await quietAtEnd(data.audio);
      // No speech in the newest audio: report a pause and skip Whisper entirely.
      if (!(await speechInTail(data.audio))) {
        self.postMessage({ type: "text", id: data.id, text: "", quiet, ms: performance.now() - t0 });
        return;
      }
      const inputs = await processor(data.audio);
      const ids = await model.generate({ ...inputs, language: "arabic", task: "transcribe", max_new_tokens: 96 });
      // Belt and braces: the fine-tune's tokenizer files don't flag every
      // <|...|> control token as special for the browser tokenizer.
      text = tokenizer.batch_decode(ids, { skip_special_tokens: true })[0].replace(/<\|[^|]*\|>/g, "").trim();
    } catch (e) {
      self.postMessage({ type: "error", message: String(e) });
    }
    self.postMessage({ type: "text", id: data.id, text, quiet, ms: performance.now() - t0 });
  }
};
