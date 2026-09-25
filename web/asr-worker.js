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

async function speechInTail(audio, tailS = 1.5) {
  const n = Math.min(audio.length, Math.round(tailS * 16000));
  let tail = audio.subarray(audio.length - n);
  let energy = 0;
  for (const x of tail) energy += x * x;
  if (!n || 10 * Math.log10(energy / n + 1e-24) < -45) return false; // digital silence
  const frames = Math.ceil(n / 512);
  const padded = new Float32Array(frames * 512);
  padded.set(tail);
  // Each frame is [64 samples of the previous frame | 512 samples], like faster-whisper.
  const input = new Float32Array(frames * 576);
  for (let f = 0; f < frames; f++) {
    if (f > 0) input.set(padded.subarray(f * 512 - 64, f * 512), f * 576);
    input.set(padded.subarray(f * 512, (f + 1) * 512), f * 576 + 64);
  }
  const zeros = () => new ort.Tensor("float32", new Float32Array(128), [1, 1, 128]);
  const out = await vad.run({ input: new ort.Tensor("float32", input, [frames, 576]), h: zeros(), c: zeros() });
  const probs = out.speech_probs.data;
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
    try {
      // No speech in the newest audio: report a pause and skip Whisper entirely.
      if (!(await speechInTail(data.audio))) {
        self.postMessage({ type: "text", id: data.id, text: "", ms: performance.now() - t0 });
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
    self.postMessage({ type: "text", id: data.id, text, ms: performance.now() - t0 });
  }
};
