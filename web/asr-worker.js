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

import { decide, looksHallucinated, makeVadProbs, stopMeasures } from "./gate.js";

env.localModelPath = new URL("./models/", self.location.href).href;
env.allowLocalModels = true; // off by default in browsers
env.allowRemoteModels = false;

// -- VAD: Silero v6 (MIT), the same model and rule as asr.speech_in_tail ------
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
let vad = null;

// Silero probabilities, framed like faster-whisper (gate.js makeVadProbs).
const vadProbs = makeVadProbs((feeds) => vad.run(feeds), ort.Tensor);

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
    // Times as epoch ms (timeOrigin + now): the worker's and the page's performance.now()
    // have different origins, so only these are comparable across the two.
    const clock = () => performance.timeOrigin + performance.now();
    const recv = clock();
    const t0 = performance.now();
    const policy = data.gate || "legacy";
    let text = "";
    let quiet = 0;
    let stops = { paused: null, voiceDb: data.voiceDb ?? null };
    let gate = null;
    let inferMs = null;
    let hallucinated = false;
    let failed = false;
    try {
      // How long they've been silent, how much of the last dt went on pauses, and the voice level
      // to remember (the page hands the last one back each time): gate.js stopMeasures.
      stops = await stopMeasures(data.audio, vadProbs, { voiceDb: data.voiceDb ?? null, dt: data.dt ?? null });
      quiet = stops.quiet;
      gate = await decide(data.audio, policy, vadProbs);
      // No speech evidence in the newest audio: report a pause and skip Whisper entirely.
      if (gate.run) {
        const t1 = performance.now();
        const inputs = await processor(data.audio);
        const ids = await model.generate({ ...inputs, language: "arabic", task: "transcribe", max_new_tokens: 96 });
        // Belt and braces: the fine-tune's tokenizer files don't flag every
        // <|...|> control token as special for the browser tokenizer.
        text = tokenizer.batch_decode(ids, { skip_special_tokens: true })[0].replace(/<\|[^|]*\|>/g, "").trim();
        inferMs = performance.now() - t1;
        hallucinated = await looksHallucinated(text);
        if (hallucinated && data.filter) text = ""; // ?filter=1 only: a separate ablation from the gate
      }
    } catch (e) {
      failed = true;
      self.postMessage({ type: "error", message: String(e) });
    }
    // skip: why there is no text. null = Whisper ran and wrote something; "empty" = it ran
    // and wrote nothing; "hallucination" = filtered (?filter=1); "error" = inference failed.
    const skip = failed ? "error" : gate && !gate.run ? gate.reason
      : hallucinated && data.filter ? "hallucination" : text ? null : "empty";
    self.postMessage({ type: "text", id: data.id, text, quiet, paused: stops.paused, voiceDb: stops.voiceDb,
      ms: performance.now() - t0, gate, skip,
      hallucinated, infer_ms: inferMs, recv, done: clock() });
  }
};
