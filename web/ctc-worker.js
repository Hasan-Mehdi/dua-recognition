// The phone CTC model in its own Web Worker (src/dua_recognition/ctc_student.py): the latest
// few seconds of audio in, letter log posteriors out, one row per 20 ms, for the word follower
// (follower.js). It runs while Whisper (asr-worker.js) runs in the other worker.
import { WhisperFeatureExtractor } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

let fe = null;
let session = null;
let meta = null;

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try {
      if (data.threads) ort.env.wasm.numThreads = data.threads;
      const base = new URL(`./models/${data.model}/`, self.location.href).href;
      meta = await (await fetch(base + "meta.json")).json();
      fe = new WhisperFeatureExtractor(await (await fetch(base + "preprocessor_config.json")).json());
      session = await ort.InferenceSession.create(base + (data.file || "model_q8.onnx"));
      // One run on silence: the first inference is the slow one (kernels compile), so it happens now.
      const { input_features: f } = await fe(new Float32Array(Math.round(meta.window_s * 16000)));
      await session.run({ input_features: new ort.Tensor("float32", f.data, f.dims) });
      self.postMessage({ type: "ready", meta });
    } catch (e) {
      self.postMessage({ type: "error", message: String(e) });
    }
    return;
  }
  if (data.type === "frames") {
    const t0 = performance.now();
    try {
      const { input_features: f } = await fe(data.audio);
      const out = await session.run({ input_features: new ort.Tensor("float32", f.data, f.dims) });
      const lp = out.logp;
      const frames = new Float32Array(lp.data);
      self.postMessage({ type: "frames", id: data.id, frames, T: lp.dims[1], C: lp.dims[2],
        ms: performance.now() - t0 }, [frames.buffer]);
    } catch (e) {
      self.postMessage({ type: "error", message: String(e), id: data.id });
    }
  }
};
