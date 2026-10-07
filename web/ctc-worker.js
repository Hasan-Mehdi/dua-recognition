// The phone CTC model in its own Web Worker (src/dua_recognition/ctc_student.py): the latest
// few seconds of audio in, letter log posteriors out, one row per 20 ms, for the word follower
// (follower.js). It runs while Whisper (asr-worker.js) runs in the other worker.
//
// A streaming model (meta.arch "stream", src/dua_recognition/stream_ctc.py) keeps its caches
// between steps and encodes only the audio that is new since the last one (ctc-stream.js); the
// page still sends the latest window and gets back the latest 2 s of frames.
import { WhisperFeatureExtractor } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

import { CtcStream } from "./ctc-stream.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

let fe = null;
let session = null;
let meta = null;
let stream = null; // a streaming model's state (CtcStream)

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try {
      if (data.threads) ort.env.wasm.numThreads = data.threads;
      const base = new URL(`./models/${data.model}/`, self.location.href).href;
      meta = await (await fetch(base + "meta.json")).json();
      session = await ort.InferenceSession.create(base + (data.file || "model_q8.onnx"));
      // One run on silence: the first inference is the slow one (kernels compile), so it happens now.
      const n = Math.round(meta.window_s * 16000);
      if (meta.arch === "stream") {
        stream = new CtcStream(ort, session, meta);
        await stream.step(new Float32Array(n), n);
        stream.reset();
      } else {
        fe = new WhisperFeatureExtractor(await (await fetch(base + "preprocessor_config.json")).json());
        const { input_features: f } = await fe(new Float32Array(n));
        await session.run({ input_features: new ort.Tensor("float32", f.data, f.dims) });
      }
      self.postMessage({ type: "ready", meta });
    } catch (e) {
      self.postMessage({ type: "error", message: String(e) });
    }
    return;
  }
  if (data.type === "frames") {
    const t0 = performance.now();
    try {
      let out;
      if (stream) out = await stream.step(data.audio, data.id);
      else {
        const { input_features: f } = await fe(data.audio);
        const lp = (await session.run({ input_features: new ort.Tensor("float32", f.data, f.dims) })).logp;
        out = { frames: new Float32Array(lp.data), T: lp.dims[1], C: lp.dims[2] };
      }
      const { frames, T, C } = out;
      self.postMessage({ type: "frames", id: data.id, frames, T, C, ms: performance.now() - t0 }, [frames.buffer]);
    } catch (e) {
      self.postMessage({ type: "error", message: String(e), id: data.id });
    }
  }
};
