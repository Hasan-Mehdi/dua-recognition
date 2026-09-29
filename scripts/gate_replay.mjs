// Replay recordings through the browser's speech gate (web/gate.js, the code the page runs)
// with onnxruntime-node and the same Silero v6 model (web/vad/silero_vad_v6.onnx).
// Hops as the device engine sends them: every 1 s of new audio, the latest <= 6 s.
// (The page skips a hop while Whisper is busy; that depends on device speed and is
// measured separately in the browser, not here.)
//
//   node scripts/gate_replay.mjs <jobs.json> <out dir>
//
// jobs.json: [{"key": "...", "pcm": "path/to/16k-float32.raw", "gain_db": 0}, ...] (scripts/gate_eval.py prepare).
// Writes <out dir>/<key>.jsonl: per hop {t, quiet, legacy, energy_assisted, ungated}.
// onnxruntime-node lives in data/cache/reliability/node (npm i onnxruntime-node).
import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "data/cache/reliability/node/package.json"));
const ort = require("onnxruntime-node");
const gate = await import(pathToFileURL(join(ROOT, "web/gate.js")).href);

const [jobsFile, outDir] = process.argv.slice(2);
const jobs = JSON.parse(readFileSync(jobsFile, "utf8"));
mkdirSync(outDir, { recursive: true });
const session = await ort.InferenceSession.create(join(ROOT, "web/vad/silero_vad_v6.onnx"),
  { intraOpNumThreads: 1, interOpNumThreads: 1 });
const vadProbs = gate.makeVadProbs((feeds) => session.run(feeds), ort.Tensor);
const W = 6 * gate.SR;

const slim = (d) => ({ run: d.run, reason: d.reason, level: d.level, via: d.via ?? null,
  vad_run: d.vad?.run ?? null, vad_peak: d.vad?.peak ?? null, energy_run: d.energy?.run ?? null });

for (const job of jobs) {
  const out = join(outDir, `${job.key}.jsonl`);
  if (existsSync(out)) continue;
  const buf = readFileSync(job.pcm);
  const y = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const gain = 10 ** ((job.gain_db ?? 0) / 20); // lower-level copies of the same audio
  if (gain !== 1) for (let i = 0; i < y.length; i++) y[i] *= gain;
  const lines = [];
  const t0 = Date.now();
  for (let end = gate.SR; end <= y.length; end += gate.SR) {
    const audio = y.slice(Math.max(0, end - W), end);
    const row = { t: end / gate.SR, quiet: Math.round((await gate.quietAtEnd(audio, vadProbs)) * 1000) / 1000 };
    for (const p of gate.POLICIES) row[p] = slim(await gate.decide(audio, p, vadProbs));
    lines.push(JSON.stringify(row));
  }
  writeFileSync(out, lines.join("\n"));
  console.log(`${job.key}: ${lines.length} hops in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
