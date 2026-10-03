// Time ONNX models in headless Chrome the way the page runs them (onnxruntime-web, WASM,
// cross-origin isolated so it may use threads): the phone CTC model against the encoder of
// the Whisper model the phone already runs, whose time on Hasan's Android is known from
// his sessions. The ratio says what the CTC model would cost on the phone.
//
//   node scripts/ctc_bench.mjs [threads=4] models/ctc-speedtest-base/model_q8.onnx:80x300 ...
//
// Each argument is a model under web/ and its input shape (input_features [1, A, B]).
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "data/cache/reliability/node/package.json"));
const puppeteer = require("puppeteer-core");

let argv = process.argv.slice(2);
let threads = 4;
if (argv[0]?.startsWith("threads=")) threads = Number(argv.shift().slice(8));
const models = argv.map((a) => {
  const [path, shape] = a.split(":");
  return { path, shape: shape.split("x").map(Number) };
});

const WEB = join(ROOT, "web");
const PAGE = `<!doctype html><script type="module">
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = ${threads};
window.bench = async (path, shape, runs) => {
  const s = await ort.InferenceSession.create(path);
  const n = shape.reduce((a, b) => a * b, 1);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(i * 0.01) * 0.5;
  const feeds = { [s.inputNames[0]]: new ort.Tensor("float32", x, [1, ...shape]) };
  for (let i = 0; i < 3; i++) await s.run(feeds);
  const ts = [];
  for (let i = 0; i < runs; i++) { const t = performance.now(); await s.run(feeds); ts.push(performance.now() - t); }
  ts.sort((a, b) => a - b);
  return { p50: ts[runs >> 1], p90: ts[Math.floor(runs * 0.9)], iso: crossOriginIsolated };
};
window.ready = true;
</script>`;
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
  if (path === "/bench.html") {
    res.setHeader("Content-Type", "text/html");
    return res.end(PAGE);
  }
  const f = join(WEB, path);
  if (!f.startsWith(WEB) || !existsSync(f) || statSync(f).isDirectory()) {
    res.statusCode = 404;
    return res.end();
  }
  res.setHeader("Content-Type", extname(f) === ".js" ? "text/javascript" : "application/octet-stream");
  res.end(readFileSync(f));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: true,
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("page:", String(e)));
await page.goto(`http://127.0.0.1:${server.address().port}/bench.html`);
await page.waitForFunction(() => window.ready, { timeout: 60_000 });
for (const m of models) {
  const r = await page.evaluate((p, s) => window.bench(p, s, 20), "/" + m.path, m.shape);
  console.log(`${m.path} [${m.shape}]  p50 ${r.p50.toFixed(0)} ms  p90 ${r.p90.toFixed(0)} ms  (isolated ${r.iso}, ${threads} threads)`);
}
await browser.close();
server.close();
