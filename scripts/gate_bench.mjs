// Desktop-browser benchmark of the gate policies through the page's real ASR worker
// (web/dev/gate-bench.html): headless Chrome, the phone checkpoint (ONNX q8, WASM), a
// development clip streamed in real time. Desktop numbers are PROVISIONAL for phones.
//
//   node scripts/gate_bench.mjs <clip.f32> <start s> <seconds> <out.json> [policy[@model]...]
//
// Serves web/ on a local port (plus the clip at /bench-audio.f32) and runs one page per
// policy, one after another. puppeteer-core lives in data/cache/reliability/node.
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "data/cache/reliability/node/package.json"));
const puppeteer = require("puppeteer-core");

const [clip, startS, seconds, outFile, ...pols] = process.argv.slice(2);
const policies = pols.length ? pols : ["legacy", "energy_assisted", "ungated"];
const SR = 16000;
const all = readFileSync(clip);
const a = Math.round(Number(startS) * SR) * 4;
const audio = all.subarray(a, a + Math.round(Number(seconds) * SR) * 4);

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".onnx": "application/octet-stream", ".css": "text/css", ".txt": "text/plain" };
const WEB = join(ROOT, "web");
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path === "/bench-audio.f32") return res.end(audio);
  const f = join(WEB, path.endsWith("/") ? path + "index.html" : path);
  if (!f.startsWith(WEB) || !existsSync(f) || statSync(f).isDirectory()) {
    res.statusCode = 404;
    return res.end();
  }
  res.setHeader("Content-Type", TYPES[extname(f)] || "application/octet-stream");
  res.end(readFileSync(f));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const port = server.address().port;

const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const results = { clip, start_s: Number(startS), seconds: Number(seconds), chrome: await browser.version(), runs: [] };
for (const run of policies) {
  // "policy" or "policy@model" (a model under web/models/, e.g. legacy@whisper-base-aug-v4-ctx8)
  const [gate, model] = run.split("@");
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && console.error(`[${run}]`, m.text()));
  await page.goto(`http://127.0.0.1:${port}/dev/gate-bench.html?gate=${gate}&seconds=${seconds}${model ? `&model=${model}` : ""}`);
  await page.waitForFunction("window.__result", { timeout: (Number(seconds) + 600) * 1000, polling: 1000 });
  const r = await page.evaluate("window.__result");
  results.runs.push({ ...r, model: model || "whisper-base-aug-v4" });
  console.log(run, r.error ?? `${r.hops.length} hops, load ${Math.round(r.load_ms)} ms`);
  await page.close();
}
await browser.close();
server.close();
writeFileSync(outFile, JSON.stringify(results, null, 1));
