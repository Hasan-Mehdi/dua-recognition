// Replay a recording through the real page, headless: the device engine (Whisper in the
// worker, web/gate.js), tracker.js, the gliding highlight and the stop detector, as a
// phone runs them but on this machine's CPU. The audio goes in through the page's own
// "Choose a file" input and plays in real time; the page's debug session comes back out
// as a .wav with its log (web/session-log.js), for scripts/session_report.py and
// scripts/session_stops.py.
//
//   node scripts/page_replay.mjs <in.wav> <out.wav> [query, e.g. "model=whisper-base-aug-v4-ctx8"] [--asr-ms N] [--ctc-ms N]
//
// --asr-ms N makes every transcription take at least N ms (the worker is served with a wait
// before it answers), to come near a phone: Whisper takes ~0.25 s per update on a desktop and
// 1-2.5 s on Hasan's Android. Chrome's CPU throttling doesn't reach the worker.
//
// A recorded session (data/sessions/*.wav) replays what the phone heard, hiss and all.
// puppeteer-core lives in data/cache/reliability/node, as for scripts/gate_bench.mjs.
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "data/cache/reliability/node/package.json"));
const puppeteer = require("puppeteer-core");

const argv = process.argv.slice(2);
const at = argv.indexOf("--asr-ms");
const asrMs = at >= 0 ? Number(argv.splice(at, 2)[1]) : 0;
// --ctc-ms N: the same for the word follower's CTC model (web/ctc-worker.js, ?words=ctc).
const ct = argv.indexOf("--ctc-ms");
const ctcMs = ct >= 0 ? Number(argv.splice(ct, 2)[1]) : 0;
const [input, output, query = ""] = argv;
if (!input || !output) {
  console.error("usage: node scripts/page_replay.mjs <in.wav> <out.wav> [query]");
  process.exit(2);
}

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".onnx": "application/octet-stream", ".css": "text/css", ".txt": "text/plain", ".svg": "image/svg+xml",
  ".wav": "audio/wav", ".woff2": "font/woff2", ".png": "image/png", ".webmanifest": "application/manifest+json" };
const WEB = join(ROOT, "web");
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const f = join(WEB, path.endsWith("/") ? path + "index.html" : path);
  if (!f.startsWith(WEB) || !existsSync(f) || statSync(f).isDirectory()) {
    res.statusCode = 404;
    return res.end();
  }
  res.setHeader("Content-Type", TYPES[extname(f)] || "application/octet-stream");
  if (asrMs && path === "/asr-worker.js") {
    const src = readFileSync(f, "utf8");
    const reply = 'self.postMessage({ type: "text"';
    if (!src.includes(reply)) throw new Error("asr-worker.js changed: update page_replay.mjs --asr-ms");
    return res.end(src.replace(reply, `await new Promise((r) => setTimeout(r, ${asrMs} - (performance.now() - t0)));
    ${reply}`));
  }
  if (ctcMs && path === "/ctc-worker.js") {
    const src = readFileSync(f, "utf8");
    const reply = 'self.postMessage({ type: "frames"';
    if (!src.includes(reply)) throw new Error("ctc-worker.js changed: update page_replay.mjs --ctc-ms");
    return res.end(src.replace(reply, `await new Promise((r) => setTimeout(r, ${ctcMs} - (performance.now() - t0)));
      ${reply}`));
  }
  res.end(readFileSync(f));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const port = server.address().port;

const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("response", (r) => r.status() >= 400 && !r.url().endsWith("/api/mode") // (no server: device mode)
  && errors.push(`${r.status()} ${r.url()}`));
page.on("console", (m) => m.type() === "error" && !m.text().includes("404") && errors.push(m.text()));
await page.goto(`http://127.0.0.1:${port}/${query ? `?${query}` : ""}`);
await page.waitForFunction(() => document.body.dataset.state === "idle", { timeout: 60_000 });
const before = await page.evaluate(async () => (await (await import("/session-log.js")).listSessions()).length);

const t0 = Date.now();
await (await page.$("#file")).uploadFile(resolve(input));
// Listening, then back to idle once the file has played to its end.
await page.waitForFunction(() => ["listening", "following"].includes(document.body.dataset.state),
  { timeout: 300_000, polling: 200 });
await page.waitForFunction(() => document.body.dataset.state === "idle", { timeout: 3_600_000, polling: 500 });
await page.waitForFunction(async (n) => {
  const list = await (await import("/session-log.js")).listSessions();
  return list.length > n && list.at(-1).ended;
}, { timeout: 60_000, polling: 500 }, before);

const b64 = await page.evaluate(async () => {
  const m = await import("/session-log.js");
  const rec = (await m.listSessions()).at(-1);
  const bytes = new Uint8Array(await (await m.sessionFile(rec.id)).arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
});
writeFileSync(output, Buffer.from(b64, "base64"));
console.log(`${output}: ${((Date.now() - t0) / 1000).toFixed(0)} s` + (errors.length ? `, ${errors.length} page errors:` : ""));
for (const e of errors) console.error("  ", e);
await browser.close();
server.close();
