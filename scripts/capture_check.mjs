// Does the page hear one second of audio per second? Opens the real page headless in Chrome or
// Firefox with a fake microphone, taps Start, listens for a while, stops, and compares the
// session's audio length (web/session-log.js) with the wall-clock time it took. A ratio of
// 2.00 means every chunk reaches the engine twice (2026-10-02: three Firefox sessions).
//
//   node scripts/capture_check.mjs [chrome|firefox] [seconds=20] [--double]
//
// --double taps Start twice, 300 ms apart (a second begin() while the first one awaits).
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "data/cache/reliability/node/package.json"));
const puppeteer = require("puppeteer-core");

const argv = process.argv.slice(2);
const double = argv.includes("--double");
const [browserName = "chrome", secs = "20"] = argv.filter((a) => !a.startsWith("--"));

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".onnx": "application/octet-stream", ".css": "text/css", ".svg": "image/svg+xml", ".wav": "audio/wav",
  ".woff2": "font/woff2", ".png": "image/png", ".webmanifest": "application/manifest+json" };
const WEB = join(ROOT, "web");
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
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

const firefox = browserName === "firefox";
const browser = await puppeteer.launch(firefox ? {
  browser: "firefox",
  executablePath: "C:/Program Files/Mozilla Firefox/firefox.exe",
  headless: true,
  extraPrefsFirefox: { "media.navigator.streams.fake": true, "media.navigator.permission.disabled": true,
    "media.autoplay.default": 0 },
} : {
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(`http://127.0.0.1:${port}/`);
await page.waitForFunction(() => document.body.dataset.state === "idle", { timeout: 120_000 });
const before = await page.evaluate(async () => (await (await import("/session-log.js")).listSessions()).length);
await page.click("#start");
if (double) {
  await new Promise((r) => setTimeout(r, 300));
  await page.evaluate(() => document.querySelector("#start").click());
}
await page.waitForFunction(() => ["listening", "following"].includes(document.body.dataset.state),
  { timeout: 300_000, polling: 200 });
const t0 = Date.now();
await new Promise((r) => setTimeout(r, Number(secs) * 1000));
await page.evaluate(() => document.querySelector("#stop").click());
const wall = (Date.now() - t0) / 1000;
await page.waitForFunction(async (n) => {
  const list = await (await import("/session-log.js")).listSessions();
  return list.length > n && list.at(-1).ended;
}, { timeout: 60_000, polling: 500 }, before);
const info = await page.evaluate(async () => {
  const m = await import("/session-log.js");
  const rec = (await m.listSessions()).at(-1);
  const buf = await (await m.sessionFile(rec.id)).arrayBuffer();
  // RIFF: find the data chunk's size (16-bit mono 16 kHz)
  const v = new DataView(buf);
  let pos = 12, samples = 0, log = null;
  while (pos + 8 <= buf.byteLength) {
    const tag = String.fromCharCode(...new Uint8Array(buf, pos, 4));
    const size = v.getUint32(pos + 4, true);
    if (tag === "data") samples = size / 2;
    if (tag === "json") log = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, pos + 8, size)));
    pos += 8 + size + (size & 1);
  }
  const vis = (log?.events || []).filter((e) => e.type === "visible" && e.shown_ms);
  const slope = vis.length > 3 ? (vis.at(-1).end - vis[0].end) / ((vis.at(-1).shown_ms - vis[0].shown_ms) / 1000) : null;
  return { samples, audio_rate: log?.audio_rate, chunk: log?.chunk, ua: log?.ua, slope, mic: log?.mic };
});
const audio = info.samples / 16000;
console.log(`${browserName}${double ? " (Start tapped twice)" : ""}: ${audio.toFixed(1)} s of audio in ~${wall.toFixed(1)} s listening`
  + ` -> ${(audio / wall).toFixed(2)}x; log clock ${info.slope?.toFixed(2)} audio s per wall s; context ${info.audio_rate} Hz;`
  + ` mic ${JSON.stringify(info.mic)}; ${info.ua}`);
for (const e of errors) console.error("  page error:", e);
await browser.close();
server.close();
