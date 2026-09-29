// Films the demo: the real app (web/) inside stage.html, frame by frame on a virtual clock.
//
//   python docs/demo/timeline.py          # the recognizer's messages for the recording
//   node docs/demo/record.mjs             # -> docs/demo.gif, docs/demo.mp4 (silent),
//                                         #    data/cache/media/demo_with_audio.mp4 (local only)
//   node docs/demo/record.mjs --seconds 8 # a quick look at the start
//
// With gifsicle (GIFSICLE=path or on PATH) the GIF is also optimized, lossily: about half the size.
//
// Every frame is exactly 1/fps after the last (vclock.js), so nothing stutters however slow
// a screenshot is. The page's /ws stream is the recorded messages, delivered when the
// server sent them, and the listening star hears the recording's own loudness. The
// recording's audio is not ours to publish: only the with-audio copy carries it, and it
// stays in the gitignored data/cache/media/.
//
// Needs Chrome, ffmpeg on PATH, and puppeteer-core (npm i puppeteer-core in
// data/cache/reliability/node, as scripts/gate_bench.mjs).
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const WEB = join(ROOT, "web");
const MEDIA = join(ROOT, "data", "cache", "media");
const require = createRequire(join(ROOT, "data", "cache", "reliability", "node", "package.json"));
const puppeteer = require("puppeteer-core");

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const FPS = Number(arg("fps", 25));
const LIMIT = Number(arg("seconds", 0)); // 0: the whole demo
const GIF_WIDTH = Number(arg("gif-width", 800));
const [WIDTH, HEIGHT] = [1100, 800]; // stage.html
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
// A Tuesday evening: the home screen offers Dua Tawassul "for tonight", as it would then.
const DATE = new Date(2026, 8, 29, 20, 40).getTime();

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".onnx": "application/octet-stream" };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const send = (body, type) => res.writeHead(200, { "content-type": type }).end(body);
  if (path === "/app/api/mode") return send(JSON.stringify({ mode: "server", model: "whisper-turbo-dua", sessions: false }), "application/json");
  if (path === "/app/coi-serviceworker.js") return send("", "text/javascript"); // server mode needs no isolation
  const [base, rest] = path.startsWith("/app/") ? [WEB, path.slice(5)] : path.startsWith("/demo/") ? [HERE, path.slice(6)] : [null];
  const file = base && resolve(base, rest || "index.html");
  if (!file || !(file === base || file.startsWith(base + sep)) || !existsSync(file) || !statSync(file).isFile()) {
    return res.writeHead(404).end();
  }
  send(readFileSync(file), TYPES[extname(file)] || "application/octet-stream");
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const url = `http://127.0.0.1:${server.address().port}/demo/stage.html`;

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required",
    "--mute-audio", "--hide-scrollbars", "--force-color-profile=srgb", "--font-render-hinting=none"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
page.on("console", (m) => m.type() === "error" && console.error("console:", m.text()));
await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 2 });
const appPrefs = `if (location.pathname.startsWith("/app/")) { try { localStorage.setItem("text-size", "1.1"); } catch {} }`;
await page.evaluateOnNewDocument(`window.__vtStart = ${DATE};\n${readFileSync(join(HERE, "vclock.js"), "utf8")}\n${appPrefs}`);
await page.goto(url);

// Let the page and the app load (not filmed): the clock runs while fonts and the corpus arrive.
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
for (let i = 0; ; i++) {
  if (i > 600) throw new Error("the stage never became ready");
  await page.evaluate(() => window.__vt?.step(50));
  await sleep(20);
  const ready = await page.evaluate(() => {
    const d = document.getElementById("app")?.contentDocument;
    return !!(window.__stageReady && d?.getElementById("today-chips")?.children.length &&
      document.fonts.status === "loaded" && d.fonts.status === "loaded");
  });
  if (ready) break;
}
for (let i = 0; i < 30; i++) await page.evaluate(() => window.__vt.step(50)); // settle

mkdirSync(join(MEDIA, "demo"), { recursive: true });
const master = join(MEDIA, "demo", "master.mkv");
const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-",
  "-vf", `scale=${WIDTH}:${HEIGHT}:flags=lanczos`, "-c:v", "libx264", "-preset", "medium", "-crf", "8", "-pix_fmt", "yuv444p", master],
  { stdio: ["pipe", "inherit", "inherit"] });
const client = await page.createCDPSession();
await page.evaluate(() => window.__demo.begin());
const t = Date.now();
let f = 0;
for (;; f++) {
  if (f > 0) await page.evaluate((ms) => window.__vt.step(ms), 1000 / FPS);
  if (f < 20 * FPS) await sleep(4); // the tap starts real async work (audio worklet, microphone)
  const { data } = await client.send("Page.captureScreenshot", { format: "png", optimizeForSpeed: true });
  if (!ff.stdin.write(Buffer.from(data, "base64"))) await new Promise((ok) => ff.stdin.once("drain", ok));
  if ((LIMIT && f + 1 >= LIMIT * FPS) || (await page.evaluate(() => window.__demo.done))) break;
  if (f % (5 * FPS) === 0) process.stdout.write(`\r${(f / FPS).toFixed(0)} s filmed (${((Date.now() - t) / 1000).toFixed(0)} s)`);
}
const t0 = await page.evaluate(() => (window.__demo.t0 - window.__demo.c0) / 1000);
ff.stdin.end();
await new Promise((ok) => ff.on("close", ok));
await browser.close();
server.close();
console.log(`\n${f + 1} frames; listening from ${t0?.toFixed(2)} s`);

const run = (args) => {
  const r = spawnSync("ffmpeg", ["-v", "error", "-y", ...args], { stdio: "inherit" });
  if (r.status) throw new Error(`ffmpeg ${args.at(-1)} failed`);
};
const tag = LIMIT ? "_preview" : "";
const out = (name) => (LIMIT ? join(MEDIA, "demo", name.replace(".", `${tag}.`)) : join(ROOT, "docs", name));

// The README's GIF: one palette for the whole film, frames diffed.
run(["-i", master, "-vf", `fps=${FPS},scale=${GIF_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];` +
  "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", out("demo.gif")]);
const gifsicle = process.env.GIFSICLE || "gifsicle";
if (!spawnSync(gifsicle, ["-O3", "--lossy=80", "-b", out("demo.gif")], { stdio: "inherit" }).error) console.log("gifsicle: optimized");
// A silent MP4 beside it, and the with-audio copy that stays local.
run(["-i", master, "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out("demo.mp4")]);
if (!LIMIT) {
  const tl = JSON.parse(readFileSync(join(HERE, "timeline.json"), "utf8")).source;
  const mp3 = join(ROOT, "data", "duaplayer", tl.dua, `${tl.audio_id}.mp3`);
  const ms = Math.round(t0 * 1000);
  run(["-i", master, "-ss", String(tl.start), "-t", String(tl.seconds), "-i", mp3, "-filter_complex",
    `[1:a]adelay=${ms}|${ms},afade=t=out:st=${(t0 + 44.2).toFixed(2)}:d=1.5,apad[a]`, "-map", "0:v", "-map", "[a]", "-shortest",
    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
    join(MEDIA, "demo_with_audio.mp4")]);
}
for (const f of [out("demo.gif"), out("demo.mp4")]) console.log(`${f}: ${(statSync(f).size / 1e6).toFixed(1)} MB`);
