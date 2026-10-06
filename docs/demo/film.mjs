// The demo, step 2: films stage.html, a row of phones running the real app (web/), each replaying
// a run capture.mjs recorded (docs/demo/runs/), frame by frame on a virtual clock, and lays their
// recordings under it.
//
//   node docs/demo/capture.mjs ... --out docs/demo/runs/1-name.json   # one per phone, in order
//   node docs/demo/film.mjs                    # -> data/cache/media/demo_with_audio.mp4 (+ _small)
//   node docs/demo/film.mjs --fast             # quick look: 30 fps, no motion blur
//   node docs/demo/film.mjs --stills 3,9,20    # PNGs at those film seconds, nothing else
//
// --stage story.html films the story instead (a boy at the mosque on a Thursday night; one phone,
// docs/demo/story-runs/), with the sounds docs/demo/story-sounds.mjs makes: --bed, a background
// the stage turns up and down (demo.bed()), and --sfx, the folder for its one-off sounds
// (demo.sfx: [{at, file, gain, pan}]). A stage's mix() may give a third number per phone, how
// muffled it is (heard through a wall: 1).
//
// Every frame is exactly 1/fps after the last (vclock.js), so nothing stutters however slow a
// screenshot is. While anything moves, a frame is the average of four samples spread over
// half the frame's time (a 180° shutter), as a film camera blurs motion; still frames are one
// sample. The app is the real page: only its engine is the stage's replay of a run, so the line
// and word each phone shows at each moment are the ones it showed beside the same audio.
//
// The sound is mixed here: each phone's recording, lined up with when that phone started
// listening, at the gain the stage gives it frame by frame and panned to where it is in the frame.
// The recitations are DuaPlayer's and other people's uploads, not ours to publish: the film with
// sound stays in the gitignored data/cache/media/.
//
// Needs Chrome, ffmpeg on PATH, and puppeteer-core (in data/cache/reliability/node, as for
// scripts/page_replay.mjs).
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { clipPath } from "./clip.mjs";

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
const FAST = process.argv.includes("--fast");
const FPS = Number(arg("fps", FAST ? 30 : 60));
const SAMPLES = FAST ? 1 : Number(arg("samples", 4)); // per moving frame
const SHUTTER = 0.5; // of the frame's time
const SCALE = Number(arg("scale", 1)); // device pixels per CSS pixel while filming
const LIMIT = Number(arg("seconds", 0)); // 0: the whole film
const FOLLOW = Number(arg("follow", 0)); // stage.html ?follow=: how long the first phone stays
const STILLS = arg("stills", "").split(",").filter(Boolean).map(Number);
const RUNS = resolve(arg("runs", join(HERE, "runs"))); // one run per phone, in name order
const STAGE = arg("stage", "stage.html");
const NAME = STAGE === "stage.html" ? "demo_with_audio" : STAGE.replace(/\.html$/, "");
const OUT = resolve(arg("out", join(MEDIA, FAST ? `${NAME}_fast.mp4` : `${NAME}.mp4`)));
const WORK = join(MEDIA, STAGE === "stage.html" ? "demo" : NAME); // masters, soundtracks, stills
const BED = arg("bed", null);
const SFX = arg("sfx", null);
const [WIDTH, HEIGHT] = [1920, 1080]; // stage.html
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
// A Wednesday evening: the home screen suggests no du'a "for tonight", so every du'a a phone
// shows is one it heard.
const DATE = new Date(2026, 8, 30, 20, 40).getTime();

// -- the runs, their clips and their loudness ------------------------------------------------
const runs = readdirSync(RUNS).filter((f) => f.endsWith(".json")).sort()
  .map((f) => JSON.parse(readFileSync(join(RUNS, f), "utf8")));
const decode = (args) => {
  const out = spawnSync("ffmpeg", ["-v", "error", ...args, "-ac", "1", "-f", "f32le", "-"], { maxBuffer: 2 ** 31 - 1 }).stdout;
  return new Float32Array(out.buffer, out.byteOffset, out.length / 4);
};
function fft(re, im) { // in place, radix 2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) [re[i], re[j], im[i], im[j]] = [re[j], re[i], im[j], im[i]];
  }
  for (let len = 2; len <= n; len <<= 1) {
    const a = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const c = Math.cos(a * k), s = Math.sin(a * k);
        const xr = re[i + k + len / 2] * c - im[i + k + len / 2] * s, xi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
        re[i + k + len / 2] = re[i + k] - xr;
        im[i + k + len / 2] = im[i + k] - xi;
        re[i + k] += xr;
        im[i + k] += xi;
      }
    }
  }
}
// Per phone: its recording's loudness from PRE s before its clip (10 ms RMS) for the lamp, and
// the 100-4000 Hz band power the page's analyser would see of the clip itself (voice.js), 60
// times a second, for its listening star.
const SR = 16000, RMS_HZ = 100, BAND_HZ = 60, PRE = Number(arg("pre", 8)), N = 1024;
const lo = Math.ceil((100 * N) / SR), hi = Math.floor((4000 * N) / SR);
const sounds = runs.map((run) => {
  const { audio, start, seconds } = run.source;
  const clip = clipPath(audio, start, seconds);
  if (!existsSync(clip)) throw new Error(`${clip} is missing: run docs/demo/capture.mjs for it again`);
  const heard = decode(["-ss", String(start - PRE), "-t", String(PRE + seconds), "-i", audio, "-ar", String(SR)]);
  const rms = [];
  for (let i = 0; i + SR / RMS_HZ <= heard.length; i += SR / RMS_HZ) {
    let e = 0;
    for (let j = i; j < i + SR / RMS_HZ; j++) e += heard[j] * heard[j];
    rms.push(Number(Math.sqrt(e / (SR / RMS_HZ)).toFixed(5)));
  }
  const pcm = decode(["-i", clip, "-ar", String(SR)]);
  const bandDb = [];
  for (let k = 0; k < Math.floor(seconds * BAND_HZ); k++) {
    const c = Math.floor((k / BAND_HZ) * SR);
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < N; i++) re[i] = (pcm[c - N + i] ?? 0) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
    fft(re, im);
    let e = 0;
    for (let b = lo; b <= hi; b++) e += re[b] * re[b] + im[b] * im[b];
    bandDb.push(Number((10 * Math.log10(e + 1e-12)).toFixed(1)));
  }
  const peak = [...rms].sort((a, b) => a - b)[Math.floor(rms.length * 0.985)] || 1;
  return { rms, rms_hz: RMS_HZ, rms_from: -PRE, band_db: bandDb, band_hz: BAND_HZ, peak };
});

// -- the stage, and the app with the stage's replay in place of its engine -------------------
const ENGINE = /  state\.engine = mode\?\.mode (===|!==) "server"\n/;
const REPLAY = '  state.engine = window.parent.__demo?.engine ? window.parent.__demo.engine(window.frameElement) : mode?.mode $1 "server"\n';
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".onnx": "application/octet-stream" };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const send = (body, type) => res.writeHead(200, { "content-type": type }).end(body);
  if (path === "/demo/runs.json") return send(JSON.stringify(runs), "application/json");
  if (path === "/demo/sounds.json") return send(JSON.stringify(sounds), "application/json");
  if (path === "/app/coi-serviceworker.js") return send("", "text/javascript"); // no threads needed: no engine
  const [base, rest] = path.startsWith("/app/") ? [WEB, path.slice(5)] : path.startsWith("/demo/") ? [HERE, path.slice(6)] : [null];
  const file = base && resolve(base, rest || "index.html");
  if (!file || !(file === base || file.startsWith(base + sep)) || !existsSync(file) || !statSync(file).isFile()) {
    return res.writeHead(404).end(); // /app/api/mode too: no server, as on a static host
  }
  if (file === join(WEB, "app.js")) {
    const src = readFileSync(file, "utf8").replace(/\r\n/g, "\n");
    if (!ENGINE.test(src)) throw new Error("app.js changed: update film.mjs (ENGINE)");
    return send(src.replace(ENGINE, REPLAY), "text/javascript");
  }
  send(readFileSync(file), TYPES[extname(file)] || "application/octet-stream");
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  protocolTimeout: 600_000,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required",
    "--mute-audio", "--hide-scrollbars", "--force-color-profile=srgb", "--font-render-hinting=none"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
page.on("console", (m) => m.type() === "error" && !m.text().includes("404") && console.error("console:", m.text()));
await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: SCALE });
const appPrefs = `if (location.pathname.startsWith("/app/")) { try { localStorage.setItem("text-size", "1.1"); } catch {} }`;
await page.evaluateOnNewDocument(`window.__vtStart = ${DATE};\n${readFileSync(join(HERE, "vclock.js"), "utf8")}\n${appPrefs}`);
await page.goto(`http://127.0.0.1:${server.address().port}/demo/${STAGE}${FOLLOW ? `?follow=${FOLLOW}` : ""}`);

// Let the page and the apps load (not filmed): the clock runs while fonts and the corpus arrive.
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
for (let i = 0; ; i++) {
  if (i > 1200) throw new Error("the stage never became ready");
  await page.evaluate(() => window.__vt?.step(50));
  await sleep(20);
  const ready = await page.evaluate(() => {
    const docs = [...document.querySelectorAll("iframe")].map((f) => f.contentDocument);
    return !!(window.__stageReady && docs.length && document.fonts.status === "loaded" &&
      docs.every((d) => d?.getElementById("footnote")?.textContent && d.fonts.status === "loaded"));
  });
  if (ready) break;
}
for (let i = 0; i < 30; i++) await page.evaluate(() => window.__vt.step(50)); // settle

const client = await page.createCDPSession();
// A screenshot can stall when the machine is short of memory (other jobs): the clock doesn't move
// while it waits, so asking again films the same instant.
const shot = async () => {
  for (let attempt = 1; ; attempt++) {
    const png = client.send("Page.captureScreenshot", { format: "png", optimizeForSpeed: true });
    const late = new Promise((_, no) => setTimeout(() => no(new Error("screenshot stalled")), 60_000));
    try {
      return Buffer.from((await Promise.race([png, late])).data, "base64");
    } catch (e) {
      png.catch(() => {});
      if (attempt >= 4) throw e;
      console.error(`\n${e.message}; again (${attempt})`);
    }
  }
};
const step = (ms) => page.evaluate((ms) => window.__vt.step(ms), ms);
// After a phone starts, the page sets up its audio (real async work: the microphone, the
// worklet) under "Preparing…": a phone that has used the app before is past it in a moment.
// The clock waits.
const settle = async () => {
  while (await page.evaluate(() => [...document.querySelectorAll("iframe")]
    .some((f) => f.contentDocument.body.dataset.state === "loading"))) await sleep(10);
};
await page.evaluate(() => window.__demo.begin());
const plan = await page.evaluate(() => window.__demo.plan);
console.log(`plan (film s): ${plan.map((p, k) => `${k + 1}: heard ${p.voice.toFixed(1)}, tap ${p.tap.toFixed(1)}, ` +
  `found ${p.found.toFixed(1)}, leave ${p.leave.toFixed(1)}`).join("; ")}; ends ${(await page.evaluate(() => window.__demo.END)).toFixed(1)}`);

// -- stills --------------------------------------------------------------------------------
if (STILLS.length) {
  mkdirSync(join(WORK, "stills"), { recursive: true });
  let t = 0;
  for (const at of [...STILLS].sort((a, b) => a - b)) {
    for (; t + 1 / 60 <= at; t += 1 / 60) {
      await step(1000 / 60);
      await settle();
    }
    const f = join(WORK, "stills", `t${at.toFixed(1).padStart(5, "0")}.png`);
    writeFileSync(f, await shot());
    console.log(f);
  }
  await browser.close();
  server.close();
  process.exit(0);
}

// -- the film ------------------------------------------------------------------------------
mkdirSync(WORK, { recursive: true });
const master = join(WORK, FAST ? "master_fast.mkv" : "master.mkv");
// Four samples in per frame out: tmix averages each frame's four (in 16-bit, so the average
// doesn't band), select keeps one per frame.
const blur = SAMPLES > 1
  ? `format=gbrp16le,tmix=frames=${SAMPLES},select='eq(mod(n\\,${SAMPLES})\\,${SAMPLES - 1})',setpts=N/${FPS}/TB,`
  : "";
const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-framerate", String(FPS * SAMPLES), "-c:v", "png", "-i", "-",
  "-vf", `${blur}scale=${WIDTH}:${HEIGHT}:flags=lanczos,format=yuv444p`, "-r", String(FPS),
  "-c:v", "libx264", "-preset", "medium", "-crf", "6", master], { stdio: ["pipe", "inherit", "inherit"] });
const write = async (buf) => {
  if (!ff.stdin.write(buf)) await new Promise((ok) => ff.stdin.once("drain", ok));
};
const frameMs = 1000 / FPS, subMs = (frameMs * SHUTTER) / SAMPLES;
const t = Date.now();
let f = 0, moving = 0, blurred = 0;
const mixes = [], beds = []; // per frame: [gain, pan, muffle] per phone; the bed's gain
for (;; f++) {
  await settle();
  const m = await page.evaluate(() => {
    const d = window.__demo;
    d.motion = 0;
    return { mix: d.mix(), bed: d.bed?.() ?? 0 };
  });
  mixes.push(m.mix);
  beds.push(m.bed);
  const first = await shot();
  if (moving > 0.4 && SAMPLES > 1) {
    // Moving: the rest of the shutter's samples, then on to the next frame.
    blurred++;
    await write(first);
    for (let s = 1; s < SAMPLES; s++) {
      await step(subMs);
      await settle();
      await write(await shot());
    }
    await step(frameMs - subMs * (SAMPLES - 1));
  } else {
    for (let s = 0; s < SAMPLES; s++) await write(first);
    await step(frameMs);
  }
  // How far things moved over this frame decides the next one (motion is smooth: springs).
  moving = await page.evaluate(() => window.__demo.motion);
  if ((LIMIT && f + 1 >= LIMIT * FPS) || (await page.evaluate(() => window.__demo.done))) break;
  if (f % (5 * FPS) === 0) process.stdout.write(`\r${(f / FPS).toFixed(0)} s filmed (${((Date.now() - t) / 1000).toFixed(0)} s, ${blurred} blurred)`);
}
const starts = await page.evaluate(() => window.__demo.starts());
const sfx = await page.evaluate(() => window.__demo.sfx ?? []);
const unpaused = await page.evaluate(() => window.__vt.unpaused);
if (unpaused) console.log(`\n${unpaused} times a CSS animation was found running and paused again (vclock.js)`);
ff.stdin.end();
await new Promise((ok) => ff.on("close", ok));
await browser.close();
server.close();
console.log(`\n${f + 1} frames (${blurred} with motion blur) in ${((Date.now() - t) / 1000).toFixed(0)} s; ` +
  `phones listening from ${starts.map((x) => x?.toFixed(2)).join(", ")} s`);

// -- the sound ---------------------------------------------------------------------------------
// Each phone's recording, lined up so its clip's first sample plays as that phone starts
// listening, brought to one loudness, at the gain the stage gave it frame by frame (the phone the
// camera is with; crossfades on the way) and panned to where the phone is in the frame.
const AR = 48000, frames = mixes.length, len = Math.ceil((frames / FPS) * AR);
const left = new Float32Array(len), right = new Float32Array(len);
runs.forEach((run, k) => {
  if (starts[k] == null) return; // never reached (a --seconds preview)
  const { audio, start: clipAt, seconds } = run.source;
  const from = clipAt - starts[k]; // the recording's second at film time 0
  const lead = Math.max(0, -from);
  const pcm = decode(["-ss", String(Math.max(0, from)), "-t", String(seconds + starts[k] - lead), "-i", audio,
    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", String(AR)]);
  const off = Math.round(lead * AR);
  // Through a wall: the same, lowpassed (two poles at 650 Hz), for the frames that muffle it.
  const muffled = mixes.some((m) => (m[k][2] ?? 0) > 0) ? (() => {
    const y = Float32Array.from(pcm), a = 1 - Math.exp((-2 * Math.PI * 650) / AR);
    for (let pass = 0; pass < 2; pass++) for (let i = 0, v = 0; i < y.length; i++) y[i] = v += a * (y[i] - v);
    return y;
  })() : null;
  for (let fr = 0; fr < frames; fr++) {
    const [g0, p0, m0 = 0] = mixes[fr][k], [g1, p1, m1 = 0] = mixes[Math.min(fr + 1, frames - 1)][k];
    if (g0 === 0 && g1 === 0) continue;
    const a = Math.round((fr / FPS) * AR), b = Math.min(len, Math.round(((fr + 1) / FPS) * AR));
    for (let n = a; n < b; n++) {
      const u = (n - a) / (b - a);
      const dry = pcm[n - off];
      if (dry === undefined) continue;
      const mu = m0 + (m1 - m0) * u;
      const x = muffled ? dry * (1 - mu) + muffled[n - off] * mu * 1.25 : dry;
      const g = g0 + (g1 - g0) * u, th = ((p0 + (p1 - p0) * u + 1) * Math.PI) / 4; // equal-power pan
      left[n] += x * g * Math.cos(th) * Math.SQRT2;
      right[n] += x * g * Math.sin(th) * Math.SQRT2;
    }
  }
});
// The bed (looped from film time 0, at the stage's gain frame by frame), and the one-off sounds.
const decodeStereo = (file) => {
  const st = spawnSync("ffmpeg", ["-v", "error", "-i", file, "-ac", "2", "-ar", String(AR), "-f", "f32le", "-"], { maxBuffer: 2 ** 31 - 1 }).stdout;
  return new Float32Array(st.buffer, st.byteOffset, st.length / 4);
};
if (BED) {
  const bed = decodeStereo(resolve(BED)), bn = bed.length / 2;
  for (let fr = 0; fr < frames; fr++) {
    const g0 = beds[fr], g1 = beds[Math.min(fr + 1, frames - 1)];
    if (g0 === 0 && g1 === 0) continue;
    const a = Math.round((fr / FPS) * AR), b = Math.min(len, Math.round(((fr + 1) / FPS) * AR));
    for (let n = a; n < b; n++) {
      const g = g0 + ((g1 - g0) * (n - a)) / (b - a), i = n % bn;
      left[n] += bed[2 * i] * g;
      right[n] += bed[2 * i + 1] * g;
    }
  }
}
for (const { at, file, gain = 1, pan = 0 } of sfx) {
  const x = decodeStereo(join(resolve(SFX ?? "."), file)), th = ((pan + 1) * Math.PI) / 4;
  for (let i = 0, n = Math.round(at * AR); i < x.length / 2 && n < len; i++, n++) {
    if (n < 0) continue;
    left[n] += x[2 * i] * gain * Math.cos(th) * Math.SQRT2;
    right[n] += x[2 * i + 1] * gain * Math.sin(th) * Math.SQRT2;
  }
}
// Two reciters overlap in a handover: the rare peak over 0.8 is rounded off rather than the
// whole film turned down.
const limit = (x) => (Math.abs(x) <= 0.8 ? x : Math.sign(x) * (0.8 + 0.18 * Math.tanh((Math.abs(x) - 0.8) / 0.18)));
const wav = Buffer.alloc(44 + len * 4);
wav.write("RIFF", 0);
wav.writeUInt32LE(36 + len * 4, 4);
wav.write("WAVEfmt ", 8);
wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20); // PCM
wav.writeUInt16LE(2, 22); // stereo
wav.writeUInt32LE(AR, 24);
wav.writeUInt32LE(AR * 4, 28);
wav.writeUInt16LE(4, 32);
wav.writeUInt16LE(16, 34);
wav.write("data", 36);
wav.writeUInt32LE(len * 4, 40);
for (let n = 0; n < len; n++) {
  wav.writeInt16LE(Math.round(limit(left[n]) * 32767), 44 + n * 4);
  wav.writeInt16LE(Math.round(limit(right[n]) * 32767), 46 + n * 4);
}
const soundtrack = join(WORK, FAST ? "soundtrack_fast.wav" : "soundtrack.wav");
writeFileSync(soundtrack, wav);
const r = spawnSync("ffmpeg", ["-v", "error", "-y", "-i", master, "-i", soundtrack, "-map", "0:v", "-map", "1:a", "-shortest",
  "-c:v", "libx264", "-preset", "slow", "-crf", FAST ? "23" : "19", "-tune", "film", "-pix_fmt", "yuv420p",
  "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", OUT], { stdio: "inherit" });
if (r.status) throw new Error("ffmpeg couldn't mux the film");
console.log(`${OUT}: ${(statSync(OUT).size / 1e6).toFixed(1)} MB`);

// A copy under 10 MB, the most GitHub takes for a video in a README on a free plan: 720p at 60
// fps, two passes to the size.
if (!FAST && !LIMIT) {
  const small = OUT.replace(/\.mp4$/, "_small.mp4");
  const dur = Number(spawnSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", OUT]).stdout);
  const kbps = Math.floor((9.3e6 * 8) / dur / 1000) - 128;
  const pass = (n, out) => spawnSync("ffmpeg", ["-v", "error", "-y", "-i", OUT, "-vf", "scale=1280:720:flags=lanczos",
    "-c:v", "libx264", "-preset", "slow", "-b:v", `${kbps}k`, "-pass", String(n), "-passlogfile", join(WORK, "x264"),
    "-pix_fmt", "yuv420p", ...(n === 1 ? ["-an", "-f", "mp4", out] : ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out])],
  { stdio: "inherit" });
  pass(1, process.platform === "win32" ? "NUL" : "/dev/null");
  pass(2, small);
  console.log(`${small}: ${(statSync(small).size / 1e6).toFixed(1)} MB`);
}
