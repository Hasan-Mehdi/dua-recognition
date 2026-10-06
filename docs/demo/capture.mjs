// The demo, step 1: what the app does with a recording, captured from the real page.
//
//   node docs/demo/capture.mjs --sid SID --line K --seconds S --out docs/demo/runs/N-name.json
//
// The film's five phones (film.mjs takes docs/demo/runs/ in name order), chosen for variety from
// a screening of 21 runs (17 reciters and congregations, all held out of training): 14 named the
// du'a in 5.9-7.2 s from a cold start mid-du'a; Iftitah (three times), Waritha and Simaat came up
// first as a text that shares their passages; Hasan's own phone readings took 16 and 27 s.
//
//   1-kumayl    --sid majlis:mj-rSPgtn4Km_w-2226:0 --line 4 --into 3.4 --seconds 84
//               --credit "Congregation at KSIJ, Dar es Salaam"
//               (from 1.2 s in it was found as line 7 began, showed line 6's last word, then
//               jumped back to its second; from 3.4 s in it lands on line 7 and only goes forward)
//   2-ashura    --sid harvest:web:sv-0E4Vf:0.0 --line 2 --into -2.8 --seconds 74 --credit "Kareem al-Ghurawi"
//               (from 1.2 s in it was found during "ahla l-bayt" but placed on "ʿalaykum", then went
//               on to the next line; from 2.8 s before the line it follows "...ʿalaykum ahla l-bayt")
//   3-nudbah    --sid harvest:soundcloud:sc-354021932:0.0 --line 4 --into 1.2 --seconds 56 --credit "Sheikh Arastu"
//   4-jawshan   --sid harvest:soundcloud:sc-821680771:0.0 --line 6 --into 1.2 --seconds 44
//               --credit "Sheikh Fadhil al-Maliki"
//   5-tawassul  --line 39 --into 1.2 --seconds 32 (DuaPlayer's Hussein Ghareeb, the default recording)
//
// Each starts 1.2 s into a line, so the film can let the reader be heard before the tap, and
// needs the length the film keeps it going for (stage.html's plan): the first phone is still
// following in the last shot.
//
// --sid takes a recording from the scenario bench's held-out sources (data/testbed/sources.jsonl,
// scripts/bench.py: studio, majlis, harvest and user lanes) and starts the clip just before its
// --line'th timed line.
//
// The clip plays through the page's own "play a recording" path in headless Chrome, with the
// on-device engine a phone runs (Whisper and the CTC follower in their workers, tracker.js,
// stream-follower.js). Each Whisper and CTC step is held to at least the time it takes on a
// phone (--asr-ms, --ctc-ms: Hasan's Android, docs/results/phone_latency.md), as
// scripts/bench_page.py does. Every update the engine hands the page is kept with the clip
// position playing at that moment, with the stop detector's reading: film.mjs replays them
// into the same page, frame by frame, beside the same audio.
//
// puppeteer-core lives in data/cache/reliability/node, as for scripts/page_replay.mjs.
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { clipPath } from "./clip.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const WEB = join(ROOT, "web");
const MEDIA = join(ROOT, "data", "cache", "media", "demo");
const require = createRequire(join(ROOT, "data", "cache", "reliability", "node", "package.json"));
const puppeteer = require("puppeteer-core");

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
};
// Without --sid: Hussein Ghareeb's Dua Tawassul (DuaPlayer; a voice held out of training), from
// line 40, "yā zayna l-ʿābidīn", in the breath before it (from line 39, 304 s, it was found
// 7.2 s in; from here, 5.9 s).
const SID = arg("sid", null);
const bench = SID && readFileSync(join(ROOT, "data", "testbed", "sources.jsonl"), "utf8").split("\n")
  .filter(Boolean).map((l) => JSON.parse(l)).find((r) => r.sid === SID);
if (SID && !bench) throw new Error(`${SID} isn't in data/testbed/sources.jsonl`);
const DUA = bench?.dua ?? arg("dua", "dua-tawassul");
const AUDIO_ID = arg("audio", "8f8a74b3-4a0d-483d-871a-08d9bf717fd8");
// --file X.wav: any recording (a bench item rendered to a file, say), with its line starts in
// X.lines.json ([{seg, from}], a line may come twice); use --start, not --line.
const FILE = arg("file", null);
const AUDIO = bench ? bench.audio : FILE ? resolve(FILE) : join(ROOT, "data", "duaplayer", DUA, `${AUDIO_ID}.mp3`);
// The bench's line times (its own, or DuaPlayer's hand timings), in the recording's seconds.
const LINES = bench ? bench.lines.map((l) => ({ seg: l.seg, from: l.from, to: l.to }))
  : FILE ? JSON.parse(readFileSync(AUDIO.replace(/\.wav$/, ".lines.json"), "utf8"))
  : Object.entries(JSON.parse(readFileSync(AUDIO.replace(/\.mp3$/, ".json"), "utf8")).slide_start_ms)
    .map(([seg, ms]) => ({ seg: Number(seg), from: ms / 1000 })).sort((a, b) => a.from - b.from);
const LINE = Number(arg("line", -1));
// The app starts listening --into s after the line begins (the film lets the reader be heard
// first, then a finger taps start), or in the breath before it; a negative --into starts it that
// long before the line, at the end of the one before (so it can be following when the line comes).
const INTO = Number(arg("into", 0));
// Where the reader comes in for the film: the breath before the line, never inside the one before;
// with a negative --into, 1.5 s before the app starts, as long as the others are heard.
const ENTRY = LINE < 0 ? null : INTO < 0 ? Number((LINES[LINE].from + INTO - 1.5).toFixed(2))
  : Number(Math.max(LINES[LINE].from - 0.3, LINES[LINE - 1]?.to ?? 0).toFixed(2));
const START = LINE >= 0 ? Number((INTO !== 0 ? LINES[LINE].from + INTO : ENTRY).toFixed(2)) : Number(arg("start", 312.0));
const CREDIT = arg("credit", bench ? bench.voice : "Hussein Ghareeb"); // who the film says is reciting
const SECONDS = Number(arg("seconds", 49.2));
const ASR_MS = Number(arg("asr-ms", 1200));
const CTC_MS = Number(arg("ctc-ms", 150));
if (!arg("out")) throw new Error("--out: where the run goes (docs/demo/runs/ holds the film's)");
const OUT = resolve(arg("out"));
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";

// -- the clip ----------------------------------------------------------------------
mkdirSync(MEDIA, { recursive: true });
if (!existsSync(AUDIO)) throw new Error(`${AUDIO} is missing (python scripts/fetch_duaplayer.py, or the bench's sources)`);
const clip = clipPath(AUDIO, START, SECONDS);
const cut = spawnSync("ffmpeg", ["-v", "error", "-y", "-ss", String(START), "-t", String(SECONDS), "-i", AUDIO,
  "-af", `afade=t=in:d=0.04,afade=t=out:st=${SECONDS - 0.06}:d=0.06`, "-c:a", "pcm_s16le", clip], { stdio: "inherit" });
if (cut.status) throw new Error("ffmpeg couldn't cut the clip");

// -- the page, with a recorder where the engine hands it updates ---------------------
const HOOK = "  await state.engine.start(update);\n";
const RECORDER = `  const demo = (window.__demo ??= { updates: [], quiet: [], letters: [] });
  // What the CTC model heard: each new frame's best column (0 blank, else the letter 0x0620 + c),
  // at its second in the clip, with the clip's position when the step came back (inside.html).
  const eng = state.engine, onFrames = eng._onFrames;
  let heardTo = -1;
  eng._onFrames = function (data) {
    const at = $("player").currentTime, hop = (this.ctcWindow || 2 * SR) / SR / data.T;
    for (let k = 0; k < data.T; k++) {
      const t = data.id / SR - (data.T - 1 - k) * hop;
      if (t <= heardTo + 1e-6) continue;
      let best = 0;
      for (let c = 1; c < data.C; c++) if (data.frames[k * data.C + c] > data.frames[k * data.C + best]) best = c;
      demo.letters.push([Number(t.toFixed(3)), best, Number(at.toFixed(3))]);
      heardTo = t;
    }
    return onFrames.call(this, data);
  };
  await state.engine.start((u) => {
    demo.updates.push({ at: $("player").currentTime, heard: state.engine.total / SR, u: JSON.parse(JSON.stringify(u)) });
    update(u);
  });
  clearInterval(demo.ear);
  demo.ear = setInterval(() => demo.quiet.push([$("player").currentTime, state.engine.ear?.quiet ?? null]), 20);
`;
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".onnx": "application/octet-stream", ".css": "text/css", ".txt": "text/plain", ".svg": "image/svg+xml",
  ".wav": "audio/wav", ".woff2": "font/woff2", ".png": "image/png", ".webmanifest": "application/manifest+json" };
// A step held to at least `ms`: the worker waits before it answers (as page_replay.mjs --asr-ms).
const hold = (src, reply, ms, name) => {
  if (!src.includes(reply)) throw new Error(`${name} changed: update capture.mjs`);
  return src.replace(reply, `await new Promise((r) => setTimeout(r, ${ms} - (performance.now() - t0)));\n    ${reply}`);
};
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const f = join(WEB, path.endsWith("/") ? path + "index.html" : path);
  if (!f.startsWith(WEB) || !existsSync(f) || statSync(f).isDirectory()) return res.writeHead(404).end();
  res.setHeader("Content-Type", TYPES[extname(f)] || "application/octet-stream");
  let src = readFileSync(f);
  if (path === "/app.js") {
    src = String(src).replace(/\r\n/g, "\n"); // a Windows checkout (autocrlf)
    if (!src.includes(HOOK)) throw new Error("app.js changed: update capture.mjs (HOOK)");
    src = src.replace(HOOK, RECORDER);
  }
  if (path === "/asr-worker.js" && ASR_MS) src = hold(String(src), 'self.postMessage({ type: "text"', ASR_MS, path);
  if (path === "/ctc-worker.js" && CTC_MS) src = hold(String(src), 'self.postMessage({ type: "frames"', CTC_MS, path);
  res.end(src);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && !m.text().includes("404") && errors.push(m.text()));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
// coi-serviceworker.js reloads the page once (cross-origin isolation, for threads): wait for the
// page that comes back.
let before;
for (let attempt = 0; ; attempt++) {
  try {
    await page.waitForFunction(() => document.body.dataset.state === "idle", { timeout: 60_000 });
    await new Promise((r) => setTimeout(r, 1500));
    await page.waitForFunction(() => document.body.dataset.state === "idle", { timeout: 60_000 });
    before = await page.evaluate(async () => (await (await import("/session-log.js")).listSessions()).length);
    break;
  } catch (e) {
    if (attempt >= 3 || !String(e).includes("context was destroyed")) throw e;
  }
}
const t0 = Date.now();
await (await page.$("#file")).uploadFile(clip);
await page.waitForFunction(() => ["listening", "following"].includes(document.body.dataset.state),
  { timeout: 300_000, polling: 200 });
await page.waitForFunction(() => document.body.dataset.state === "idle", { timeout: 600_000, polling: 500 });
await page.waitForFunction(async (n) => {
  const list = await (await import("/session-log.js")).listSessions();
  return list.length > n && list.at(-1).ended;
}, { timeout: 60_000, polling: 500 }, before);

const demo = await page.evaluate(() => ({ updates: window.__demo.updates, quiet: window.__demo.quiet,
  letters: window.__demo.letters }));
// The session's log (session-log.js: a "json" RIFF chunk after the audio): what the page showed.
const wav = Buffer.from(await page.evaluate(async () => {
  const m = await import("/session-log.js");
  const bytes = new Uint8Array(await (await m.sessionFile((await m.listSessions()).at(-1).id)).arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}), "base64");
await browser.close();
server.close();

let log = null;
for (let p = 12; p + 8 <= wav.length; ) {
  const id = wav.toString("ascii", p, p + 4), n = wav.readUInt32LE(p + 4);
  if (id === "json") log = JSON.parse(wav.toString("utf8", p + 8, p + 8 + n));
  p += 8 + n + (n & 1);
}
const events = (log?.events ?? []).filter((e) => ["dua", "line", "word", "preview", "hush"].includes(e.type ?? e.e));

const corpus = JSON.parse(readFileSync(join(WEB, "corpus.json"), "utf8"));
const dua = corpus.find((d) => d.id === DUA);
// When the reader starts each line, in the clip's seconds, for checking the run (not used to film).
const truth = LINES.map((l) => ({ seg: l.seg, from: Number((l.from - START).toFixed(2)) }))
  .filter((l) => l.from > -15 && l.from < SECONDS);
writeFileSync(OUT, JSON.stringify({
  source: { dua: DUA, name: dua?.name_en, name_ar: dua?.name_ar, credit: CREDIT, sid: SID,
    voice: bench?.voice ?? "studio:Hussein Ghareeb", audio: AUDIO, entry: ENTRY ?? START, start: START, seconds: SECONDS,
    asr_ms: ASR_MS, ctc_ms: CTC_MS },
  updates: demo.updates.map((r) => ({ ...r, at: Number(r.at.toFixed(3)), heard: Number(r.heard.toFixed(3)) })),
  quiet: demo.quiet.filter(([, q], i, a) => i === 0 || q !== a[i - 1][1]).map(([at, q]) => [Number(at.toFixed(3)), q == null ? null : Number(q.toFixed(2))]),
  shown: events,
  truth,
  // Each Whisper update (the window it read ends at `end`; `t` is when it came back) and each CTC
  // frame's best column, for inside.html.
  hops: (log?.events ?? []).filter((e) => (e.type ?? e.e) === "hop" && e.skip == null)
    .map((e) => ({ t: e.t, end: e.end, text: e.text, dua: e.dua, dua_p: e.dua_p })),
  letters: demo.letters,
}));

// What the page showed against when the reader got there: the du'a, and each line it entered
// (session time is the clip's within a few ms: the log's samples are the clip's own).
const found = demo.updates.find((r) => r.u.dua);
const from = Object.fromEntries(truth.map((l) => [l.seg, l.from]));
const shownLines = events.filter((e) => e.type === "line").map((e) => {
  const late = from[e.seg] == null ? "?" : (e.t - Math.max(0, from[e.seg])).toFixed(2);
  return `${e.seg}${from[e.seg] == null ? "(not read here)" : ""}@${e.t.toFixed(1)}:${late}`;
});
console.log(`${OUT}: ${demo.updates.length} updates in ${((Date.now() - t0) / 1000).toFixed(0)} s; ` +
  `${found ? `${found.u.dua}${found.u.dua === DUA ? "" : ` (WRONG, read: ${DUA})`} at ${found.at.toFixed(2)} s` : "never found"}`);
console.log(`  lines (shown@s:late): ${shownLines.join(" ")}`);
if (errors.length) console.error(`${errors.length} page errors:\n  ${errors.join("\n  ")}`);
