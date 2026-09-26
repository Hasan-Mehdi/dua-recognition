// Debug sessions: on for now (?log=0 turns them off). Each listening session
// keeps the 16 kHz audio the recognizer heard and a timestamped log of what it
// made of it (every window's transcript and tracker state, every line and word
// the page showed, every tap), so a session can be replayed and scored offline
// (scripts/session_report.py) instead of screen-recorded.
//
// A session is one .wav file with the log in a "json" RIFF chunk after the
// audio: players ignore the chunk, and one file can't lose its other half.
// Sessions live in this browser's IndexedDB, written every few seconds so a
// killed tab loses little, until they're sent: automatically to the server the
// page came from when it takes them (app/server.py), or from the home screen.
//
// Every event carries t, seconds into the session's audio (so t = 73.2 is 73.2 s
// into the .wav), and ms, wall-clock milliseconds since the start.

const SR = 16000;
const FLUSH_MS = 5000;
const KEEP_BYTES = 400e6; // past this, the oldest sessions are deleted
const round = (x, k = 3) => (x == null || !Number.isFinite(x) ? x : Number(x.toFixed(k)));

// -- IndexedDB -----------------------------------------------------------------
let dbPromise = null;
function db() {
  dbPromise ??= new Promise((ok, err) => {
    const req = indexedDB.open("dua-sessions", 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore("sessions", { keyPath: "id" });
      req.result.createObjectStore("chunks", { keyPath: ["id", "seq"] });
    };
    req.onsuccess = () => ok(req.result);
    req.onerror = () => err(req.error);
  });
  return dbPromise;
}

async function run(name, mode, fn) {
  const tx = (await db()).transaction(name, mode);
  const req = fn(tx.objectStore(name));
  return new Promise((ok, err) => {
    tx.oncomplete = () => ok(req?.result);
    tx.onerror = tx.onabort = () => err(tx.error);
  });
}

const chunkRange = (id) => IDBKeyRange.bound([id, 0], [id, Infinity]);
export const listSessions = () => run("sessions", "readonly", (s) => s.getAll())
  .then((all) => all.sort((a, b) => a.started - b.started));
const putSession = (rec) => run("sessions", "readwrite", (s) => s.put(rec));

async function deleteSession(id) {
  await run("chunks", "readwrite", (s) => s.delete(chunkRange(id)));
  await run("sessions", "readwrite", (s) => s.delete(id));
}

export async function clearSessions(keep = null) {
  for (const rec of await listSessions()) if (rec.id !== keep) await deleteSession(rec.id);
}

// -- one session as a file -------------------------------------------------------
const ascii = (text) => new TextEncoder().encode(text);

function riffChunk(tag, part, size) {
  const n = new DataView(new ArrayBuffer(4));
  n.setUint32(0, size, true);
  return size % 2 ? [ascii(tag), n, part, new Uint8Array(1)] : [ascii(tag), n, part];
}

export async function sessionFile(id) {
  const rec = await run("sessions", "readonly", (s) => s.get(id));
  const chunks = await run("chunks", "readonly", (s) => s.getAll(chunkRange(id)));
  chunks.sort((a, b) => a.seq - b.seq);
  const pcm = new Blob(chunks.map((c) => c.pcm));
  const head = { id, started: new Date(rec.started).toISOString(), ...rec.meta, ended: rec.ended ?? "unknown" };
  // One event per line: still JSON, and readable with grep.
  const events = chunks.flatMap((c) => c.events).map((e) => JSON.stringify(e)).join(",\n");
  const json = ascii(`${JSON.stringify(head).slice(0, -1)},"events":[\n${events}\n]}`);
  const fmt = new DataView(new ArrayBuffer(16));
  fmt.setUint16(0, 1, true); // PCM
  fmt.setUint16(2, 1, true); // mono
  fmt.setUint32(4, SR, true);
  fmt.setUint32(8, SR * 2, true);
  fmt.setUint16(12, 2, true);
  fmt.setUint16(14, 16, true);
  const body = [ascii("WAVE"), ...riffChunk("fmt ", fmt, 16), ...riffChunk("data", pcm, pcm.size),
    ...riffChunk("json", json, json.byteLength)];
  const size = body.reduce((n, p) => n + (p.size ?? p.byteLength), 0);
  return new File(riffChunk("RIFF", new Blob(body), size), `${id}.wav`, { type: "audio/wav" });
}

// -- recording -----------------------------------------------------------------
export class SessionLog {
  constructor({ enabled = true, upload = false } = {}) {
    Object.assign(this, { enabled, upload, live: null, writes: Promise.resolve() });
    if (!enabled) return;
    // A session's last few seconds are written when the page is hidden or closed.
    addEventListener("visibilitychange", () => {
      this.event(document.hidden ? "hidden" : "visible");
      if (document.hidden) this.flush();
    });
    addEventListener("pagehide", () => this.flush());
  }

  get t() {
    return this.live ? this.live.samples / SR : 0;
  }

  start(meta) {
    if (!this.enabled) return;
    if (this.live) this.end("restarted");
    const now = new Date();
    const stamp = now.toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
    const id = `${stamp}-${Math.random().toString(36).slice(2, 6)}`;
    this.live = { t0: performance.now(), samples: 0, seq: 0, pcm: [], events: [],
      rec: { id, started: now.getTime(), meta, seconds: 0, bytes: 0, sent: false } };
    this.write(() => putSession(this.live.rec));
    this.timer = setInterval(() => this.flush(), FLUSH_MS);
    return id;
  }

  audio(chunk) {
    const s = this.live;
    if (!s) return;
    const pcm = new Int16Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(chunk[i] * 32767)));
    s.pcm.push(pcm);
    s.samples += chunk.length;
  }

  event(type, data = {}) {
    const s = this.live;
    if (!s) return;
    const e = { t: round(s.samples / SR, 2), ms: Math.round(performance.now() - s.t0), type };
    for (const [k, v] of Object.entries(data)) e[k] = typeof v === "number" ? round(v) : v;
    s.events.push(e);
  }

  // Writes run one after another; a failure (quota, private mode) costs that write only.
  write(fn) {
    this.writes = this.writes.then(fn).catch((e) => console.warn("session log:", e));
    return this.writes;
  }

  flush() {
    const s = this.live;
    if (!s || (!s.pcm.length && !s.events.length)) return this.writes;
    const n = s.pcm.reduce((k, p) => k + p.length, 0);
    const pcm = new Int16Array(n);
    let at = 0;
    for (const p of s.pcm) pcm.set(p, (at += p.length) - p.length);
    const chunk = { id: s.rec.id, seq: s.seq++, pcm: pcm.buffer, events: s.events };
    Object.assign(s, { pcm: [], events: [] });
    s.rec.seconds = s.samples / SR;
    s.rec.bytes += pcm.byteLength;
    const rec = { ...s.rec };
    return this.write(async () => {
      await run("chunks", "readwrite", (st) => st.put(chunk));
      await putSession(rec);
    });
  }

  async end(reason) {
    const s = this.live;
    if (!s) return;
    this.event("end", { reason });
    s.rec.ended = reason;
    this.flush();
    clearInterval(this.timer);
    this.live = null;
    await this.writes;
    await this.prune();
    if (this.upload) await this.sendPending();
  }

  async prune() {
    let all = await listSessions().catch(() => []);
    let total = all.reduce((n, r) => n + r.bytes, 0);
    // Sent sessions go first, then the oldest.
    all = [...all.filter((r) => r.sent), ...all.filter((r) => !r.sent)];
    for (const r of all) {
      if (total <= KEEP_BYTES) break;
      if (r.id === this.live?.rec.id) continue;
      await deleteSession(r.id);
      total -= r.bytes;
    }
  }

  // Upload every finished session the server hasn't had yet (a tab killed
  // mid-session leaves one unfinished: it's sent as it stands).
  async sendPending() {
    if (this.sending) return this.sending;
    this.sending = (async () => {
      for (const rec of await listSessions()) {
        if (rec.sent || rec.id === this.live?.rec.id || !rec.bytes) continue;
        const file = await sessionFile(rec.id);
        const r = await fetch(`api/sessions/${file.name}`, { method: "POST", body: file });
        if (!r.ok) throw new Error(`upload ${rec.id}: ${r.status}`);
        await putSession({ ...rec, sent: true });
      }
    })().catch((e) => console.warn("session log:", e)).finally(() => (this.sending = null));
    return this.sending;
  }
}

// Hand sessions over by hand: the share sheet (AirDrop, Files, Drive...) where
// there is one, downloads otherwise. Build the files first (sessionFile): Safari
// only shares straight from a tap, and reading a long session takes a moment.
export async function shareFiles(files) {
  if (navigator.canShare?.({ files })) {
    await navigator.share({ files, title: "Du'a Companion debug sessions" });
  } else {
    for (const f of files) {
      const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(f), download: f.name });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    }
  }
  const ids = files.map((f) => f.name.replace(/\.wav$/, ""));
  for (const rec of await listSessions()) if (ids.includes(rec.id)) await putSession({ ...rec, sent: true });
}
