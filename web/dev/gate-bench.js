// Real-time replay through the page's ASR worker (../asr-worker.js): ~250 ms chunks at
// capture pace, one 6 s window per 1 s of new audio, a hop skipped while the worker is
// busy (DeviceEngine._maybeSend's rule, repeated here because app.js runs on import).
// Per hop: gate decision, skip reason, inference time, and evidence age = when the
// result could be painted minus when the window's last sample was "captured".
const params = new URLSearchParams(location.search);
const SR = 16000;
const CHUNK = 4000;
const clock = () => performance.timeOrigin + performance.now();
const out = document.getElementById("out");

async function main() {
  const gate = params.get("gate") || "legacy";
  const seconds = Number(params.get("seconds") || 60);
  const buf = await fetch(params.get("audio") || "/bench-audio.f32").then((r) => r.arrayBuffer());
  const y = new Float32Array(buf).subarray(0, seconds * SR);
  const worker = new Worker("../asr-worker.js", { type: "module" });
  const t0 = clock();
  await new Promise((ok, err) => {
    worker.onmessage = ({ data }) => (data.type === "ready" ? ok() : data.type === "error" ? err(data.message) : 0);
    worker.postMessage({ type: "load", model: params.get("model") || "whisper-base-aug-v4" });
  });
  const loadMs = clock() - t0;
  const W = 6 * SR;
  const win = new Float32Array(W);
  let filled = 0, total = 0, lastSent = 0, busy = false, sentAt = 0;
  const marks = [];
  const hops = [];
  let skippedBusy = 0;
  const capturedAt = (id) => {
    const m = marks.find(([n]) => n >= id);
    return m ? m[1] - ((m[0] - id) / SR) * 1000 : null;
  };
  const done = new Promise((finish) => {
    worker.onmessage = ({ data }) => {
      if (data.type !== "text") return;
      busy = false;
      const captured = capturedAt(data.id);
      requestAnimationFrame(() => {
        const shown = clock();
        hops.push({ end: data.id / SR, gate: data.gate?.policy, ran: data.gate?.run ?? null, skip: data.skip,
          reason: data.gate?.reason ?? null, level: data.gate?.level, infer_ms: data.infer_ms, ms: data.ms,
          queue_ms: data.recv - sentAt, age_ms: shown - captured, text: data.text, cold: hops.length === 0 });
        if (total >= y.length && !busy) finish();
      });
    };
  });
  const maybeSend = () => {
    if (busy || total - lastSent < SR) return;
    if (total - lastSent >= 2 * SR) skippedBusy += Math.floor((total - lastSent) / SR) - 1;
    busy = true;
    lastSent = total;
    sentAt = clock();
    const audio = win.slice(W - filled);
    worker.postMessage({ type: "transcribe", id: total, audio, gate }, [audio.buffer]);
  };
  // capture pace: one chunk every 250 ms of wall time
  for (let i = 0; i < y.length; i += CHUNK) {
    const c = y.subarray(i, i + CHUNK);
    win.copyWithin(0, c.length);
    win.set(c, W - c.length);
    filled = Math.min(W, filled + c.length);
    total += c.length;
    marks.push([total, clock()]);
    if (marks.length > 64) marks.shift();
    maybeSend();
    await new Promise((r) => setTimeout(r, (c.length / SR) * 1000));
  }
  const settle = setInterval(maybeSend, 50);
  await Promise.race([done, new Promise((r) => setTimeout(r, 30000))]);
  clearInterval(settle);
  const result = { gate, seconds, load_ms: loadMs, hops, skipped_busy: skippedBusy, ua: navigator.userAgent,
    threads: navigator.hardwareConcurrency };
  window.__result = result;
  out.textContent = JSON.stringify({ ...result, hops: hops.length }, null, 1);
}

main().catch((e) => {
  window.__result = { error: String(e) };
  out.textContent = String(e);
});
