// A virtual clock for frame-exact capture (film.mjs injects this into every frame).
//
// Timers, requestAnimationFrame, performance.now/Date and CSS animations/transitions all
// follow one clock that only moves when the recorder calls __vt.step(ms), so every frame
// is captured exactly 1/fps apart however long a screenshot takes. The stage page owns the
// clock; the app's iframe shares it (same origin).
//
// In the app's frame it also stands in for what a replay can't have live:
//   AnalyserNode      the microphone level behind the listening star: the recording's own
//   AudioWorkletNode  the microphone's chunks never reach the page: the engine is the stage's
//                     replay of the captured run (film.mjs), and real-time audio under a
//                     virtual clock would look like a microphone at the wrong speed (checkClock)
(() => {
  const shared = window !== window.parent && window.parent.__vt;
  const vt = shared || {
    now: 0,
    dateBase: window.__vtStart ?? Date.now(), // film.mjs picks the wall-clock date the page sees
    queue: [], // {at, fn, args, id, every}
    rafs: [],
    nextId: 1,
    top: document, // the stage: its frames are the phones
    seen: new WeakMap(),
    async step(ms) {
      const until = vt.now + ms;
      for (;;) {
        vt.queue.sort((a, b) => a.at - b.at || a.id - b.id);
        const t = vt.queue[0];
        if (!t || t.at > until) break;
        vt.queue.shift();
        vt.now = Math.max(vt.now, t.at);
        if (t.every != null) vt.queue.push({ ...t, at: vt.now + Math.max(1, t.every) });
        try {
          t.fn(...t.args);
        } catch (e) {
          console.error(e);
        }
        await Promise.resolve(); // let promise callbacks run between timers
      }
      vt.now = until;
      const rafs = vt.rafs.splice(0);
      for (const r of rafs) {
        try {
          r.fn(vt.now);
        } catch (e) {
          console.error(e);
        }
      }
      await Promise.resolve();
      vt.animate();
    },
    // CSS animations and transitions: paused, and set to where the clock says they are. The
    // documents are looked up afresh every step (the stage's, and whatever each frame holds now)
    // and every animation is paused again: in one long film two phones' listening stars ran on
    // the real clock (a 5 s cycle in a quarter of a second of film), which neither can now.
    unpaused: 0,
    animate() {
      const docs = [vt.top, ...[...vt.top.querySelectorAll("iframe")].map((f) => f.contentDocument).filter(Boolean)];
      for (const d of docs) {
        for (const a of d.getAnimations()) {
          if (!vt.seen.has(a)) vt.seen.set(a, vt.now);
          else if (a.playState === "running") vt.unpaused++;
          if (a.playState !== "paused" && a.playState !== "finished") a.pause();
          const t = vt.now - vt.seen.get(a);
          const end = a.effect?.getComputedTiming().endTime;
          if (Number.isFinite(end) && t >= end) a.finish();
          else a.currentTime = t;
        }
      }
    },
  };
  window.__vt = vt;

  const now = () => vt.now;
  performance.now = now;
  const RealDate = Date;
  class VDate extends RealDate {
    constructor(...a) {
      super(...(a.length ? a : [vt.dateBase + vt.now]));
    }
    static now() {
      return vt.dateBase + vt.now;
    }
  }
  window.Date = VDate;
  const add = (fn, ms, args, every) => {
    const id = vt.nextId++;
    if (typeof fn === "function") vt.queue.push({ at: vt.now + Math.max(0, Number(ms) || 0), fn, args, id, every });
    return id;
  };
  const clear = (id) => {
    vt.queue = vt.queue.filter((t) => t.id !== id);
  };
  window.setTimeout = (fn, ms, ...args) => add(fn, ms, args, null);
  window.setInterval = (fn, ms, ...args) => add(fn, ms, args, Number(ms) || 0);
  window.clearTimeout = window.clearInterval = clear;
  window.requestAnimationFrame = (fn) => {
    const id = vt.nextId++;
    vt.rafs.push({ fn, id });
    return id;
  };
  window.cancelAnimationFrame = (id) => {
    vt.rafs = vt.rafs.filter((r) => r.id !== id);
  };

  if (!shared) return;

  // -- the app's frame ------------------------------------------------------------
  const demo = () => window.parent.__demo;
  window.AudioWorkletNode = class extends AudioWorkletNode {
    constructor(...a) {
      super(...a);
      Object.defineProperty(this.port, "onmessage", { set() {}, get: () => null, configurable: true });
    }
  };
  const real = AnalyserNode.prototype.getFloatFrequencyData;
  AnalyserNode.prototype.getFloatFrequencyData = function (out) {
    const db = demo()?.bandDb(window.frameElement);
    if (db == null) return real.call(this, out);
    // Spread the level over voice.js's band so voiceBandDb reads it back unchanged.
    const hz = this.context.sampleRate / this.fftSize;
    const lo = Math.max(1, Math.round(100 / hz));
    const hi = Math.min(out.length - 1, Math.round(4000 / hz));
    out.fill(-160);
    out.fill(db - 10 * Math.log10(hi - lo + 1), lo, hi + 1);
  };
})();
