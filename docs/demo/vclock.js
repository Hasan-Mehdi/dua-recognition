// A virtual clock for frame-exact capture (record.mjs injects this into every frame).
//
// Timers, requestAnimationFrame, performance.now/Date and CSS animations/transitions all
// follow one clock that only moves when the recorder calls __vt.step(ms), so every frame
// is captured exactly 1/fps apart however long a screenshot takes. The stage page owns the
// clock; the app's iframe shares it (same origin).
//
// In the app's frame it also stands in for the two things a replay can't have live:
//   WebSocket     the /ws stream: the stage delivers the recognizer's recorded messages
//   AnalyserNode  the microphone level behind the listening star: the recording's own
(() => {
  const shared = window !== window.parent && window.parent.__vt;
  const vt = shared || {
    now: 0,
    dateBase: window.__vtStart ?? Date.now(), // record.mjs picks the wall-clock date the page sees
    queue: [], // {at, fn, args, id, every}
    rafs: [],
    nextId: 1,
    docs: [],
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
    // CSS animations and transitions: paused, and set to where the clock says they are.
    animate() {
      vt.docs = vt.docs.filter((d) => d.defaultView);
      for (const d of vt.docs) {
        for (const a of d.getAnimations()) {
          if (!vt.seen.has(a)) {
            vt.seen.set(a, vt.now);
            a.pause();
          }
          const t = vt.now - vt.seen.get(a);
          const end = a.effect?.getComputedTiming().endTime;
          if (Number.isFinite(end) && t >= end) a.finish();
          else a.currentTime = t;
        }
      }
    },
  };
  window.__vt = vt;
  vt.docs.push(document);

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
  window.WebSocket = class {
    static OPEN = 1;
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      setTimeout(() => {
        this.readyState = 1;
        this.onopen?.({});
        if (/\/ws(\?|$)/.test(url)) demo()?.connected(this);
      }, 5);
    }
    send() {}
    close() {
      this.readyState = 3;
      this.onclose?.({});
    }
  };
  const real = AnalyserNode.prototype.getFloatFrequencyData;
  AnalyserNode.prototype.getFloatFrequencyData = function (out) {
    const db = demo()?.bandDb();
    if (db == null) return real.call(this, out);
    // Spread the level over voice.js's band so voiceBandDb reads it back unchanged.
    const hz = this.context.sampleRate / this.fftSize;
    const lo = Math.max(1, Math.round(100 / hz));
    const hi = Math.min(out.length - 1, Math.round(4000 / hz));
    out.fill(-160);
    out.fill(db - 10 * Math.log10(hi - lo + 1), lo, hi + 1);
  };
})();
