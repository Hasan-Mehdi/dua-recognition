// The story film's own sounds, synthesized (nothing recorded, nothing to license): the night
// outside the mosque (crickets, a little air), and small sounds for what happens on screen: a
// page turning, a book closing, footsteps on the path, a finger on the glass, the cat purring.
//
//   node docs/demo/story-sounds.mjs        # -> data/cache/media/story/*.wav (48 kHz)
//
// film.mjs --stage story.html --bed .../night.wav --sfx .../ lays them under the recording, at
// the gains and times story.html gives.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = join(resolve(dirname(fileURLToPath(import.meta.url)), "..", ".."), "data", "cache", "media", "story");
const SR = 48000;
mkdirSync(OUT, { recursive: true });

let seed = 12345; // the same sounds every time
const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
const noise = () => rnd() * 2 - 1;

function wav(name, left, right = left) {
  const n = left.length, buf = Buffer.alloc(44 + n * 4);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + n * 4, 4);
  buf.write("WAVEfmt ", 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(2, 22);
  buf.writeUInt32LE(SR, 24);
  buf.writeUInt32LE(SR * 4, 28);
  buf.writeUInt16LE(4, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(n * 4, 40);
  const c = (x) => Math.round(Math.max(-1, Math.min(1, x)) * 32767);
  for (let i = 0; i < n; i++) {
    buf.writeInt16LE(c(left[i]), 44 + i * 4);
    buf.writeInt16LE(c(right[i]), 46 + i * 4);
  }
  writeFileSync(join(OUT, `${name}.wav`), buf);
  console.log(join(OUT, `${name}.wav`));
}
// One-pole filters, run in place.
function lowpass(x, hz) {
  const a = 1 - Math.exp((-2 * Math.PI * hz) / SR);
  let y = 0;
  for (let i = 0; i < x.length; i++) x[i] = y += a * (x[i] - y);
  return x;
}
function highpass(x, hz) {
  const a = Math.exp((-2 * Math.PI * hz) / SR);
  let px = 0, y = 0;
  for (let i = 0; i < x.length; i++) {
    y = a * (y + x[i] - px);
    px = x[i];
    x[i] = y;
  }
  return x;
}
const band = (x, lo, hi) => lowpass(lowpass(highpass(highpass(x, lo), lo), hi), hi);

// -- the night: a few crickets, each chirping at its own pitch and pace, and some air ----------
{
  const seconds = 40, n = seconds * SR;
  const L = new Float32Array(n), R = new Float32Array(n);
  const crickets = [
    { f: 4450, rate: 31, pulses: 3, period: 0.62, amp: 0.05, pan: -0.6 },
    { f: 4800, rate: 28, pulses: 4, period: 0.81, amp: 0.035, pan: 0.5 },
    { f: 5150, rate: 34, pulses: 3, period: 0.55, amp: 0.022, pan: 0.85 },
    { f: 4100, rate: 30, pulses: 2, period: 0.93, amp: 0.028, pan: -0.15 },
    { f: 5500, rate: 36, pulses: 3, period: 0.7, amp: 0.014, pan: -0.9 },
  ];
  for (const c of crickets) {
    const gl = Math.cos(((c.pan + 1) * Math.PI) / 4), gr = Math.sin(((c.pan + 1) * Math.PI) / 4);
    for (let t = rnd() * c.period; t < seconds - 1; t += c.period * (0.88 + 0.24 * rnd())) {
      for (let p = 0; p < c.pulses; p++) {
        const at = Math.floor((t + p / c.rate) * SR), len = Math.floor(0.013 * SR);
        const f = c.f * (1 + (rnd() - 0.5) * 0.01);
        for (let i = 0; i < len; i++) {
          const w = Math.sin((Math.PI * i) / len) ** 2;
          const x = Math.sin((2 * Math.PI * f * i) / SR) * w * c.amp * (0.85 + 0.15 * rnd());
          L[at + i] += x * gl;
          R[at + i] += x * gr;
        }
      }
    }
  }
  // Air: soft noise, lowpassed, swelling slowly.
  const airL = band(Float32Array.from({ length: n }, noise), 60, 420);
  const airR = band(Float32Array.from({ length: n }, noise), 60, 420);
  for (let i = 0; i < n; i++) {
    const swell = 0.7 + 0.3 * Math.sin((2 * Math.PI * i) / SR / 9.5);
    L[i] += airL[i] * 0.09 * swell;
    R[i] += airR[i] * 0.09 * swell;
  }
  // The loop's seam: the last second fades into the first.
  for (let i = 0; i < SR; i++) {
    const k = i / SR;
    L[i] = L[i] * k + L[n - SR + i] * (1 - k);
    R[i] = R[i] * k + R[n - SR + i] * (1 - k);
  }
  wav("night", L.subarray(0, n - SR), R.subarray(0, n - SR));
}

// -- a page turning: a dry rustle and the page settling -----------------------------------------
{
  const n = Math.floor(0.42 * SR);
  const x = band(Float32Array.from({ length: n }, noise), 900, 7000);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const swish = Math.min(1, t / 0.03) * Math.exp(-t / 0.09) * (0.7 + 0.3 * Math.sin(2 * Math.PI * 38 * t));
    const land = t > 0.16 ? Math.exp(-(t - 0.16) / 0.03) * 0.5 : 0;
    x[i] *= (swish + land) * 0.9;
  }
  wav("page", x);
}
// -- a small book closing -------------------------------------------------------------------------
{
  const n = Math.floor(0.3 * SR);
  const x = band(Float32Array.from({ length: n }, noise), 120, 1800);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    x[i] = x[i] * Math.exp(-t / 0.035) * 1.6 + Math.sin(2 * Math.PI * 110 * t) * Math.exp(-t / 0.05) * 0.5;
  }
  wav("thud", x);
}
// -- a footstep on a stone path ------------------------------------------------------------------
{
  const n = Math.floor(0.2 * SR);
  const x = band(Float32Array.from({ length: n }, noise), 150, 2400);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    x[i] = x[i] * Math.exp(-t / 0.028) * 1.2 + Math.sin(2 * Math.PI * 85 * t) * Math.exp(-t / 0.04) * 0.4;
  }
  wav("step", x);
}
// -- a fingertip on glass -------------------------------------------------------------------------
{
  const n = Math.floor(0.08 * SR);
  const x = band(Float32Array.from({ length: n }, noise), 1500, 9000);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    x[i] = x[i] * Math.exp(-t / 0.004) * 0.9 + Math.sin(2 * Math.PI * 1900 * t) * Math.exp(-t / 0.012) * 0.25;
  }
  wav("tap", x);
}
// -- a cat purring: a low rumble, breathing in and out ------------------------------------------
{
  const n = Math.floor(3.2 * SR);
  const x = band(Float32Array.from({ length: n }, noise), 40, 900);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const pulse = 0.55 + 0.45 * Math.sin(2 * Math.PI * 26 * t); // the purr's flutter
    const breath = 0.6 + 0.4 * Math.sin(2 * Math.PI * 0.55 * t - 1.2);
    const env = Math.min(1, t / 0.4) * Math.min(1, (3.2 - t) / 0.6);
    x[i] *= pulse * breath * env * 1.4;
  }
  wav("purr", x);
}
