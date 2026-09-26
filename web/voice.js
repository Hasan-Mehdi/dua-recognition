// How loudly someone is reciting, as a calm 0..1 value for animation.
//
// Microphones differ by 30 dB or more (a phone held close vs a laptop across a
// room), so there are no fixed thresholds: the meter learns the room's quiet
// (floor) and the voice's loud (peak) and places each moment between them.
// The shaping follows what makes voice orbs feel calm rather than twitchy:
// a fast attack (60 ms) and slow release (320 ms), so it rises with each
// phrase instead of chattering on every syllable, all timed in seconds so
// 60 Hz and 120 Hz screens behave alike.

const ease = (dt, tau) => 1 - Math.exp(-dt / tau);
const SPAN = 18; // dB: the least distance kept between floor and peak

export class VoiceLevel {
  floor = null; // dB
  peak = null; // dB
  level = 0; // 0..1

  update(db, dt) {
    if (this.floor === null) {
      this.floor = db;
      this.peak = db + SPAN;
    }
    // Floor: drops to quiet quickly, creeps up slowly (a steady fan becomes the new quiet).
    this.floor += (db - this.floor) * ease(dt, db < this.floor ? 0.25 : 6);
    // Peak: jumps to loud quickly, relaxes slowly (a softer voice is still read as a voice).
    this.peak += (db - this.peak) * ease(dt, db > this.peak ? 0.05 : 5);
    // A long phrase must not become the "floor", nor silence the "peak".
    this.floor = Math.min(this.floor, this.peak - SPAN);
    this.peak = Math.max(this.peak, this.floor + SPAN);

    const gate = this.floor + 4; // just above the room
    const raw = Math.min(1, Math.max(0, (db - gate) / (this.peak - gate)));
    const target = raw * raw * (3 - 2 * raw); // smoothstep: soft at both ends
    this.level += (target - this.level) * ease(dt, target > this.level ? 0.06 : 0.32);
    return this.level;
  }
}

// Energy in the voice band (about 100 Hz to 4 kHz) from an analyser's spectrum, in dB.
// Measuring the spectrum rather than the raw waveform leaves out hum and rumble.
export function voiceBandDb(spectrum, sampleRate, fftSize) {
  const hz = sampleRate / fftSize;
  const lo = Math.max(1, Math.round(100 / hz));
  const hi = Math.min(spectrum.length - 1, Math.round(4000 / hz));
  let power = 0;
  for (let i = lo; i <= hi; i++) power += 10 ** (spectrum[i] / 10);
  return 10 * Math.log10(power + 1e-12);
}
