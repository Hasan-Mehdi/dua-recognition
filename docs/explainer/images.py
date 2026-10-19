#!/usr/bin/env python
"""The explainer's pictures of sound, as PNGs the scenes place (ManimGL draws images from files).

Every picture is the log-mel spectrogram both models read (80 bands every 10 ms, Whisper's own), coloured
from the background's near-black through blue to white. From the reading (reading.py) and data/reading.npz:
- line 2 of Kumayl in his voice and in a second reciter's, and the made-up song Shazam's beat draws;
- the windows the recognition beats show, and a strip of the whole reading;
- one 3 s window as each training augmentation changes it, for real: faster speech, a higher voice (the
  mel bands warped, as VTLP does), noise, a room, a hall through a PA with people talking, SpecAugment's
  blanked stripes.

    python docs/explainer/images.py      # -> data/cache/media/explainer10/img/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "data" / "cache" / "media" / "explainer10"
OUT = CACHE / "img"
STOPS = [(0.0, "#101114"), (0.32, "#14304a"), (0.55, "#256f97"), (0.75, "#58c4dd"), (0.9, "#bfeaf3"), (1.0, "#ffffff")]


def colour(v: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    lo = np.percentile(v, 30) if lo is None else lo
    hi = np.percentile(v, 99.8) if hi is None else hi
    x = np.clip((v - lo) / (hi - lo), 0, 1) ** 1.25
    pos = np.array([p for p, _ in STOPS])
    rgb = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for _, c in STOPS], float)
    out = np.stack([np.interp(x, pos, rgb[:, k]) for k in range(3)], -1)
    return out.astype(np.uint8)


def save(mel: np.ndarray, name: str, lo=None, hi=None, px_per_frame: int = 2, band_px: int = 4) -> None:
    from PIL import Image

    img = colour(mel[::-1], lo, hi)  # low bands at the bottom
    im = Image.fromarray(img).resize((img.shape[1] * px_per_frame, img.shape[0] * band_px), Image.BICUBIC)
    im.save(OUT / f"{name}.png")


def logmel(y: np.ndarray) -> np.ndarray:
    import torch

    from dua_recognition.ctc_student import LogMel

    return LogMel()(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None])[0].numpy()


def peaks(img: np.ndarray, size: tuple = (9, 15), per_chunk: int = 9, chunk: int = 50) -> np.ndarray:
    """Shazam's constellation: points louder than everything near them (9 bands by 150 ms), the strongest few in
    every half second, so they spread over the whole picture. `img` is the unclipped log-mel (bands x frames)."""
    from scipy.ndimage import maximum_filter

    is_max = (img == maximum_filter(img, size=size, mode="constant", cval=-1e9)) & (img > np.percentile(img, 60))
    f, t = np.nonzero(is_max)
    v = img[f, t]
    keep = []
    for c0 in range(0, img.shape[1], chunk):
        idx = np.flatnonzero((t >= c0) & (t < c0 + chunk))
        keep += list(idx[np.argsort(-v[idx])][:per_chunk])
    return np.array(sorted((int(t[i]), int(f[i]), float(v[i])) for i in keep), dtype=np.float32)


def song(sr: int = 16000, secs: float = 6.0) -> np.ndarray:
    """A made-up song for Shazam's pictures (only drawn, never played): a tune over chords, a bass line, drums."""
    rng = np.random.default_rng(3)
    t = np.arange(int(sr * secs)) / sr
    y = np.zeros_like(t)
    hz = lambda n: 440.0 * 2 ** ((n - 69) / 12)  # noqa: E731

    def note(n, a, d, amp, harm=6, vib=0.0):
        on = (t >= a) & (t < a + d)
        tt = t[on] - a
        env = np.minimum(1, tt / 0.02) * np.exp(-2.2 * tt / d)
        f = hz(n) * (1 + vib * np.sin(2 * np.pi * 5.5 * tt))
        for h in range(1, harm + 1):
            y[on] += amp / h ** 1.2 * env * np.sin(2 * np.pi * f * h * tt + h)

    tune = [64, 67, 69, 71, 72, 71, 69, 67, 64, 62, 64, 67, 69, 67, 64, 62]
    for k, n in enumerate(tune):
        note(n, k * 0.375, 0.36, 0.5, vib=0.004)
    for k, ch in enumerate([[57, 60, 64], [55, 59, 62], [53, 57, 60], [55, 59, 64]]):
        for n in ch:
            note(n, k * 1.5, 1.45, 0.12, harm=4)
        note(ch[0] - 12, k * 1.5, 0.7, 0.35, harm=3)
        note(ch[0] - 12, k * 1.5 + 0.75, 0.7, 0.3, harm=3)
    for k in range(int(secs / 0.375)):
        a = k * 0.375
        if k % 2 == 0:
            on = (t >= a) & (t < a + 0.12)
            tt = t[on] - a
            y[on] += 0.6 * np.sin(2 * np.pi * (110 - 300 * tt) * tt) * np.exp(-30 * tt)
        else:
            on = (t >= a) & (t < a + 0.06)
            y[on] += 0.08 * rng.standard_normal(on.sum()) * np.exp(-60 * (t[on] - a))
    return (y / np.abs(y).max() * 0.5).astype(np.float32)


def main() -> None:
    import soundfile as sf
    from scipy.signal import butter, resample_poly, sosfilt

    import bench

    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(HERE / "data" / "reading.npz")
    meta = json.loads((HERE / "data" / "reading.json").read_text(encoding="utf-8"))
    y, sr = sf.read(CACHE / "reading.wav", dtype="float32")
    full = logmel(y)
    lo, hi = np.percentile(full, 30), np.percentile(full, 99.8)

    # line 2 in his voice, and in a second reciter's (DuaPlayer's hand-placed line start), unclipped
    a2, b2 = (float(x) for x in z["line2_span"])
    ma = logmel(y[int(a2 * sr) : int(b2 * sr)])
    second = ROOT / "data" / "duaplayer" / "dua-kumayl" / "38e2b8f0-355e-4e46-e0a4-08da084c84ac"
    s2 = json.loads(second.with_suffix(".json").read_text(encoding="utf-8"))["slide_start_ms"]
    mb = logmel(bench.decode(str(second.with_suffix(".mp3")), s2["2"] / 1000, s2["3"] / 1000))
    e = mb.mean(0)
    on = int(np.argmax(e > np.percentile(e, 20) + 0.5 * (e.max() - np.percentile(e, 20)))) - 15
    mb = mb[:, max(0, on) : max(0, on) + ma.shape[1]]  # the second reciter from his first word, as long as line 2
    ms = logmel(song())
    for name, m in (("line2_a", ma), ("line2_b", mb), ("song", ms)):
        save(m, name)
    # Shazam's points on each picture: (frame, band, value), the frames these pictures show
    np.savez(OUT / "peaks.npz", a=peaks(ma), b=peaks(mb), s=peaks(ms), n_a=ma.shape[1], n_b=mb.shape[1],
             n_s=ms.shape[1])
    # the reading: a strip of its first 32 s, the window Whisper heard Jawshan Kabir's opening in, the next one
    save(full[:, :3200], "reading_0_32", lo, hi, px_per_frame=1)
    save(full[:, 6000:10200], "reading_60_102", lo, hi, px_per_frame=1)
    for name, end in (("win_a", 16.97), ("win_b", 18.74)):
        a = int((end - 6.0) * 100)
        save(full[:, a : a + 600], name, lo, hi)

    # one 3 s window of line 2, as each augmentation changes it
    l2 = [x for x in meta["said"] if x[0] == 2]
    a = l2[0][2] - 0.1
    w = y[int(a * sr) : int((a + 3.0) * sr)]
    rng = np.random.default_rng(4)
    variants = {"clean": w}
    variants["faster"] = resample_poly(np.r_[w, y[int((a + 3) * sr) : int((a + 3.6) * sr)]], 20, 23)[: len(w)]
    variants["noise"] = bench.add_noise(w, bench.colored_noise(len(w), rng, "pink"), 4.0)
    variants["room"] = bench.reverb(w, 0.7, 2.0, rng)[: len(w)]
    band = sosfilt(butter(4, [250, 3800], "bandpass", fs=sr, output="sos"), w)
    pa = np.tanh(3.0 * band / np.abs(band).max()) * 0.5
    for d, g in ((0.035, 0.6), (0.08, 0.4)):
        k = int(d * sr)
        pa = pa + g * np.r_[np.zeros(k, np.float32), pa[:-k]]
    pa = bench.reverb(pa.astype(np.float32), 1.4, -2.0, rng)[: len(w)]
    talk = bench._talk_clips(40)
    crowd = np.zeros_like(w)
    for c in talk[:3]:
        x = np.load(ROOT / c["clip"]).astype(np.float32)
        x = x / 32768.0 if np.abs(x).max() > 2.0 else x
        crowd += np.r_[x, np.zeros(len(w))][: len(w)]
    variants["hall"] = bench.add_noise(pa.astype(np.float32), crowd, 8.0)
    mels = {k: logmel(v.astype(np.float32)) for k, v in variants.items()}
    # a higher voice: the bands warped upwards by 12%, as VTLP warps the spectrum
    m = mels["clean"]
    src = np.clip(np.arange(m.shape[0]) / 1.12, 0, m.shape[0] - 1)
    mels["higher"] = np.stack([np.interp(src, np.arange(m.shape[0]), m[:, t]) for t in range(m.shape[1])], 1)
    m = mels["clean"].copy()
    floor = m.min()
    for t0 in (70, 190):
        m[:, t0 : t0 + 22] = floor
    m[28:38] = floor
    mels["masked"] = m
    clo, chi = np.percentile(mels["clean"], 30), np.percentile(mels["clean"], 99.8)
    for k, v in mels.items():
        save(v[:, :300], f"aug_{k}", clo, chi)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
