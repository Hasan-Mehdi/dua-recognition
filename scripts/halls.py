"""A phone in a masjid hall, listening to the reciter through the PA: effects for training and
for the noise bench (scripts/noise_bench.py), and WPE dereverberation as a front end.

The models' training so far had one kind of room (transcribe_windows.room: a 0.5 s noise tail
with the direct sound on top). A hall listened to through a PA differs in four ways:

- the reverberation is long (1-3 s) and the direct sound is weak once the phone is a few
  metres from a loudspeaker (direct-to-reverberant ratio well below 0 dB);
- several loudspeakers carry the same voice, each arriving at its own delay (5-60 ms apart);
- the PA itself: a loudspeaker's band (no lows, no highs), a compressor, some saturation;
- people: talk, responses, children, a fan or the AC, all in the same hall.

hall_aug() draws all of these at random for training; the noise bench uses measured rooms
(OpenSLR 28, the real RIRs) so what is scored is not what was trained on.
"""
from __future__ import annotations

import math

import numpy as np

SR = 16000


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f = np.fft.rfftfreq(x.size, 1 / SR)
    s = np.fft.rfft(x)
    s[(f < lo) | (f >= hi)] = 0
    return np.fft.irfft(s, x.size)


def hall_rir(rng: np.random.Generator, rt60: float, drr_db: float, speakers: int = 1,
             spread_ms: tuple[float, float] = (5.0, 60.0)) -> np.ndarray:
    """A hall's impulse response heard from one seat: one direct arrival per loudspeaker (the
    nearest at t=0, so labels stay aligned), a few early reflections after each, and a late
    tail that decays faster in the highs (RT60 x1.15 below 500 Hz, x0.65 above 2.5 kHz), scaled
    so that the direct arrivals over everything else is drr_db."""
    n = int(min(rt60 * 1.1, 3.5) * SR)
    t = np.arange(n) / SR
    tail = np.zeros(n)
    for lo, hi, k in ((0, 500, 1.15), (500, 2500, 1.0), (2500, SR / 2, 0.65)):
        tail += _band(rng.standard_normal(n), lo, hi) * np.exp(-6.9 * t / (rt60 * k))
    onset = int(rng.uniform(0.005, 0.03) * SR)  # the tail builds up after the first reflections
    tail[:onset] *= np.linspace(0, 1, onset) ** 2
    direct = np.zeros(n)
    early = np.zeros(n)
    for s in range(speakers):
        d = 0 if s == 0 else int(rng.uniform(*spread_ms) / 1000 * SR)
        g = 1.0 if s == 0 else rng.uniform(0.3, 0.9)
        if d >= n:
            continue
        direct[d] += g
        for _ in range(int(rng.integers(6, 16))):
            e = d + int(rng.uniform(0.002, 0.08) * SR)
            if e < n:
                early[e] += g * rng.uniform(0.2, 0.7) * math.exp(-6.9 * (e - d) / SR / rt60) * rng.choice([-1, 1])
    tail *= math.sqrt((np.sum(early ** 2) + 1e-3) * 3.0 / np.sum(tail ** 2))  # late energy 3x the early
    rest = early + tail
    e_dir, e_rest = np.sum(direct ** 2), np.sum(rest ** 2)
    rest *= math.sqrt(e_dir / e_rest / 10 ** (drr_db / 10))
    h = direct + rest
    return (h / np.abs(h).max()).astype(np.float32)


def convolve(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    out = fftconvolve(y, h)[: y.size]
    p_in = float(np.mean(y.astype(np.float64) ** 2)) + 1e-12
    p_out = float(np.mean(out ** 2)) + 1e-12
    return (out * math.sqrt(p_in / p_out)).astype(np.float32)


def pa(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A PA chain: the loudspeaker's band, a peak or dip somewhere in the mids, a compressor
    (ratio 2-6 above a threshold 10-25 dB under the peak level) and soft saturation."""
    from scipy.signal import butter, sosfilt

    lo, hi = rng.uniform(90, 220), rng.uniform(4500, 7600)
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)
    fc, gain = rng.uniform(800, 4000), rng.uniform(-6, 6)
    peak = _band(y, fc / 1.4, fc * 1.4)
    y = y + peak * (10 ** (gain / 20) - 1)
    hop = 160
    n = y.size // hop * hop
    env = np.sqrt(np.mean(y[:n].reshape(-1, hop) ** 2, axis=1) + 1e-10)
    env_db = 20 * np.log10(env)
    sm = np.empty_like(env_db)
    a_att, a_rel, cur = 0.5, 0.92, env_db[0] if env_db.size else -100.0
    for i, v in enumerate(env_db):  # fast attack, slower release
        cur = a_att * cur + (1 - a_att) * v if v > cur else a_rel * cur + (1 - a_rel) * v
        sm[i] = cur
    thr = np.percentile(env_db, 95) - rng.uniform(10, 25)
    ratio = rng.uniform(2, 6)
    red = np.where(sm > thr, (sm - thr) * (1 - 1 / ratio), 0.0)
    g = np.repeat(10 ** (-red / 20), hop)
    y = y.copy()
    y[:n] *= g
    drive = rng.uniform(1.0, 3.0)
    pk = np.abs(y).max() + 1e-9
    y = np.tanh(drive * y / pk) / np.tanh(drive) * pk
    return y.astype(np.float32)


def crowd_noise(n: int, rng: np.random.Generator, voices: int | None = None) -> np.ndarray:
    """Speech-like noise without speech data: speech-shaped noise in several streams, each
    switched on and off at syllable rate, plus a little fan hum."""
    voices = voices or int(rng.integers(4, 12))
    f = np.fft.rfftfreq(n, 1 / SR)
    shape = 1 / np.sqrt(np.maximum(f, 50.0)) * (f > 120) * np.exp(-f / 4000)
    # each voice's on/off envelope at 100 Hz (a 50 ms ramp), then one spectral shaping for all:
    # the envelopes change slowly enough that shaping the sum is shaping each
    n100 = n // 160 + 2
    env = np.zeros((voices, n100))
    for v in range(voices):
        m = int(n / SR * rng.uniform(2.0, 6.0)) + 2
        steps = (rng.random(m) < 0.6) * rng.uniform(0.3, 1.0, m)
        e = np.interp(np.arange(n100), np.linspace(0, n100, m), steps)
        env[v] = np.convolve(e, np.ones(5) / 5, mode="same")
    t100 = np.arange(n) / 160
    w = rng.standard_normal((voices, n)) * np.stack([np.interp(t100, np.arange(n100), e) for e in env])
    out = np.fft.irfft(np.fft.rfft(w.sum(axis=0)) * shape, n)
    hum = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.maximum(f, 20.0), n)
    out = out / (out.std() + 1e-9) + 0.3 * hum / (hum.std() + 1e-9)
    return out.astype(np.float32)


_VOICES: list[str] = []


def crowd_voices(n: int, rng: np.random.Generator, voices: int | None = None) -> np.ndarray:
    """People talking: 3-8 speakers from Common Voice's training side (data/cache/finetune/cv_ar.jsonl;
    the bench's talking cells use cv_ar_test), each saying a clip or two at random times, some at
    once, some quiet."""
    if not _VOICES:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        rows = (root / "data" / "cache" / "finetune" / "cv_ar.jsonl").read_text(encoding="utf-8").splitlines()
        _VOICES.extend(str(root / json.loads(r)["clip"]) for r in rows if r)
    out = np.zeros(n, np.float32)
    for _ in range(voices or int(rng.integers(3, 9))):
        x = np.load(_VOICES[int(rng.integers(len(_VOICES)))]).astype(np.float32)
        x /= np.sqrt(np.mean(x ** 2)) + 1e-9
        for _ in range(max(1, int(round(n / max(x.size, 1) * rng.uniform(0.3, 1.0))))):
            at = int(rng.integers(-x.size // 2, n))
            a0, b0 = max(0, at), min(n, at + x.size)
            if b0 > a0:
                out[a0:b0] += x[a0 - at : b0 - at] * rng.uniform(0.4, 1.0)
    return out


_BANK: dict = {"hall": [], "crowd": []}
BANK_SIZE = 300


def _banked(kind: str, g: np.random.Generator, make):
    """A response from a bank that fills up to BANK_SIZE, then is drawn from (per process)."""
    bank = _BANK[kind]
    if len(bank) < BANK_SIZE:
        bank.append(make())
        return bank[-1]
    return bank[int(g.integers(len(bank)))]


def add_at_snr(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    act = np.abs(y) > 0.1 * np.abs(y).max()
    p_s = float(np.mean(y[act].astype(np.float64) ** 2)) if act.any() else float(np.mean(y ** 2))
    p_n = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    return (y + noise * math.sqrt(p_s / p_n / 10 ** (snr_db / 10))).astype(np.float32)


def hall_aug(x: np.ndarray, rng, voices: float = 0.0) -> np.ndarray:
    """Training: a random hall, seat, PA and crowd. rng: random.Random or np Generator. voices: the
    share of crowds that are real people talking (crowd_voices) rather than speech-shaped noise."""
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng.randrange(1 << 30))
    if not np.any(x):
        return x
    def make():
        rt60 = float(np.exp(g.uniform(math.log(0.4), math.log(3.0))))
        return hall_rir(g, rt60, g.uniform(-10, 6), int(g.choice([1, 1, 2, 3, 4]))), hall_rir(g, rt60, -20.0)

    h, h_crowd = _banked("hall", g, make)
    y = pa(x, g) if g.random() < 0.6 else x.astype(np.float32)
    y = convolve(y, h)
    if g.random() < 0.8:
        nz = crowd_voices(y.size, g) if voices and g.random() < voices else crowd_noise(y.size, g)
        nz = convolve(nz, h_crowd)  # the crowd is in the same hall, all around
        y = add_at_snr(y, nz, g.uniform(5, 25))
    pk = np.abs(y).max() + 1e-9
    return np.clip(y / pk * g.uniform(0.3, 0.9), -1, 1).astype(np.float32)


def wpe_online(y: np.ndarray, taps: int = 20, delay: int = 2, alpha: float = 0.9995, n_fft: int = 512,
               hop: int = 256) -> np.ndarray:
    """Causal WPE (recursive least squares, as nara_wpe's OnlineWPE; Caroselli et al. 2017): each
    STFT frame is cleaned as it arrives, from the frames before it only, so a phone could run it
    on the live microphone. The prediction filter forgets with `alpha` per frame (0.9995 at 16 ms
    hops: ~30 s of memory)."""
    from scipy.signal import istft, stft

    _, _, Y = stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
    F, T = Y.shape
    K = taps
    Rinv = np.tile(np.eye(K, dtype=Y.dtype)[None], (F, 1, 1))
    G = np.zeros((F, K), dtype=Y.dtype)
    buf = np.zeros((F, delay - 1 + K), dtype=Y.dtype)  # buf[:, j] = Y[:, t - 1 - j]
    X = np.empty_like(Y)
    for t in range(T):
        yt = Y[:, t]
        yk = buf[:, delay - 1 :]  # Y[t - delay - k], k = 0..K-1
        x = yt - np.einsum("fk,fk->f", G.conj(), yk)
        lam = np.maximum((np.abs(yt) ** 2 + np.sum(np.abs(buf[:, : delay - 1]) ** 2, axis=1)) / delay, 1e-10)
        ry = np.einsum("fij,fj->fi", Rinv, yk)
        den = alpha * lam + np.einsum("fi,fi->f", yk.conj(), ry).real
        k = ry / den[:, None]
        Rinv = (Rinv - k[:, :, None] * ry.conj()[:, None, :]) / alpha
        G = G + k * x.conj()[:, None]
        X[:, t] = x
        buf = np.roll(buf, 1, axis=1)
        buf[:, 0] = yt
    _, out = istft(X, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
    out = out[: y.size]
    if out.size < y.size:
        out = np.r_[out, np.zeros(y.size - out.size)]
    return out.astype(np.float32)


def wpe(y: np.ndarray, taps: int = 10, delay: int = 3, iterations: int = 3, n_fft: int = 512,
        hop: int = 128) -> np.ndarray:
    """Single-channel WPE dereverberation (Nakatani et al. 2010; as in nara_wpe): per frequency,
    predict the late reverberation from `taps` STFT frames starting `delay` frames back and
    subtract it, weighting frames by the estimated clean power. Offline (the whole recording at
    once): what dereverberation can do at best; a phone would need the online (RLS) form."""
    from scipy.signal import istft, stft

    _, _, Y = stft(y, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
    F, T = Y.shape
    if T <= delay + taps + 1:
        return y
    X = np.empty_like(Y)
    eye = np.eye(taps)
    for f in range(F):
        yf = Y[f]
        Yt = np.zeros((taps, T), dtype=Y.dtype)  # Yt[k, t] = Y[f, t - delay - k]
        for k in range(taps):
            s = delay + k
            Yt[k, s:] = yf[: T - s]
        xf = yf
        for _ in range(iterations):
            lam = np.maximum(np.abs(xf) ** 2, 1e-10)
            Yw = Yt / lam
            R = Yw @ Yt.conj().T
            P = Yw @ yf.conj()
            R += (1e-6 * np.trace(R).real / taps + 1e-12) * eye
            g = np.linalg.solve(R, P)
            xf = yf - g.conj() @ Yt
        X[f] = xf
    _, x = istft(X, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
    x = x[: y.size]
    if x.size < y.size:
        x = np.r_[x, np.zeros(y.size - x.size)]
    return x.astype(np.float32)
