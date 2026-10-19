#!/usr/bin/env python
"""Everything the explainer's pictures are drawn from, replayed from the page's own code on the reading
(reading.py) and written to docs/explainer/data/ (numbers only, no audio).

- the tracker at a phone's pace (Whisper every 2 s, shown 1.8 s late), as `bench.py score` drives it: per
  update what Whisper heard, the du'as' probabilities, the share on "not in the texts", the du'a named;
- the stream decoder over the streaming CTC student's frames, anchored on the tracker: per 0.1 s step the
  probability on each of Kumayl's first lines, the share off the text, the word shown;
- the letter model's frames (likeliest letter per 20 ms), the reading's loudness, log-mel pictures of
  line 2 in his voice and in a second reciter's, the peaks of those pictures, a made-up song's picture and
  peaks for Shazam (never heard);
- how far the window that hears "allahumma inni as'aluka bi-rahmatika" is from every word of the 522 texts,
  and how many texts read each longer opening of line 2;
- the scenario bench's test grid (on the line, per way of reading and kind of voice).

    python docs/explainer/dump.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DUA_BENCH_ASR_EVERY", "2")

CACHE = ROOT / "data" / "cache" / "media" / "explainer10"
DATA = HERE / "data"
DELAY, F_DELAY, EVERY = 1.8, 0.1, 2
LINES = 12  # Kumayl's first lines, drawn
SECOND = ROOT / "data" / "duaplayer" / "dua-kumayl" / "38e2b8f0-355e-4e46-e0a4-08da084c84ac"  # another reciter


def tracker(ix, d):
    import evaluate as ev

    from dua_recognition.align import encode
    from dua_recognition.tracker import Tracker, TrackerConfig

    a = json.loads((CACHE / "asr.json").read_text(encoding="utf-8"))
    rows, quiet = a["rows"][EVERY - 1 :: EVERY], a["quiet"][EVERY - 1 :: EVERY]
    costs = ev.LazyCosts(ix, [x for _, x in rows])
    cfg = TrackerConfig()
    tr = Tracker(ix, cfg)
    prior = tr._dua_mass().copy()
    anchors, updates, masses = [], [], []
    prev = rows[0][0] - (rows[1][0] - rows[0][0])
    for (t, text), c, q in zip(rows, costs, quiet):
        p = tr.update_costs(c, float(t - prev), DELAY + cfg.display_lead, n_letters=len(encode(text)) if text else 0,
                            quiet=q)
        prev = t
        now = tr.position()
        w = now.word if now.dua is not None else None
        anchors.append((t, w, tr.line_masses(int(ix.word_dua[w])) if w is not None else None))
        mass = tr._dua_mass()
        masses.append(mass.astype(np.float32))
        order = np.argsort(-mass)[:8]
        updates.append({"t": t + DELAY, "window": [max(0.0, t - 6.0), t], "heard": text,
                        "top": [[ix.dua_ids[i], round(float(mass[i]), 4)] for i in order],
                        "null": round(float(tr.null), 4), "named": p.dua, "conf": round(float(p.dua_confidence), 4),
                        "line_p": [round(float(x), 4) for x in tr.line_masses(d)[:LINES]]})
    return rows, anchors, updates, np.array(masses), prior


def decoder(ix, d, anchors):
    import bench
    from dua_recognition.stream_ctc import window_rows
    from dua_recognition.stream_follower import StreamConfig, StreamFollower

    lo, hi = ix.dua_word_span[d]
    seg = ix.word_segment[lo:hi]
    qz = np.load(CACHE / "quiet.npz")
    qt, qv = qz["t"], qz["q"]

    def quiet_at(t):
        i = int(np.searchsorted(qt, t, side="right")) - 1
        return float(qv[i]) if i >= 0 else None

    z = np.load(CACHE / "ctc.npz")
    fin, ten, ends, ts = z["final"], z["tent"], z["ends"], z["t"]
    rows = int(round(bench.CTC_WINDOW / 0.02))
    sf = StreamFollower(ix, StreamConfig())
    seen, real = {}, sf.posterior

    def posterior(*args):
        seen["p"] = real(*args)
        return seen["p"]

    sf.posterior = posterior
    at = [x[0] + DELAY for x in anchors]
    line_no = bench._line_no(ix)
    steps = []
    for k, t in enumerate(ts):
        j = int(np.searchsorted(at, t, side="right")) - 1
        w_anchor = anchors[j][1] if j >= 0 else None
        a_t = at[j] if j >= 0 else None
        lm = None
        if j >= 0 and anchors[j][2] is not None and w_anchor is not None:
            first = sf._dua(int(ix.word_dua[w_anchor]))
            lm = lambda w, m=anchors[j][2], f=int(line_no[first.lo]): float(m[int(line_no[w]) - f])  # noqa: E731
        seen.clear()
        w = sf.step(window_rows(fin, ten[k], int(ends[k]), rows), float(t), w_anchor, quiet_now=quiet_at(t),
                    line_mass=lm, anchor_t=a_t, quiet_then=quiet_at(t - sf.cfg.lookahead))
        per_line, off = np.zeros(LINES), 0.0
        if "p" in seen and sf.dua == d:
            pw, off = seen["p"]
            per_line = np.array([pw[seg == s].sum() for s in range(1, LINES + 1)])
        on = w is not None and int(ix.word_dua[w]) == d
        steps.append((round(float(t) + F_DELAY, 3), per_line, float(off), int(w - lo) if on else (-2 if w is not None else -1)))
    letters = fin.astype(np.float32)
    return steps, letters.argmax(-1).astype(np.int8), np.exp(letters.max(-1)).astype(np.float16)


def logmel(y):
    import torch

    from dua_recognition.ctc_student import LogMel

    return LogMel()(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None])[0].numpy()


def peaks(img: np.ndarray, size: int = 9, per_s: int = 18, hop_s: float = 0.01) -> np.ndarray:
    """Shazam's constellation: points that are the maximum of their neighbourhood, the strongest few per second."""
    from scipy.ndimage import maximum_filter

    is_max = (img == maximum_filter(img, size=size)) & (img > np.percentile(img, 60))
    f, t = np.nonzero(is_max)
    v = img[f, t]
    keep = []
    n_s = img.shape[1] * hop_s
    order = np.argsort(-v)[: int(per_s * n_s)]
    for i in order:
        keep.append((int(t[i]), int(f[i]), float(v[i])))
    return np.array(sorted(keep), dtype=np.float32)


def song(sr: int = 16000, secs: float = 6.0) -> np.ndarray:
    """A made-up song for Shazam's pictures (only drawn, never played): chords, a bass line, a hi-hat."""
    rng = np.random.default_rng(3)
    t = np.arange(int(sr * secs)) / sr
    y = np.zeros_like(t)
    beat = 0.5
    chords = [[220, 277, 330], [196, 247, 294], [175, 220, 262], [196, 247, 330]]
    for k in range(int(secs / (4 * beat)) + 1):
        a, b = k * 4 * beat, (k + 1) * 4 * beat
        on = (t >= a) & (t < b)
        for f in chords[k % 4]:
            for h in range(1, 6):
                y[on] += 0.3 / h * np.sin(2 * np.pi * f * h * t[on]) * np.exp(-1.2 * (t[on] - a))
    for k in range(int(secs / beat)):
        a = k * beat
        on = (t >= a) & (t < a + 0.35)
        f = [110, 98, 87, 98][(k // 4) % 4]
        y[on] += 0.8 * np.sin(2 * np.pi * f * t[on]) * np.exp(-6 * (t[on] - a))
        hh = (t >= a + beat / 2) & (t < a + beat / 2 + 0.04)
        y[hh] += 0.25 * rng.standard_normal(hh.sum())
    return (y / np.abs(y).max() * 0.5).astype(np.float32)


def prefix_counts(ix, line_text: str) -> list[list]:
    from dua_recognition.text import normalize

    joined = {}
    for i, dd in enumerate(ix.dua_ids):
        lo, hi = ix.dua_word_span[i]
        joined[dd] = "".join(normalize(ix.words[w].text) for w in range(lo, hi)).replace(" ", "")
    words = normalize(line_text).split()
    out = []
    for n in range(1, len(words) + 1):
        ph = "".join(words[:n])
        hits = [dd for dd, s in joined.items() if ph in s]
        out.append([" ".join(words[:n]), len(hits), [ix.dua_ids.index(h) for h in hits]])
    return out


def grid() -> dict:
    """On the line, per scenario and lane, the phone's test run (README's numbers come from the same file)."""
    d = json.loads((ROOT / "data" / "testbed" / "results" / "full_ph_G1_test.json").read_text(encoding="utf-8"))
    cells = {}
    for x in d["items"].values():
        k = (x["scenario"], x["lane"])
        c = cells.setdefault(k, [0, 0])
        c[0] += x.get("on_line_ticks", 0)
        c[1] += x.get("read_ticks", 0)
    return {f"{s}|{lane}": round(a / b, 4) for (s, lane), (a, b) in cells.items() if b}


def main() -> None:
    import soundfile as sf

    import evaluate as ev

    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    it = json.loads((CACHE / "item.json").read_text(encoding="utf-8"))
    d = ix.dua_ids.index(it["dua"])
    lo, hi = ix.dua_word_span[d]
    seg = ix.word_segment[lo:hi]
    dua = duas[it["dua"]]
    by = {x.id: x for x in dua.segments}

    rows, anchors, updates, masses, prior = tracker(ix, d)
    steps, letter, letter_p = decoder(ix, d, anchors)

    y, sr = sf.read(CACHE / "reading.wav", dtype="float32")
    n = int(0.01 * sr)
    blocks = y[: len(y) // n * n].reshape(-1, n)
    said = [[int(seg[w]), int(w), float(a), float(b)] for dd, w, a, b in it["words"] if dd == it["dua"]]

    # line 2 in his voice (from the reading) and in a second reciter's (DuaPlayer's hand-placed line start)
    l2 = [x for x in said if x[0] == 2]
    a2, b2 = l2[0][2] - 0.15, l2[-1][3] + 0.15
    mel_a = logmel(y[int(a2 * sr) : int(b2 * sr)])
    meta2 = json.loads(SECOND.with_suffix(".json").read_text(encoding="utf-8"))
    import bench

    s2 = meta2["slide_start_ms"]
    y2 = bench.decode(str(SECOND.with_suffix(".mp3")), s2["2"] / 1000, s2["3"] / 1000)
    mel_b = logmel(y2)
    mel_s = logmel(song())

    # the window that hears line 2's opening: its distance from every word of every text
    pick = max(range(len(rows)), key=lambda i: ("برحمتك" in rows[i][1]) * 10 - abs(rows[i][0] - (l2[3][3] + 0.5)))
    heard = rows[pick][1]
    cost = ix.word_costs(heard).astype(np.float32)

    def text(s):
        return {"ar": by[s].arabic, "tl": by[s].transliteration, "en": by[s].translation}

    np.savez_compressed(
        DATA / "reading.npz",
        step_t=np.array([s[0] for s in steps], np.float32), step_line=np.array([s[1] for s in steps], np.float32),
        step_off=np.array([s[2] for s in steps], np.float32), step_word=np.array([s[3] for s in steps], np.int16),
        dua_mass=masses.astype(np.float16), dua_t=np.array([u["t"] for u in updates], np.float32),
        prior=prior.astype(np.float32), letter=letter, letter_p=letter_p,
        wave_lo=blocks.min(1).astype(np.float16), wave_hi=blocks.max(1).astype(np.float16),
        mel_a=np.clip(mel_a * 255, 0, 255).astype(np.uint8), mel_b=np.clip(mel_b * 255, 0, 255).astype(np.uint8),
        mel_s=np.clip(mel_s * 255, 0, 255).astype(np.uint8),
        peaks_a=peaks(mel_a), peaks_b=peaks(mel_b), peaks_s=peaks(mel_s), line2_span=np.array([a2, b2], np.float32),
        cost=cost, word_dua=ix.word_dua.astype(np.int16), dua_words=np.array([b - a for a, b in ix.dua_word_span]),
    )
    meta = {
        "voice": it["voice"], "dua": it["dua"], "name": dua.name_en, "duration": it["duration"], "events": it["events"],
        "still": it["still"], "said": said, "updates": updates,
        "lines": {s: text(s) for s in range(1, LINES + 1)},
        "line_words": {s: [int(w) for w in np.flatnonzero(seg == s)] for s in range(1, LINES + 1)},
        "word_text": {int(w): ix.words[lo + w].text for w in np.flatnonzero((seg >= 1) & (seg <= LINES))},
        "names": {x: duas[x].name_en for u in updates for x, _ in u["top"]},
        "dua_ids": list(ix.dua_ids), "recordings": [int(x.recordings) for x in ix.duas],
        "kumayl": d, "n_words": int(len(ix.words)),
        "heard": {"t": rows[pick][0], "text": heard, "best": [ix.dua_ids[int(ix.word_dua[w])] for w in np.argsort(cost)[:6]]},
        "prefix": prefix_counts(ix, by[2].arabic + " " + by[3].arabic),
        "grid": grid(),
        "rows": rows,
    }
    (DATA / "reading.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Whisper (every 2 s, shown 1.8 s late):")
    for u in updates:
        top = ", ".join(f"{x} {p:.2f}" for x, p in u["top"][:3])
        print(f"{u['t']:6.1f}  named {u['named'] or '-':12s} conf {u['conf']:.2f} null {u['null']:.2f}  [{top}]  "
              f"heard: {u['heard'][:50]}")
    print("\nheard for the matching picture:", meta["heard"])
    print("prefix counts:", [(p[0], p[1]) for p in meta["prefix"][:7]])
    print("\nDecoder, shown line changes:")
    last = None
    for t, _, off, w in steps:
        ln = None if w < 0 else int(seg[w])
        if (ln, w == -2) != last:
            print(f"{t:6.2f}  line {ln}{' (other)' if w == -2 else ''}  off {off:.2f}")
            last = (ln, w == -2)


if __name__ == "__main__":
    main()
