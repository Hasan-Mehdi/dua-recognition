#!/usr/bin/env python
"""Replay the page's display on the explainer's recitation (docs/anim/recitation.py) and keep what it
decided at every step, for the explainer's diagrams.

As `bench.py score --display stream` replays an item: the tracker over the Whisper rows (on screen
1.2 s after each window), then the stream decoder over the CTC windows, anchored on the tracker. Kept:
- per Whisper update: what Whisper heard, the top du'as and their probabilities, the share on "not
  in the corpus", the du'a the page names (None below 70%), and the probability of each line of
  Iftitah;
- per decoder step (0.1 s): the probability of each line of Iftitah and off the text (talk), and the
  word shown;
- what he was reading when (the item's truth), the pause and the talk.

    python docs/anim/dump_recitation.py    # -> docs/anim/recitation.npz, recitation.json
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

CACHE = ROOT / "data" / "cache" / "media" / "explainer8"
DELAY, F_DELAY = 1.2, 0.15  # bench.py score's defaults
TOP = 8


def _lines_read(said) -> list[tuple[int, float, float]]:
    """Each run of one line as he read it: (line, start, end)."""
    out = []
    for ln, _, a, b in said:
        if out and out[-1][0] == ln:
            out[-1] = (ln, out[-1][1], b)
        else:
            out.append((ln, a, b))
    return out


def main() -> None:
    import bench
    import evaluate as ev

    from dua_recognition.align import encode
    from dua_recognition.stream_follower import StreamConfig, StreamFollower
    from dua_recognition.tracker import Tracker, TrackerConfig

    it = json.loads((CACHE / "item.json").read_text(encoding="utf-8"))
    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    d = ix.dua_ids.index(it["dua"])
    lo, hi = ix.dua_word_span[d]
    seg = ix.word_segment[lo:hi]
    lines = np.unique(seg)

    # The tracker, as word_eval.hmm_updates drives it, keeping its probabilities.
    a = json.loads((CACHE / "asr.json").read_text(encoding="utf-8"))
    rows = a["rows"]
    costs = ev.LazyCosts(ix, [x for _, x in rows])
    cfg = TrackerConfig()
    tr = Tracker(ix, cfg)
    anchors, updates, masses = [], [], []
    for (t, text), c, q in zip(rows, costs, a["quiet"]):
        p = tr.update_costs(c, 1.0, DELAY + cfg.display_lead, n_letters=len(encode(text)) if text else 0, quiet=q)
        now = tr.position()
        w = now.word if now.dua is not None else None
        anchors.append((t, w, tr.line_masses(int(ix.word_dua[w])) if w is not None else None))
        mass = tr._dua_mass()
        masses.append(mass)
        order = np.argsort(-mass)[:TOP]
        updates.append({
            "t": t + DELAY, "window": [max(0.0, t - bench.WINDOW), t], "heard": text,
            "top": [[ix.dua_ids[i], round(float(mass[i]), 4)] for i in order], "null": round(float(tr.null), 4),
            "named": p.dua, "conf": round(float(p.dua_confidence), 4),
            "line": None if p.segment is None or p.dua != it["dua"] else int(p.segment),
            "line_p": [round(float(x), 4) for x in tr.line_masses(d)],
        })

    # The stream decoder (bench.stream_updates), keeping its probability on each line.
    qz = np.load(CACHE / "quiet.npz")
    qt, qv = qz["t"], qz["q"]

    def quiet_at(t: float) -> float | None:
        i = int(np.searchsorted(qt, t, side="right")) - 1
        return float(qv[i]) if i >= 0 else None

    z = np.load(CACHE / "ctc.npz")
    lp, nf, ts = z["lp"], z["n_frames"], z["t"]
    sf = StreamFollower(ix, StreamConfig())
    seen = {}
    real_posterior = sf.posterior

    def posterior(*args):
        seen["p"] = real_posterior(*args)
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
        w = sf.step(lp[k, : nf[k]].astype(np.float32), float(t), w_anchor, quiet_now=quiet_at(t), line_mass=lm,
                    anchor_t=a_t, quiet_then=quiet_at(t - sf.cfg.lookahead))
        per_line, off = np.zeros(len(lines)), 0.0
        if "p" in seen and sf.dua == d:
            pw, off = seen["p"]
            per_line = np.array([pw[seg == s].sum() for s in lines])
        on = w is not None and int(ix.word_dua[w]) == d
        steps.append({"t": round(float(t) + F_DELAY, 3), "line_p": per_line, "off": float(off),
                      "word": int(w - lo) if on else (-2 if w is not None else -1)})

    said = [(int(seg[w]), int(w), float(s), float(e)) for dd, w, s, e in it["words"] if dd == it["dua"]]
    dua = duas[it["dua"]]
    keep = [int(s) for s in lines if s <= 20]
    # The letter model's newest 0.1 s from each window, end to end: one frame per 20 ms over the
    # whole recitation, its likeliest column (0 = no letter) and that column's probability.
    per = int(round(bench.CTC_HOP / 0.02))
    newest = np.concatenate([lp[k, nf[k] - per : nf[k]] for k in range(len(ts))]).astype(np.float32)
    # Loudness every 0.1 s (RMS, dB), for drawing the recording.
    import soundfile as sf

    y, sr = sf.read(CACHE / "recitation.wav", dtype="float32")
    n = int(0.1 * sr)
    rms = np.sqrt(np.mean(y[: len(y) // n * n].reshape(-1, n) ** 2, axis=1))
    env = 20 * np.log10(np.maximum(rms, 1e-5))
    # The basmala, for "how a phone hears": its waveform (min and max every 2 ms), 20 ms of its
    # samples as they are (inside "allāh"), and its picture as both models see it: Whisper's log-mel
    # spectrogram, 80 bands every 10 ms (ctc_student.LogMel, the same numbers as Whisper's own).
    import torch

    from dua_recognition.ctc_student import LogMel

    b0, b1 = [(a, b) for ln, a, b in _lines_read(said) if ln == 1][0]
    b0, b1 = b0 - 0.25, b1 + 0.25
    seg_y = y[int(b0 * sr) : int(b1 * sr)]
    k = int(0.002 * sr)
    blocks = seg_y[: len(seg_y) // k * k].reshape(-1, k)
    w1 = [w for ln, w, a, b in said if ln == 1][1]  # "allāh"
    a1, e1 = next((a, b) for ln, w, a, b in said if w == w1)
    mid = int(((a1 + e1) / 2) * sr)
    mel = LogMel()(torch.from_numpy(seg_y)[None])[0].numpy()
    np.savez(HERE / "recitation.npz", t=np.array([s["t"] for s in steps]),
             line_p=np.array([s["line_p"] for s in steps], dtype=np.float32)[:, lines <= 20],
             off=np.array([s["off"] for s in steps], dtype=np.float32),
             word=np.array([s["word"] for s in steps]), lines=lines[lines <= 20],
             dua_mass=np.array(masses, dtype=np.float32), dua_t=np.array([u["t"] for u in updates]),
             letter=newest.argmax(-1).astype(np.int8), letter_p=np.exp(newest.max(-1)).astype(np.float16),
             env=env.astype(np.float32), letters_lp=newest.astype(np.float16),
             basmala_span=np.array([b0, b1]), basmala_lo=blocks.min(1), basmala_hi=blocks.max(1),
             basmala_samples=y[mid - int(0.01 * sr) : mid + int(0.01 * sr)], basmala_at=np.array([mid / sr - 0.01]),
             basmala_mel=np.clip(mel * 255, 0, 255).astype(np.uint8))
    by = {x.id: x for x in dua.segments}
    meta = {
        "voice": it["voice"], "dua": it["dua"], "name": dua.name_en, "duration": it["duration"],
        "events": it["events"], "still": it["still"], "said": said, "updates": updates,
        "lines": {s: {"ar": by[s].arabic, "tl": by[s].transliteration, "en": by[s].translation} for s in keep},
        "line_words": {s: [int(w) for w in np.flatnonzero(seg == s)] for s in keep},
        "word_text": {int(w): ix.words[lo + w].text for w in np.flatnonzero(np.isin(seg, keep))},
        "names": {x: duas[x].name_en for u in updates for x, _ in u["top"]},
        "dua_ids": list(ix.dua_ids), "recordings": [int(x.recordings) for x in ix.duas],
    }
    (HERE / "recitation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # The story, to write the script by: the line shown against the line read.
    def reading(t):
        for ln, _, a0, b0 in reversed(said):
            if a0 <= t:
                return ln if t <= b0 + 1.0 else f"{ln}(after)"
        return None

    def still(t):
        return next((k for a0, b0, k in it["still"] if a0 <= t < b0), "")

    print("Whisper updates:")
    for u in updates:
        top = ", ".join(f"{x} {p:.2f}" for x, p in u["top"][:3])
        print(f"{u['t']:6.1f}  named {u['named'] or '-':14s} conf {u['conf']:.2f}  line {u['line']}  null "
              f"{u['null']:.2f}  [{top}]  heard: {u['heard'][:60]}")
    print("\nDecoder, shown line changes:")
    last = None
    for s in steps:
        ln = None if s["word"] < 0 else int(seg[s["word"]])
        key = (ln, s["word"] == -2)
        if key != last:
            print(f"{s['t']:6.2f}  shown line {ln}{' (other du_a)' if s['word'] == -2 else ''}  reading "
                  f"{reading(s['t'])} {still(s['t'])}  off {s['off']:.2f}")
            last = key


if __name__ == "__main__":
    main()
