#!/usr/bin/env python
"""Dump the stream decoder's probabilities over the lines of Dua Tawassul while a reader goes back,
for the explainer's "following" part.

The scene is the scenario bench's item back-57581e9b66: Hussein Ghareeb (a held-out reciter) from
line 39, cut so that after line 43 he goes back to line 41, as a reader might. The bench has the
phone models' outputs for it cached (bench.py asr: Whisper rows, CTC frames, the stop detector), so
this replays the page's display exactly as `bench.py score --display stream` does, and keeps, at
every 0.1 s step, the probability on each line of the du'a and off it (talk, salawat), and the word
shown.

    python docs/anim/dump_follow_data.py      # -> docs/anim/explainer_follow.npz, explainer_follow.json
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

ITEM = "back-57581e9b66"
SPAN = (0.0, 50.0)  # item seconds kept


def main() -> None:
    import bench
    import evaluate as ev
    import word_eval as we

    from dua_recognition.align import encode
    from dua_recognition.stream_follower import StreamConfig, StreamFollower
    from dua_recognition.tracker import TrackerConfig

    it = json.loads((bench.BENCH / "items" / "back" / f"{ITEM}.json").read_text(encoding="utf-8"))
    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    fa, fc, fq = bench.asr_paths(it, bench.ASR_TAG, bench.CTC_TAG)
    delay, f_delay = 1.2, 0.15  # bench.py score's defaults: Whisper on screen 1.2 s after its window

    # The tracker over the Whisper rows (bench.replay, without writing its shared cache).
    a = json.loads(fa.read_text(encoding="utf-8"))
    rows = a["rows"]
    costs = ev.LazyCosts(ix, [x for _, x in rows])
    letters = [len(encode(x)) if x else 0 for _, x in rows]
    anchors = []
    we.hmm_updates(ix, rows, costs, a["quiet"], TrackerConfig(), delay, "fixed", None, still=True, anchors=anchors,
                   letters=letters)
    anchors = [(t, w, None if m is None else np.asarray(m, dtype=np.float32)) for t, w, m in anchors]
    q = np.load(fq)
    qt, qv = q["t"], q["q"]

    def quiet_at(t: float) -> float | None:
        i = int(np.searchsorted(qt, t, side="right")) - 1
        return float(qv[i]) if i >= 0 else None

    # bench.stream_updates, keeping the belief the display decided on at each step.
    z = np.load(fc)
    lp, nf, ts = z["lp"], z["n_frames"], z["t"]
    sf = StreamFollower(ix, StreamConfig())
    seen = {}
    real_posterior = sf.posterior

    def posterior(*args):
        seen["p"] = real_posterior(*args)
        return seen["p"]

    sf.posterior = posterior
    at = [x[0] + delay for x in anchors]
    d = ix.dua_ids.index(it["dua"])
    lo, hi = ix.dua_word_span[d]
    seg = ix.word_segment[lo:hi]
    lines = np.unique(seg)
    steps = []
    for k, t in enumerate(ts):
        j = int(np.searchsorted(at, t, side="right")) - 1
        w_anchor = anchors[j][1] if j >= 0 else None
        a_t = at[j] if j >= 0 else None
        lm = None
        if j >= 0 and anchors[j][2] is not None and w_anchor is not None:
            masses = anchors[j][2]
            first = sf._dua(int(ix.word_dua[w_anchor]))
            line0 = int(bench._line_no(ix)[first.lo])
            lm = lambda w, m=masses, f=line0: float(m[int(bench._line_no(ix)[w]) - f])  # noqa: E731
        seen.clear()
        w = sf.step(lp[k, : nf[k]].astype(np.float32), float(t), w_anchor, quiet_now=quiet_at(t), line_mass=lm,
                    anchor_t=a_t, quiet_then=quiet_at(t - sf.cfg.lookahead))
        if not (SPAN[0] <= t <= SPAN[1]):
            continue
        per_line, off = np.zeros(len(lines)), 0.0
        if "p" in seen and sf.dua == d:
            pw, off = seen["p"]
            per_line = np.array([pw[seg == s].sum() for s in lines])
        steps.append({"t": round(float(t) + f_delay, 3), "line_p": per_line, "off": float(off),
                      "word": None if w is None else int(w - lo), "anchor": None if w_anchor is None else int(w_anchor - lo),
                      "whisper": a_t is not None and (k == 0 or j != int(np.searchsorted(at, ts[k - 1], side="right")) - 1)})

    # What he was reading, in item seconds: (line, word, from, to).
    said = [(int(seg[w]), int(w), float(s), float(e)) for dd, w, s, e in it["words"] if dd == it["dua"]
            and SPAN[0] <= s <= SPAN[1]]
    dua = duas[it["dua"]]
    keep = range(37, 47)
    np.savez(HERE / "explainer_follow.npz", t=np.array([s["t"] for s in steps]),
             line_p=np.array([s["line_p"] for s in steps], dtype=np.float32), off=np.array([s["off"] for s in steps]),
             word=np.array([-1 if s["word"] is None else s["word"] for s in steps]),
             whisper=np.array([s["whisper"] for s in steps]), lines=lines)
    meta = {
        "item": ITEM, "voice": it["voice"], "events": it["events"], "still": it["still"], "span": SPAN,
        "said": said,
        "lines": {int(x.id): x.arabic for x in dua.segments if x.id in keep},
        "line_words": {int(s): [int(w) for w in np.flatnonzero(seg == s)] for s in keep},
        "word_text": {int(w): ix.words[lo + w].text for w in np.flatnonzero(np.isin(seg, list(keep)))},
        "found_at": next((s["t"] for s in steps if s["word"] is not None), None),
    }
    (HERE / "explainer_follow.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # The item's audio for filming the real page on it (docs/demo/capture.mjs --file), with its line
    # starts. It goes to the ignored cache: the recording is DuaPlayer's.
    import soundfile as sf

    media = ROOT / "data" / "cache" / "media"
    media.mkdir(parents=True, exist_ok=True)
    y = bench.render(it)[: int((SPAN[1] + 2.0) * bench.SR)]
    sf.write(media / "explainer_back.wav", y, bench.SR, subtype="PCM_16")
    starts = [{"seg": ln, "from": round(s, 2)} for k, (ln, _, s, _) in enumerate(said) if k == 0 or said[k - 1][0] != ln]
    (media / "explainer_back.lines.json").write_text(json.dumps(starts), encoding="utf-8")

    # A summary to check the scene by: the line shown against the line read, around the go-back.
    shown_line = [None if s["word"] is None else int(seg[s["word"]]) for s in steps]
    reading = lambda t: next((ln for ln, _, a0, b0 in reversed(said) if a0 <= t), None)  # noqa: E731
    last = None
    for s, ln in zip(steps, shown_line):
        if ln != last:
            print(f"{s['t']:6.2f}  shown line {ln}  (reading {reading(s['t'])})  off {s['off']:.2f}")
            last = ln


if __name__ == "__main__":
    main()
