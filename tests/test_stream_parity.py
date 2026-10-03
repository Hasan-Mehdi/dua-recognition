"""The browser stream decoder (web/stream-follower.js) must move exactly like the Python one
(stream_follower.py), and the numba pass must equal the numpy reference."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dua_recognition.align import CorpusIndex
from dua_recognition.corpus import load_all
from dua_recognition.ctc import N_COLS
from dua_recognition.stream_follower import StreamConfig, StreamFollower

WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")
DUAS = ["dua-hujjat", "dua-kumayl", "dua-tawassul"]


def _frames(ix, words, rng, salawat_after=()):
    """Synthetic CTC output: each word's letters for a few frames, blanks between, pauses, and
    noise frames where a stray letter wins (room tone)."""
    from dua_recognition.align import encode
    from dua_recognition.text import normalize

    rows = []

    def frame(col, p):
        lp = np.full(N_COLS, np.log((1 - p) / (N_COLS - 1)), dtype=np.float32)
        lp[col] = np.log(p)
        lp += np.array([rng.gauss(0, 0.3) for _ in range(N_COLS)], dtype=np.float32)
        return lp - np.logaddexp.reduce(lp)

    def say(codes):
        for c in codes:
            for _ in range(rng.randint(1, 3)):
                rows.append(frame(int(c), rng.uniform(0.4, 0.95)))
            for _ in range(rng.randint(0, 4)):
                rows.append(frame(0, 0.97))

    first = np.r_[0, np.flatnonzero(np.diff(ix.letter_word) != 0) + 1]
    sal = encode(normalize("اللهم صل على محمد وآل محمد"))
    for w in words:
        lo = first[w]
        hi = first[w + 1] if w + 1 < len(first) else ix.letters.size
        say(ix.letters[lo:hi])
        if w in salawat_after:
            rows += [frame(0, 0.99) for _ in range(15)]
            say(sal)
        if rng.random() < 0.1:  # a pause, with a stray letter in it
            rows += [frame(0, 0.99) for _ in range(rng.randint(20, 80))]
            rows.append(frame(rng.randint(1, N_COLS - 1), 0.8))
            rows += [frame(0, 0.99) for _ in range(10)]
    return np.stack(rows).astype(np.float32)


def test_numba_equals_numpy():
    corpus = {k: v for k, v in load_all().items() if k in DUAS}
    ix = CorpusIndex(corpus)
    rng = np.random.default_rng(0)
    d = ix.dua_ids.index("dua-tawassul")
    lo, _ = ix.dua_word_span[d]
    fr = np.log(rng.dirichlet(np.ones(N_COLS) * 0.3, size=40) + 1e-9)
    out = {}
    for jit in (False, True):
        sf = StreamFollower(ix, use_jit=jit)
        sf._start(d, lo + 30)
        out[jit] = sf.posterior(*sf._advance(sf.L, sf.B, sf.F, sf.IL, sf.IB, fr))
    assert np.abs(out[True][0] - out[False][0]).max() < 1e-9
    assert abs(out[True][1] - out[False][1]) < 1e-9


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("extra", [{}, {"trackerWeight": 0.0, "nextSteps": 1}, {"quietPen": 0.0, "interjection": ""}])
def test_js_stream_matches_python(tmp_path, extra):
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in DUAS}
    ix = CorpusIndex(corpus)
    rng = random.Random(1)
    lo, hi = ix.dua_word_span[1]
    words = list(range(lo + 5, lo + 90))
    words[40:40] = list(range(lo + 20, lo + 32))  # the reciter goes back two lines and reads on
    frames = _frames(ix, words, rng, salawat_after={lo + 60})
    steps, anchors, quiet, masses_at = [], [], [], []
    true_at = np.repeat(words, int(np.ceil(len(frames) / len(words))))[: len(frames)]
    for end in range(100, len(frames), 5):  # windows of 2 s every 0.1 s
        a = int(true_at[end - 1]) + rng.choice([-2, -1, 0, 0, 0, 1, 3])
        steps.append(end)
        anchors.append(min(max(lo, a), hi - 1))
        if 600 <= end < 700:
            anchors[-1] = None  # the tracker lost the du'a for a while
        tail = frames[max(0, end - 25) : end]
        quiet.append(round(float((tail[:, 0] > tail[:, 1:].max(axis=1)).mean()) * 0.6, 3))
    # The tracker's belief: most of it on the anchor's line, the rest spread.
    seg = ix.word_segment
    lines = sorted({int(seg[w]) for w in range(lo, hi)})
    js_cfg = {"trackerWeight": 0.3, "nextSteps": 2, "quietPen": 4.0,
              "interjection": "اللهم صل على محمد وآل محمد|وعجل فرجهم", **extra}
    py_cfg = StreamConfig(tracker_weight=js_cfg["trackerWeight"], next_steps=js_cfg["nextSteps"],
                          quiet_pen=js_cfg["quietPen"],
                          interjections=(js_cfg["interjection"],) if js_cfg["interjection"] else ())

    def mass_fn(anchor):
        if anchor is None:
            return None
        hot = int(seg[anchor])
        return lambda w: 0.8 if int(seg[w]) == hot else 0.2 / len(lines)

    f = StreamFollower(ix, py_cfg)
    expected = []
    for i, (end, a, q) in enumerate(zip(steps, anchors, quiet)):
        expected.append(f.step(frames[max(0, end - 100) : end].astype(np.float32), end * 0.02, a, quiet_now=q,
                               line_mass=mass_fn(a), anchor_t=(end // 50) * 1.0 if a is not None else None))
    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    frames.tofile(tmp_path / "frames.f32")
    (tmp_path / "steps.json").write_text(json.dumps({"steps": steps, "anchors": anchors, "quiet": quiet,
                                                     "nLines": len(lines)}))
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex }} from "{(WEB / 'tracker.js').as_uri()}";
import {{ StreamFollower }} from "{(WEB / 'stream-follower.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const buf = readFileSync("{(tmp_path / 'frames.f32').as_posix()}");
const frames = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const {{ steps, anchors, quiet, nLines }} = JSON.parse(readFileSync("{(tmp_path / 'steps.json').as_posix()}", "utf8"));
const C = {N_COLS};
const ix = new CorpusIndex(corpus);
const f = new StreamFollower(ix, {json.dumps(js_cfg, ensure_ascii=False)});
const massFn = (a) => a == null ? null : (w) => ix.wordSegment[w] === ix.wordSegment[a] ? 0.8 : 0.2 / nLines;
console.log(JSON.stringify(steps.map((end, i) => {{
  const s = Math.max(0, end - 100);
  const a = anchors[i];
  return f.step(frames.slice(s * C, end * C), end - s, C, end * 0.02, a, quiet[i], massFn(a),
    a != null ? Math.floor(end / 50) * 1.0 : null);
}})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8",
                         check=True)
    got = json.loads(out.stdout)
    assert len(got) == len(expected)
    assert len(set(expected)) > 30  # it really moved
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatches, mismatches[:5]
