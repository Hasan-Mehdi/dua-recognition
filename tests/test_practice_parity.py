"""The browser's practice checker (web/practice.js) must decide exactly like the Python one
(src/dua_recognition/practice.py): same lines, same moments, same scores."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dua_recognition.align import encode
from dua_recognition.practice import PracticeChecker, PracticeConfig, committed_frames, judge_lines

WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")
C = 64
LINES = ["الحمد لله رب العالمين", "الرحمن الرحيم", "مالك يوم الدين", "اياك نعبد واياك نستعين",
         "اهدنا الصراط المستقيم", "صراط الذين انعمت عليهم", "غير المغضوب عليهم", "ولا الضالين",
         "يا مجير", "اجرنا من النار يا مجير"]


def _session(rng, skip=(3, 7)):
    """A synthetic reading (lines in `skip` left out), as CTC windows of 2 s every 0.1 s, and a
    display that follows a little late."""
    letters = [encode(t.replace(" ", "")).astype(np.int64) for t in LINES]
    rows, line_at = [], []

    def frame(col, p):
        x = np.full(C, np.log((1 - p) / (C - 1)))
        x[col] = np.log(p)
        x += np.array([rng.gauss(0, 0.3) for _ in range(C)])
        return x - np.logaddexp.reduce(x)

    for k, lt in enumerate(letters):
        if k in skip:
            continue
        for c in lt:
            for _ in range(rng.randint(1, 3)):
                rows.append(frame(int(c), rng.uniform(0.5, 0.95)))
                line_at.append(k)
        for _ in range(rng.randint(5, 25)):
            rows.append(frame(0, 0.97))
            line_at.append(k)
    rows += [frame(0, 0.98)] * 120
    line_at += [line_at[-1]] * 120
    F = np.array(rows, dtype=np.float32)
    n = len(F)
    ts = [round(0.1 * (i + 1), 3) for i in range(int(n * 0.02 / 0.1))]
    wins = np.zeros((len(ts), 100, C), np.float32)
    nf = np.zeros(len(ts), np.int64)
    steps = []
    for i, t in enumerate(ts):
        end = int(round(t / 0.02))
        w = F[max(0, end - 100) : end]
        wins[i, : len(w)] = w
        nf[i] = len(w)
        j = max(0, end - 1 - 30)  # the display: 0.6 s behind
        steps.append((t, int(line_at[j]) if j < len(line_at) else None))
    first_words = [len(encode(t.split()[0])) for t in LINES]
    return letters, first_words, wins, nf, np.array(ts), steps


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_decides_like_python(tmp_path):
    rng = random.Random(4)
    letters, fw, wins, nf, ts, steps = _session(rng)
    cfg = PracticeConfig(span_min_s=6.0)  # a short session: a span floor that bites
    frames, ftimes = committed_frames(wins, nf, ts)
    py = judge_lines(frames, ftimes, steps, letters, 0, len(LINES) - 1, cfg, first_words=fw)
    np.save(tmp_path / "wins.npy", wins)
    (tmp_path / "in.json").write_text(json.dumps({
        "letters": [x.tolist() for x in letters], "fw": fw, "nf": nf.tolist(), "ts": ts.tolist(),
        "steps": steps, "T": 100, "C": C}), encoding="utf-8")
    (tmp_path / "wins.bin").write_bytes(wins.astype(np.float32).tobytes())
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ PracticeChecker, FrameCommitter }} from {json.dumps((WEB / "practice.js").as_uri())};
const inp = JSON.parse(readFileSync({json.dumps(str(tmp_path / "in.json"))}, "utf8"));
const raw = readFileSync({json.dumps(str(tmp_path / "wins.bin"))});
const wins = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const W = inp.T * inp.C;
const ch = new PracticeChecker(inp.letters.map((x) => Int32Array.from(x)), 0, inp.letters.length - 1,
  {{ spanMinS: 6.0 }}, inp.fw);
const com = new FrameCommitter();
const out = [];
for (let i = 0; i < inp.ts.length; i++) {{
  const T = inp.nf[i];
  const fr = wins.subarray(i * W, i * W + T * inp.C);
  const {{ rows, times }} = com.add(fr, T, inp.C, inp.ts[i]);
  ch.addFrames(rows, inp.C, times);
  out.push(...ch.step(inp.steps[i][0], inp.steps[i][1]));
}}
out.push(...ch.finish(inp.steps[inp.steps.length - 1][0]));
console.log(JSON.stringify(out));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    res = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8",
                         timeout=120)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    assert [(d["line"], round(d["t"], 3), d["shown"]) for d in js] == [(k, round(t, 3), sh) for k, t, _, sh in py]
    for d, (_, _, sc, _) in zip(js, py):
        assert abs(d["score"] - sc) <= 1e-6 * max(1.0, abs(sc))
    from dua_recognition.practice import gated_verdicts
    assert [d["verdict"] for d in js] == gated_verdicts(py, cfg)
    # and the session's verdicts are the ones a reader should get: lines 3 and 7 left out
    v = {r[0]: g for r, g in zip(py, gated_verdicts(py, cfg))}
    assert v[3] == "left out" and v[7] == "left out"
    assert all(v[k] == "heard" for k in v if k not in (3, 7))


def test_online_checker_matches_the_batch_wrapper():
    rng = random.Random(5)
    letters, fw, wins, nf, ts, steps = _session(rng, skip=(2,))
    frames, ftimes = committed_frames(wins, nf, ts)
    batch = judge_lines(frames, ftimes, steps, letters, 0, len(LINES) - 1, PracticeConfig(), first_words=fw)
    ch = PracticeChecker(letters, 0, len(LINES) - 1, PracticeConfig(), fw)
    out, a = [], 0
    for t, li in steps:  # frames a few steps at a time, as a slow phone delivers them
        b = int(np.searchsorted(ftimes, t - 0.2, side="right"))
        ch.add_frames(frames[a:b], ftimes[a:b])
        a = b
        out += ch.step(t, li)
    out += ch.finish(steps[-1][0])
    assert out == batch


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_lines_match_python(tmp_path):
    """The page cuts a du'a into the checker's lines (practice.js linesOf) exactly as the bench does."""
    from dua_recognition.align import CorpusIndex
    from dua_recognition.corpus import load_all

    duas = {k: v for k, v in load_all().items() if k in ("dua-hujjat", "ziyarat-ashura", "dua-kumayl")}
    ix = CorpusIndex(duas)
    wl = np.r_[0, np.flatnonzero(np.diff(ix.letter_word) != 0) + 1, ix.letters.size]
    want = {}
    for d, did in enumerate(ix.dua_ids):
        lo, hi = ix.dua_word_span[d]
        lines, fw, cur = [], [], None
        for w in range(lo, hi):
            if ix.word_segment[w] != cur:
                cur = ix.word_segment[w]
                lines.append([])
                fw.append(int(wl[w + 1] - wl[w]))
            lines[-1] += ix.letters[wl[w] : wl[w + 1]].tolist()
        want[did] = {"letters": lines, "fw": fw}
    payload = [{"id": k, "segments": [{"id": s.id, "ar": s.arabic} for s in v.segments]} for k, v in duas.items()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex }} from {json.dumps((WEB / "tracker.js").as_uri())};
import {{ linesOf }} from {json.dumps((WEB / "practice.js").as_uri())};
const ix = new CorpusIndex(JSON.parse(readFileSync({json.dumps(str(tmp_path / "corpus.json"))}, "utf8")));
const out = {{}};
ix.duaIds.forEach((id, d) => {{ const L = linesOf(ix, d); out[id] = {{ letters: L.letters.map((x) => [...x]), fw: L.firstWords }}; }});
console.log(JSON.stringify(out));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    res = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8",
                         timeout=120)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout) == want
