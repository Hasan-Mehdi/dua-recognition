"""The browser word follower (web/follower.js) must move exactly like the Python one (follower.py)."""
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
from dua_recognition.follower import FollowerConfig, LocalFollower

WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")
DUAS = ["dua-hujjat", "dua-kumayl", "dua-tawassul"]


def _frames(ix, words, rng):
    """Synthetic CTC output: each word's letters for a few frames, blanks between, pauses."""
    rows = []
    def frame(col, p):
        lp = np.full(N_COLS, np.log((1 - p) / (N_COLS - 1)), dtype=np.float32)
        lp[col] = np.log(p)
        lp += np.array([rng.gauss(0, 0.3) for _ in range(N_COLS)], dtype=np.float32)
        return lp - np.logaddexp.reduce(lp)
    first = np.r_[0, np.flatnonzero(np.diff(ix.letter_word) != 0) + 1]
    for w in words:
        lo = first[w]
        hi = first[w + 1] if w + 1 < len(first) else ix.letters.size
        for c in ix.letters[lo:hi]:
            for _ in range(rng.randint(1, 3)):
                rows.append(frame(int(c), rng.uniform(0.4, 0.95)))
            for _ in range(rng.randint(0, 4)):
                rows.append(frame(0, 0.97))
        if rng.random() < 0.1:  # a pause
            rows += [frame(0, 0.99) for _ in range(rng.randint(20, 80))]
    return np.stack(rows).astype(np.float32)


EXTRA = {"restart_cost": "restartCost", "restart_lines": "restartLines", "ahead_stuck": "aheadStuck",
         "quiet_after": "quietAfter", "beta_back": "betaBack", "line_confirm": "lineConfirm",
         "reset_stuck": "resetStuck", "repeat_hold": "repeatHold", "line_quiet": "lineQuiet",
         "jump_margin": "jumpMargin", "jump_confirm": "jumpConfirm", "jump_window": "jumpWindow", "jump_hold": "jumpHold",
         "jump_alone": "jumpAlone"}


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("confirm,lapse,leap,back,extra", [
    (1, 0.0, 0.0, (1, 1), {}), (2, 0.0, 0.0, (1, 1), {}), (1, 3.0, 0.0, (1, 1), {}), (1, 3.0, 3.0, (2, 3), {}),
    (1, 0.0, 4.0, (2, 2), {"quiet_after": 0.0}),
    (1, 0.0, 4.0, (2, 4), {"restart_cost": 3.0, "ahead_stuck": True, "beta_back": 2.0}),
    (1, 0.0, 4.0, (2, 2), {"restart_cost": 2.0, "restart_lines": 2, "ahead_stuck": True}),
    (1, 0.0, 4.0, (2, 4), {"restart_cost": 3.0, "ahead_stuck": True, "beta_back": 2.0, "line_confirm": 2}),
    (1, 0.0, 4.0, (2, 2), {"reset_stuck": False, "line_confirm": 1}),
    (1, 0.0, 4.0, (2, 2), {"repeat_hold": 0.0, "ahead_stuck": False}),
    (1, 0.0, 4.0, (2, 2), {"ahead_stuck": False}),
    (1, 0.0, 4.0, (2, 2), {"line_quiet": 0.0}), (1, 0.0, 4.0, (2, 2), {"line_quiet": 0.1, "line_confirm": 1}),
    (1, 0.0, 4.0, (2, 4), {"jump_margin": 2.0}), (1, 0.0, 4.0, (2, 4), {"jump_margin": 4.0, "jump_confirm": 3, "jump_window": 1.6})])  # repeat_hold (default) holds the re-anchor off the repeat
def test_js_follower_matches_python(tmp_path, confirm, lapse, leap, back, extra):
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in DUAS}
    ix = CorpusIndex(corpus)
    rng = random.Random(1)
    lo, hi = ix.dua_word_span[1]
    words = list(range(lo + 5, lo + 90))
    words[40:40] = list(range(lo + 30, lo + 38))  # the reciter repeats a stretch
    frames = _frames(ix, words, rng)
    # Windows of 3 s every 0.2 s; the tracker's anchor wanders, and once jumps far away.
    steps, anchors = [], []
    true_at = np.repeat(words, int(np.ceil(len(frames) / len(words))))[: len(frames)]
    for end in range(150, len(frames), 10):
        a = int(true_at[end - 1]) + rng.choice([-2, -1, 0, 0, 0, 1, 3])
        if 300 <= end < 420:
            a = lo + 150  # far off: the follower re-anchors after reset_after
        steps.append(end)
        anchors.append(min(max(lo, a), hi - 1))
        if 600 <= end < 700 or 900 <= end < 1200:
            anchors[-1] = None  # the tracker lost the du'a (a short and a long lapse)
        elif 1300 <= end < 1500 and anchors[-1] is not None:
            anchors[-1] = min(hi - 1, anchors[-1] + 5)  # the tracker runs ahead: forward re-anchor
    cfg = FollowerConfig(confirm_steps=confirm, lapse_hold=lapse, leap_margin=leap, back_min=back[0], back_confirm=back[1],
                         **extra)
    js_extra = "".join(f", {EXTRA[k]}: {json.dumps(v)}" for k, v in extra.items())
    f = LocalFollower(ix, cfg)
    # The page's stop detector: seconds without voice, from the frames (blank-dominated tails).
    quiet = []
    for end in steps:
        tail = frames[max(0, end - 25) : end]
        quiet.append(round(float((tail[:, 0] > tail[:, 1:].max(axis=1)).mean()) * 0.6, 3))
    expected = []
    for end, a, q in zip(steps, anchors, quiet):
        expected.append(f.step(frames[max(0, end - 150) : end], end * 0.02, a, quiet_now=q))

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    frames.tofile(tmp_path / "frames.f32")
    (tmp_path / "steps.json").write_text(json.dumps({"steps": steps, "anchors": anchors, "quiet": quiet}))
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex }} from "{(WEB / 'tracker.js').as_uri()}";
import {{ LocalFollower }} from "{(WEB / 'follower.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const buf = readFileSync("{(tmp_path / 'frames.f32').as_posix()}");
const frames = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const {{ steps, anchors, quiet }} = JSON.parse(readFileSync("{(tmp_path / 'steps.json').as_posix()}", "utf8"));
const C = {N_COLS};
const f = new LocalFollower(new CorpusIndex(corpus), {{ confirmSteps: {confirm}, lapseHold: {lapse}, leapMargin: {leap}, backMin: {back[0]}, backConfirm: {back[1]}{js_extra} }});
console.log(JSON.stringify(steps.map((end, i) => {{
  const s = Math.max(0, end - 150);
  return f.step(frames.slice(s * C, end * C), end - s, C, end * 0.02, anchors[i], quiet[i]);
}})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    assert len(got) == len(expected)
    assert len(set(expected)) > 30  # it really moved
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatches, mismatches[:5]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_quiet_catch_up_matches_python(tmp_path):
    """A line's last word the CTC frames never show, then silence, with the tracker's anchor on
    that word: both followers move to it quiet_after seconds into the silence, at the same step."""
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in DUAS}
    ix = CorpusIndex(corpus)
    rng = random.Random(3)
    lo, _ = ix.dua_word_span[0]
    seg = ix.word_segment
    end = next(w for w in range(lo + 8, lo + 60) if seg[w + 1] != seg[w] and seg[w - 1] == seg[w] == seg[w - 2])
    said = list(range(end - 6, end))  # the line's last word (`end`) is never heard
    frames = _frames(ix, said, rng)
    silence = np.stack([np.full(N_COLS, np.log(0.01 / (N_COLS - 1)), dtype=np.float32) for _ in range(150)])
    silence[:, 0] = np.log(0.99)
    frames = np.concatenate([frames, silence]).astype(np.float32)
    steps = list(range(150, len(frames), 10))
    anchors = [end - 1 if s < len(frames) - 120 else end for s in steps]
    f = LocalFollower(ix, FollowerConfig())
    expected = [f.step(frames[max(0, s - 150) : s], s * 0.02, a) for s, a in zip(steps, anchors)]
    assert expected[-1] == end
    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    frames.tofile(tmp_path / "frames.f32")
    (tmp_path / "steps.json").write_text(json.dumps({"steps": steps, "anchors": anchors}))
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex }} from "{(WEB / 'tracker.js').as_uri()}";
import {{ LocalFollower }} from "{(WEB / 'follower.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const buf = readFileSync("{(tmp_path / 'frames.f32').as_posix()}");
const frames = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const {{ steps, anchors }} = JSON.parse(readFileSync("{(tmp_path / 'steps.json').as_posix()}", "utf8"));
const C = {N_COLS};
const f = new LocalFollower(new CorpusIndex(corpus));
console.log(JSON.stringify(steps.map((end, i) => {{
  const s = Math.max(0, end - 150);
  return f.step(frames.slice(s * C, end * C), end - s, C, end * 0.02, anchors[i]);
}})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(out.stdout) == expected



@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("extra", [{"jump_margin": 5.0}, {"jump_margin": 7.0, "jump_confirm": 4, "jump_repeated": 4.0},
                                   {"jump_margin": 4.0, "jump_alone": 8.0, "jump_mass": 0.02},
                                   {"jump_margin": 5.0, "jump_seg": True, "jump_window": 2.0}])
def test_js_jump_search_matches_python(tmp_path, extra):
    """A reader who reads Kumayl's opening lines, then jumps far ahead and back: the jump search
    (follower.py _jump_search) moves the follower, and the JS one makes the same moves."""
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in DUAS}
    ix = CorpusIndex(corpus)
    rng = random.Random(2)
    d = list(corpus).index("dua-kumayl")
    lo, hi = ix.dua_word_span[d]
    seg = ix.word_segment
    line = lambda k: [w for w in range(lo, hi) if seg[w] == seg[lo] + k]  # noqa: E731
    words = [w for k in range(4) for w in line(k)] + line(80) + line(81) + line(20)
    frames = _frames(ix, words, rng)
    true_at = np.repeat(words, int(np.ceil(len(frames) / len(words))))[: len(frames)]
    steps = list(range(150, len(frames), 10))
    # The tracker: the true word 1.5 s ago; its belief in the line it says (for jump_mass): 0.5.
    anchors = [int(true_at[max(0, s - 75) - 1]) for s in steps]
    full = {"jump_confirm": 2, "jump_repeated": 0.0, "jump_alone": 0.0, "jump_mass": 0.0, "jump_seg": False,
            "jump_window": 1.2, **extra}  # every jump setting spelled out, the same on both sides
    expected = []
    f = LocalFollower(ix, FollowerConfig(**full))
    for s, a in zip(steps, anchors):
        expected.append(f.step(frames[max(0, s - 150) : s], s * 0.02, a,
                               line_mass=lambda w, a=a: 0.5 if seg[w] == seg[a] else 0.0))
    assert any(e is not None and e2 is not None and abs(e2 - e) > 20 for e, e2 in zip(expected, expected[1:]))
    js_extra = ", ".join(f"{k}: {json.dumps(v)}" for k, v in {
        "jumpMargin": full["jump_margin"], "jumpConfirm": full["jump_confirm"], "jumpRepeated": full["jump_repeated"],
        "jumpAlone": full["jump_alone"], "jumpMass": full["jump_mass"], "jumpSeg": full["jump_seg"],
        "jumpWindow": full["jump_window"]}.items())
    payload = [{"id": x.id, "name_en": x.name_en, "name_ar": x.name_ar,
                "segments": [{"id": sg.id, "ar": sg.arabic} for sg in x.segments]} for x in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    frames.tofile(tmp_path / "frames.f32")
    (tmp_path / "steps.json").write_text(json.dumps({"steps": steps, "anchors": anchors}))
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex }} from "{(WEB / 'tracker.js').as_uri()}";
import {{ LocalFollower }} from "{(WEB / 'follower.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const buf = readFileSync("{(tmp_path / 'frames.f32').as_posix()}");
const frames = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const {{ steps, anchors }} = JSON.parse(readFileSync("{(tmp_path / 'steps.json').as_posix()}", "utf8"));
const C = {N_COLS};
const ix = new CorpusIndex(corpus);
const f = new LocalFollower(ix, {{ {js_extra} }});
console.log(JSON.stringify(steps.map((end, i) => {{
  const s = Math.max(0, end - 150);
  const a = anchors[i];
  return f.step(frames.slice(s * C, end * C), end - s, C, end * 0.02, a, null,
    (w) => (ix.wordSegment[w] === ix.wordSegment[a] ? 0.5 : 0));
}})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert len(got) == len(expected) and not mismatches, mismatches[:5]
