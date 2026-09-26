"""The browser tracker (web/tracker.js) must follow exactly like the Python one."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from dua_recognition.align import CorpusIndex
from dua_recognition.corpus import load_all
from dua_recognition.tracker import Tracker, TrackerConfig

WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")


def _stream(duas, seed=0):
    """Sliding 'transcripts' over two real du'as, with some misheard words and pauses."""
    rng = random.Random(seed)
    noise = ["يا", "الله", "رحمتك", "كلمات", "شيء"]
    texts = []
    for dua in duas:
        words = [w for s in dua.segments for w in s.arabic.split()][:120]
        for i in range(0, len(words), 2):
            win = words[max(0, i - 5) : i + 1]
            if rng.random() < 0.15:
                win = win + [rng.choice(noise)]
            texts.append("" if rng.random() < 0.08 else " ".join(win))
    return texts


# A fixed slice of the corpus (it was "the first 12" until more texts sorted in front).
PARITY_DUAS = ["dua-aahad", "dua-abu-hamza-thumali", "dua-al-hajj", "dua-allahhuma-adhkil-ala",
               "dua-allahuma-arzukni", "dua-allahuma-laan-qatala", "dua-allahuma-ya-man-yaml",
               "dua-allahumma-laka-sumtu", "dua-amaal-quran", "dua-arafat", "dua-baha", "dua-eid-maghrib-isha"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("lead,drop,pauses", [(0.0, False, False), (1.3, False, False), (1.3, True, False),
                                              (1.3, False, True)])
def test_js_tracker_matches_python(tmp_path, lead, drop, pauses):
    """lead 1.3: the live display's lookahead. drop: the second du'a streamed is
    missing from the corpus, which exercises the "not in the corpus" state.
    pauses: the reciter falls silent now and then (quiet > 0), with every
    pause rule switched on."""
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in PARITY_DUAS}
    texts = _stream([corpus[k] for k in list(corpus)[3:5]])
    if drop:
        corpus.pop(list(corpus)[4])

    rng = random.Random(3)
    quiet = [rng.choice([0.0, 0.0, 0.0, 0.2, 0.7, 1.5, 3.0]) if pauses else 0.0 for _ in texts]
    rules = {"still_motion_after": 0.3, "lead_cross_quiet": 0.25, "retreat_after": 1.0} if pauses else {}
    tracker = Tracker(CorpusIndex(corpus), TrackerConfig(**rules))
    expected = []
    for t, q in zip(texts, quiet):
        p = tracker.update(t, 1.0, lead, quiet=q)
        expected.append([p.dua, p.segment, p.word])
    js_rules = json.dumps({"stillMotionAfter": 0.3, "leadCrossQuiet": 0.25, "retreatAfter": 1.0} if pauses else {})

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "quiet.json").write_text(json.dumps(quiet), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex, Tracker }} from "{(WEB / 'tracker.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const texts = JSON.parse(readFileSync("{(tmp_path / 'texts.json').as_posix()}", "utf8"));
const quiet = JSON.parse(readFileSync("{(tmp_path / 'quiet.json').as_posix()}", "utf8"));
const tr = new Tracker(new CorpusIndex(corpus), {js_rules});
console.log(JSON.stringify(texts.map((t, i) => {{ const p = tr.update(t, 1.0, {lead}, quiet[i]); return [p.dua, p.segment ?? null, p.word ?? null]; }})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    assert len(got) == len(expected)
    assert sum(e[0] is not None for e in expected) > len(expected) // 2  # mostly locked on, not all blanks
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatches, mismatches[:5]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_highlight_matches_python(tmp_path):
    from dua_recognition.display import Highlight

    rng = random.Random(1)
    # (time, line, word, line range, speed): the line changes now and then, the word wobbles.
    updates, t, line, word = [], 0.0, 0, 3
    for _ in range(60):
        t += rng.uniform(0.6, 1.4)
        if rng.random() < 0.15:
            line, word = line + 1, 0
        word = max(0, min(9, word + rng.choice([-1, 0, 1, 1, 2])))
        updates.append([t, f"d:{line}", word, 0, 10, rng.uniform(0.3, 1.5)])
    ticks = [round(0.1 * k, 1) for k in range(int(t * 10) + 20)]

    hl, expected, k = Highlight(ease=1.0, speed_scale=1.0), [], 0
    for tick in ticks:
        while k < len(updates) and updates[k][0] <= tick:
            u = updates[k]
            hl.update(u[0], "d", int(u[1][2:]), u[2], (u[3], u[4]), u[5])
            k += 1
        expected.append(None if hl.line is None else round(hl.position(tick), 6))

    (tmp_path / "u.json").write_text(json.dumps([updates, ticks]), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ Highlight }} from "{(WEB / 'display.js').as_uri()}";
const [updates, ticks] = JSON.parse(readFileSync("{(tmp_path / 'u.json').as_posix()}", "utf8"));
const hl = new Highlight({{ ease: 1.0, speedScale: 1.0 }});
let k = 0;
console.log(JSON.stringify(ticks.map((tick) => {{
  while (k < updates.length && updates[k][0] <= tick) {{
    const [t, line, word, lo, hi, speed] = updates[k++];
    hl.update(t, line, word, lo, hi, speed);
  }}
  return hl.line == null ? null : Math.round(hl.position(tick) * 1e6) / 1e6;
}})));
"""
    (tmp_path / "hl.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "hl.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    bad = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if (e is None) != (g is None) or (e is not None and abs(e - g) > 1e-5)]
    assert not bad, bad[:5]
