"""The browser tracker (web/tracker.js) must follow exactly like the Python one."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from dua_recognition.align import CorpusIndex
from dua_recognition.corpus import load_all
from dua_recognition.tracker import Tracker

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


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_tracker_matches_python(tmp_path):
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in list(all_duas)[:12]}
    texts = _stream([corpus[k] for k in list(corpus)[3:5]])

    tracker = Tracker(CorpusIndex(corpus))
    expected = []
    for t in texts:
        p = tracker.update(t, 1.0)
        expected.append([p.dua, p.segment, p.word])

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex, Tracker }} from "{(WEB / 'tracker.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const texts = JSON.parse(readFileSync("{(tmp_path / 'texts.json').as_posix()}", "utf8"));
const tr = new Tracker(new CorpusIndex(corpus));
console.log(JSON.stringify(texts.map((t) => {{ const p = tr.update(t, 1.0); return [p.dua, p.segment ?? null, p.word ?? null]; }})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    assert len(got) == len(expected)
    assert sum(e[0] is not None for e in expected) > len(expected) // 2  # mostly locked on, not all blanks
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert not mismatches, mismatches[:5]
