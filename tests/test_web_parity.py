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
# Ayat al-Kursi is read word for word in all three (the shared-passage case reads the last two).
SHARED_DUAS = ["duasorg-eid-e-mubahila-1", "duasorg-eid-e-mubahila-3", "duasorg-namaz-e-wahshat"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("lead,drop,pauses,pop,shared", [(0.0, False, False, 0.0, False), (1.3, False, False, 0.0, False),
                                                         (1.3, True, False, 0.0, False), (1.3, False, True, 0.0, False),
                                                         (1.3, False, False, 0.5, False), (1.3, False, False, 0.5, True)])
def test_js_tracker_matches_python(tmp_path, lead, drop, pauses, pop, shared):
    """lead 1.3: the live display's lookahead. drop: the second du'a streamed is
    missing from the corpus, which exercises the "not in the corpus" state.
    pauses: the reciter falls silent now and then (quiet > 0), with every
    pause rule switched on. pop: the prior over du'as by how often each is recited.
    shared: a reading through a passage three texts share, with the shared-passage rules on."""
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in PARITY_DUAS + (SHARED_DUAS if shared else [])}
    texts = _stream([corpus[k] for k in (SHARED_DUAS[1:] if shared else list(corpus)[3:5])])
    if drop:
        corpus.pop(list(corpus)[4])

    rng = random.Random(3)
    quiet = [rng.choice([0.0, 0.0, 0.0, 0.2, 0.7, 1.5, 3.0]) if pauses else 0.0 for _ in texts]
    rules = {"still_motion_after": 0.3, "lead_cross_quiet": 0.25, "retreat_after": 1.0} if pauses else {}
    rules |= {"keep_in_passage": True, "switch_confirm": 2} if shared else {}
    tracker = Tracker(CorpusIndex(corpus), TrackerConfig(**rules, popularity=pop))
    expected = []
    for t, q in zip(texts, quiet):
        p = tracker.update(t, 1.0, lead, quiet=q)
        expected.append([p.dua, p.segment, p.word])
    js_rules = json.dumps({**({"stillMotionAfter": 0.3, "leadCrossQuiet": 0.25, "retreatAfter": 1.0} if pauses else {}),
                           "popularity": pop, **({"keepInPassage": True, "switchConfirm": 2} if shared else {})})

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar, "rec": d.recordings,
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


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_tracker_matches_python_on_a_respelled_shared_passage_and_a_tap(tmp_path):
    """Ayat al-Kursi and the texts that share it (one spelled differently), then an
    "I'm here" tap into another du'a mid-stream (Tracker.seek)."""
    all_duas = load_all()
    shared = ["duasorg-namaz-e-wahshat", "duasorg-eid-e-mubahila-3", "sahifa-54"]
    corpus = {k: all_duas[k] for k in shared + PARITY_DUAS[:6]}
    texts = _stream([corpus["duasorg-namaz-e-wahshat"]], seed=5)
    seek_at, target = len(texts) // 2, ("duasorg-eid-e-mubahila-3", corpus["duasorg-eid-e-mubahila-3"].segments[6].id)
    texts += _stream([corpus["duasorg-eid-e-mubahila-3"]], seed=6)[40:70]

    tracker = Tracker(CorpusIndex(corpus))
    expected = []
    for i, t in enumerate(texts):
        if i == seek_at:
            p = tracker.seek(*target)
            expected.append(["seek", p.dua, p.segment, p.word])
        p = tracker.update(t, 1.0, 1.3)
        expected.append([p.dua, p.segment, p.word, sorted(p.same_as)])
    assert sum(e[0] == "duasorg-namaz-e-wahshat" for e in expected) > seek_at // 2  # shown, not withheld

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar, "rec": d.recordings,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex, Tracker }} from "{(WEB / 'tracker.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const texts = JSON.parse(readFileSync("{(tmp_path / 'texts.json').as_posix()}", "utf8"));
const tr = new Tracker(new CorpusIndex(corpus));
const out = [];
texts.forEach((t, i) => {{
  if (i === {seek_at}) {{ const p = tr.seek({json.dumps(target[0])}, {target[1]}); out.push(["seek", p.dua, p.segment, p.word]); }}
  const p = tr.update(t, 1.0, 1.3);
  out.push([p.dua, p.segment ?? null, p.word ?? null, [...(p.sameAs || [])].sort()]);
}});
console.log(JSON.stringify(out));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert len(got) == len(expected) and not mismatches, mismatches[:5]


# Every display rule from the ordinary-voice sessions on at once (tracker.py,
# lead_cross_words ... keep_dua_confidence), in both spellings of the config.
STABLE_PY = {"lead_cross_words": 2, "seek_pins_line": True, "back_confirm": 2, "null_rate_locked": 0.6,
             "keep_dua_confidence": 0.5, "min_dua_confidence": 0.95, "still_motion_after": 0.3, "lead_cross_quiet": 0.25, "retreat_after": 1.0}
STABLE_JS = {"leadCrossWords": 2, "seekPinsLine": True, "backConfirm": 2, "nullRateLocked": 0.6,
             "keepDuaConfidence": 0.5, "minDuaConfidence": 0.95, "stillMotionAfter": 0.3, "leadCrossQuiet": 0.25, "retreatAfter": 1.0}


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_tracker_matches_python_with_the_stability_rules(tmp_path):
    """Short lines (Ayat al-Kursi), misheard words, pauses, taps into lines ahead
    and behind, and a du'a missing from the corpus at the end."""
    all_duas = load_all()
    shared = ["duasorg-namaz-e-wahshat", "duasorg-eid-e-mubahila-3", "sahifa-54"]
    corpus = {k: all_duas[k] for k in shared + PARITY_DUAS[:6]}
    texts = _stream([corpus["duasorg-namaz-e-wahshat"]], seed=7) + _stream([all_duas["dua-kumayl"]], seed=8)[:25]
    texts[36:40] = ["شهور النامين يوم يام", "بمشهد و صيق السوء", "و سيكون السماوات", "كلمات شيء"]  # misheard
    rng = random.Random(4)
    quiet = [rng.choice([0.0, 0.0, 0.0, 0.2, 0.7, 1.5]) for _ in texts]
    segs = [s.id for s in corpus["duasorg-namaz-e-wahshat"].segments]
    taps = {12: segs[6], 20: segs[3], 30: segs[12]}

    tracker = Tracker(CorpusIndex(corpus), TrackerConfig(**STABLE_PY))
    expected = []
    for i, (t, q) in enumerate(zip(texts, quiet)):
        if i in taps:
            p = tracker.seek("duasorg-namaz-e-wahshat", taps[i])
            expected.append(["seek", p.dua, p.segment, p.word])
        p = tracker.update(t, 1.0, 1.3, quiet=q)
        expected.append([p.dua, p.segment, p.word])
    assert sum(e[0] == "duasorg-namaz-e-wahshat" for e in expected) > len(texts) // 3

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar, "rec": d.recordings,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps([texts, quiet, taps], ensure_ascii=False), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex, Tracker }} from "{(WEB / 'tracker.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const [texts, quiet, taps] = JSON.parse(readFileSync("{(tmp_path / 'in.json').as_posix()}", "utf8"));
const tr = new Tracker(new CorpusIndex(corpus), {json.dumps(STABLE_JS)});
const out = [];
texts.forEach((t, i) => {{
  if (taps[i] !== undefined) {{ const p = tr.seek("duasorg-namaz-e-wahshat", taps[i]); out.push(["seek", p.dua, p.segment, p.word]); }}
  const p = tr.update(t, 1.0, 1.3, quiet[i]);
  out.push([p.dua, p.segment ?? null, p.word ?? null]);
}});
console.log(JSON.stringify(out));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert len(got) == len(expected) and not mismatches, mismatches[:5]



@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_tracker_matches_python_through_stops(tmp_path):
    """The stop rules (docs/results/stops.md): the page's live quiet (quiet_now), the
    pauses since the last update (paused), a step back within the line and the
    display catching up with the evidence while the reciter is silent."""
    all_duas = load_all()
    shared = ["duasorg-namaz-e-wahshat", "duasorg-eid-e-mubahila-3", "sahifa-54"]
    corpus = {k: all_duas[k] for k in shared + PARITY_DUAS[:6]}
    texts = _stream([corpus["duasorg-namaz-e-wahshat"]], seed=9) + _stream([all_duas["dua-kumayl"]], seed=10)[:20]
    rng = random.Random(11)
    quiet = [rng.choice([0.0, 0.0, 0.0, 0.2, 0.7, 1.5, 3.0]) for _ in texts]
    quiet_now = [rng.choice([None, 0.0, 0.0, 0.1, 0.5, 1.2, 2.5]) for _ in texts]
    paused = [rng.choice([None, 0.0, 0.0, 0.3, 0.6, 1.0]) for _ in texts]
    dts = [rng.choice([1.0, 1.0, 1.25, 2.5]) for _ in texts]
    py = {**STABLE_PY, "retreat_in_line": True, "still_catch_up": True, "pause_motion": True}
    js = {**STABLE_JS, "retreatInLine": True, "stillCatchUp": True, "pauseMotion": True}

    tracker = Tracker(CorpusIndex(corpus), TrackerConfig(**py))
    expected = []
    for t, q, qn, pz, dt in zip(texts, quiet, quiet_now, paused, dts):
        p = tracker.update(t, dt, dt + 0.3, quiet=q, quiet_now=qn, paused=pz)
        expected.append([p.dua, p.segment, p.word])
    assert sum(e[0] == "duasorg-namaz-e-wahshat" for e in expected) > len(texts) // 3

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar, "rec": d.recordings,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps([texts, quiet, quiet_now, paused, dts], ensure_ascii=False),
                                      encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex, Tracker }} from "{(WEB / 'tracker.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const [texts, quiet, quietNow, paused, dts] = JSON.parse(readFileSync("{(tmp_path / 'in.json').as_posix()}", "utf8"));
const tr = new Tracker(new CorpusIndex(corpus), {json.dumps(js)});
console.log(JSON.stringify(texts.map((t, i) => {{
  const p = tr.update(t, dts[i], dts[i] + 0.3, quiet[i], quietNow[i], paused[i]);
  return [p.dua, p.segment ?? null, p.word ?? null];
}})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert len(got) == len(expected) and not mismatches, mismatches[:5]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_highlight_matches_python_through_stops(tmp_path):
    """The glide stopped and restarted between updates (pace), and updates that
    may take it back (back)."""
    from dua_recognition.display import Highlight

    rng = random.Random(2)
    events, t, line, word = [], 0.0, 0, 3
    for _ in range(80):
        t += rng.uniform(0.3, 2.6)
        if rng.random() < 0.3:
            events.append(["pace", round(t, 3), rng.choice([0.0, 0.0, 1.1])])
            continue
        if rng.random() < 0.12:
            line, word = line + 1, 0
        word = max(0, min(9, word + rng.choice([-2, -1, 0, 1, 1, 2])))
        events.append(["update", round(t, 3), f"d:{line}", word, 0, 10, rng.choice([0.0, 0.7, 1.4]),
                       rng.random() < 0.3])
    ticks = [round(0.1 * k, 1) for k in range(int(t * 10) + 20)]

    hl, expected, k = Highlight(ease=1.0, speed_scale=1.2), [], 0
    for tick in ticks:
        while k < len(events) and events[k][1] <= tick:
            e = events[k]
            if e[0] == "pace":
                hl.pace(e[1], e[2])
            else:
                hl.update(e[1], "d", int(e[2][2:]), e[3], (e[4], e[5]), e[6], back=e[7])
            k += 1
        expected.append(None if hl.line is None else round(hl.position(tick), 6))

    (tmp_path / "u.json").write_text(json.dumps([events, ticks]), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ Highlight }} from "{(WEB / 'display.js').as_uri()}";
const [events, ticks] = JSON.parse(readFileSync("{(tmp_path / 'u.json').as_posix()}", "utf8"));
const hl = new Highlight({{ ease: 1.0, speedScale: 1.2 }});
let k = 0;
console.log(JSON.stringify(ticks.map((tick) => {{
  while (k < events.length && events[k][1] <= tick) {{
    const e = events[k++];
    if (e[0] === "pace") hl.pace(e[1], e[2]);
    else hl.update(e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
  }}
  return hl.line == null ? null : Math.round(hl.position(tick) * 1e6) / 1e6;
}})));
"""
    (tmp_path / "hl.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "hl.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    bad = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got))
           if (e is None) != (g is None) or (e is not None and abs(e - g) > 1e-5)]
    assert not bad, bad[:5]
    assert any(e is not None and e2 is not None and e2 < e - 0.5 for e, e2 in zip(expected, expected[1:]))  # went back


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_live_quiet_matches_python(tmp_path):
    """The page's stop detector on the audio as it arrives (asr.LiveQuiet, gate.js LiveQuiet)."""
    import numpy as np

    from dua_recognition.asr import LiveQuiet

    rng = np.random.default_rng(0)
    sr = 16000
    parts = []
    for _ in range(14):  # bursts of "voice" at -14 dBFS, gaps of hiss rising like a phone's gain control
        n = int(rng.uniform(0.2, 1.6) * sr)
        parts.append((rng.standard_normal(n) * 10 ** (-14 / 20)).astype(np.float32))
        n = int(rng.uniform(0.05, 1.5) * sr)
        parts.append((rng.standard_normal(n) * 10 ** (-48 / 20) * np.linspace(1, 6, n)).astype(np.float32))
    y = np.concatenate(parts)
    chunks = [y[a : a + 4000] for a in range(0, y.size, 4000)]
    voice = [None if i < 3 else -14.0 + (i % 5) for i in range(len(chunks))]

    lq, expected = LiveQuiet(), []
    for c, v in zip(chunks, voice):
        lq.voice_db = v
        lq.push(c)
        expected.append(None if lq.quiet is None else round(lq.quiet, 6))
    assert any(q is not None and q > 0.5 for q in expected) and 0.0 in expected

    y.astype("<f4").tofile(tmp_path / "y.f32")
    (tmp_path / "v.json").write_text(json.dumps(voice), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ LiveQuiet }} from "{(WEB / 'gate.js').as_uri()}";
const buf = readFileSync("{(tmp_path / 'y.f32').as_posix()}");
const y = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const voice = JSON.parse(readFileSync("{(tmp_path / 'v.json').as_posix()}", "utf8"));
const lq = new LiveQuiet();
const out = [];
for (let i = 0, a = 0; a < y.length; i++, a += 4000) {{
  lq.voiceDb = voice[i];
  lq.push(y.subarray(a, Math.min(y.length, a + 4000)));
  out.push(lq.quiet == null ? null : Math.round(lq.quiet * 1e6) / 1e6);
}}
console.log(JSON.stringify(out));
"""
    (tmp_path / "lq.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "lq.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(out.stdout) == expected


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_tracker_matches_python_on_lines_out_of_order(tmp_path):
    """A reader who jumps around one du'a (p_line_jump: a jump to any line's start within the du'a)."""
    all_duas = load_all()
    corpus = {k: all_duas[k] for k in PARITY_DUAS[:6] + ["dua-kumayl"]}
    kumayl = corpus["dua-kumayl"]
    rng = random.Random(11)
    order = list(range(0, 6)) + [rng.randrange(6, 120) for _ in range(8)]
    texts = []
    for i in order:  # each line as windows sliding over its words, then the next line read
        words = kumayl.segments[i].arabic.split()
        for k in range(0, len(words), 2):
            texts.append(" ".join(words[max(0, k - 5) : k + 1]))
    cfg_py, cfg_js = {"p_line_jump": 0.05}, {"pLineJump": 0.05}
    tracker = Tracker(CorpusIndex(corpus), TrackerConfig(**cfg_py))
    expected = []
    for t in texts:
        p = tracker.update(t, 1.0, 0.0)
        mass = round(tracker.line_mass(p.word), 9) if p.word is not None else None  # the follower's jumpMass
        expected.append([p.dua, p.segment, p.word, mass])
    assert len({e[1] for e in expected if e[0] == "dua-kumayl"}) > 8  # it went to many lines

    payload = [{"id": d.id, "name_en": d.name_en, "name_ar": d.name_ar, "rec": d.recordings,
                "segments": [{"id": s.id, "ar": s.arabic} for s in d.segments]} for d in corpus.values()]
    (tmp_path / "corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ CorpusIndex, Tracker }} from "{(WEB / 'tracker.js').as_uri()}";
const corpus = JSON.parse(readFileSync("{(tmp_path / 'corpus.json').as_posix()}", "utf8"));
const texts = JSON.parse(readFileSync("{(tmp_path / 'in.json').as_posix()}", "utf8"));
const tr = new Tracker(new CorpusIndex(corpus), {json.dumps(cfg_js)});
console.log(JSON.stringify(texts.map((t) => {{
  const p = tr.update(t, 1.0, 0.0);
  const mass = p.word == null ? null : Math.round(tr.lineMass(p.word) * 1e9) / 1e9;
  return [p.dua, p.segment ?? null, p.word ?? null, mass];
}})));
"""
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    out = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    assert len(got) == len(expected) and not mismatches, mismatches[:5]
