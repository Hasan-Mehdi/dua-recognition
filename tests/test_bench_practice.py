"""Practice mode on the scenario bench (scripts/bench.py): the mistake scenarios keep their truth
consistent, and the v0 checker marks what the display leaves out."""
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402
from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.corpus import Dua, Segment  # noqa: E402

WORDS = ("رحمة قدرة عظمة سلطان كرامة جلالة حكمة نعمة رأفة عزة مغفرة قوة هداية نصرة حماية "
         "كفاية عافية سلامة توبة رضوان بركة طاعة ذكرى صدقة شفاعة علامة قربة منزلة وسيلة دعوة "
         "حاجة فرحة سكينة نجاة كفالة عبادة إجابة وقاية رعاية ولاية").split()


def _corpus(n_lines=10, per=4):
    lines = [" ".join(WORDS[i * per : (i + 1) * per]) for i in range(n_lines)]
    return CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(lines)])})


def _source(ix):
    """Every word 0.4 s, 0.1 s between words, 0.6 s between lines."""
    lines, t = [], 1.0
    for seg in sorted(set(int(s) for s in ix.word_segment)):
        ws = []
        for g in np.flatnonzero(ix.word_segment == seg):
            ws.append([int(g), round(t, 3), round(t + 0.4, 3)])
            t += 0.5
        t += 0.5
        lines.append({"seg": seg, "words": ws, "from": ws[0][1] - 0.05, "to": ws[-1][2] + 0.1})
    return {"sid": "test:a:0", "lane": "studio", "audio": "x.wav", "dua": "d", "voice": "test:v", "lines": lines,
            "tags": {}}


def test_skipword_cuts_the_word_out_of_truth_and_audio():
    ix = _corpus()
    src = _source(ix)
    it = bench.sc_skipword(src, ix, random.Random(3)).item("skipword")
    cut = [e["word"][1] for e in it["errors"]]
    assert cut, "some line has a word to cut"
    read = [w for _, w, _, _ in it["words"]]
    assert not set(cut) & set(read)
    assert len(read) == ix.n_words - len(cut)
    for w in cut:  # never a line's first or last word
        seg = ix.word_segment[w]
        ws = np.flatnonzero(ix.word_segment == seg)
        assert ws[0] < w < ws[-1]
    starts = [a for _, _, a, _ in it["words"]]
    assert starts == sorted(starts) and it["words"][-1][3] <= it["duration"] + 1e-6
    faded = [op for op in it["ops"] if len(op) == 5]
    assert len(faded) == len(it["ops"]) == len(cut) + 1  # one span between each pair of cuts
    # each cut sits in the gap: the output skips exactly the cut span of source time
    src_s = sum(op[2] - op[1] for op in it["ops"])
    assert abs(src_s - it["duration"]) < 1e-6


def test_skipline_leaves_the_line_out():
    ix = _corpus()
    it = bench.sc_skipline(_source(ix), ix, random.Random(1)).item("skipline")
    skipped = {e["seg"] for e in it["errors"]}
    assert skipped
    read_segs = {int(ix.word_segment[w]) for _, w, _, _ in it["words"]}
    assert not skipped & read_segs
    assert sum(e["kind"] == "skip" for e in it["events"]) == len(skipped)


def test_ending_stops_early_with_the_target_to_the_end():
    ix = _corpus()
    src = _source(ix)
    it = bench.sc_ending(src, ix, random.Random(2)).item("ending")
    unread = it["errors"][0]["segs"]
    read_segs = {int(ix.word_segment[w]) for _, w, _, _ in it["words"]}
    assert unread and not set(unread) & read_segs
    assert it["target"] == ["d", src["lines"][0]["seg"], src["lines"][-1]["seg"]]
    assert it["duration"] >= it["words"][-1][3] + 8.0 - 1e-6


def test_fade_ramps_the_cut_ends_only():
    y = bench.fade(np.ones(2000, np.float32), 0.008, 0.008)
    n = int(0.008 * bench.SR)
    assert y[0] == 0.0 and y[-1] == 0.0
    assert np.all(np.diff(y[:n]) >= 0) and np.allclose(y[n:-n], 1.0)


def _shows(ix, plan, t0=0.0, dt=0.1):
    """Display steps: (line index, word in line, steps) -> stream_updates rows."""
    _, _, first_w = bench._dua_lines(ix, 0)
    ups, t = [], t0
    for li, wi, n in plan:
        w = first_w[li] + wi
        for _ in range(n):
            ups.append((round(t, 3), "d", int(ix.word_segment[w]), w, None, 0.0))
            t += dt
    return ups


def _item(ix, read_lines, scenario="flow", errors=(), events=(), target=None, duration=60.0):
    lo, _ = ix.dua_word_span[0]
    words, t = [], 0.0
    for li in read_lines:
        for w in np.flatnonzero(ix.word_segment == li + 1):
            words.append(["d", int(w - lo), round(t, 3), round(t + 0.4, 3)])
            t += 0.5
    it = {"dua": "d", "words": words, "duration": duration, "scenario": scenario, "errors": list(errors),
          "events": list(events)}
    if target:
        it["target"] = target
    return it


def test_v0_marks_a_skipped_line_two_lines_on_and_an_ending_at_the_end():
    ix = _corpus()
    seg = lambda li: li + 1  # noqa: E731
    # the display keeps time with the reader (2 s a line), who leaves out line 2 after 4.0 s
    ups = _shows(ix, [(0, 0, 10), (0, 3, 10), (1, 0, 10), (1, 3, 10), (3, 0, 10), (3, 3, 10), (4, 0, 20),
                      (5, 0, 20)])
    it = _item(ix, [0, 1, 3, 4, 5], scenario="skipline",
               errors=[{"kind": "skipline", "t": 4.0, "dua": "d", "seg": seg(2)},
                       {"kind": "ending", "t": 13.0, "dua": "d", "segs": [seg(6), seg(7)]}],
               target=["d", seg(0), seg(7)])
    tg = bench.practice_target(it, ix)
    assert tg == (0, 0, 7)
    res = bench.practice_v0(ups, ix, tg)
    assert set(res["line"]) == {2, 6, 7}
    t_line4 = next(u[0] for u in ups if u[2] == seg(4))
    assert res["line"][2][0] == t_line4  # decided when the display reached line 4, not before
    m = bench.practice_metrics(it, res, ix, tg)
    assert m["false_lines"] == 0
    assert m["missed"] == {"skipline": (1, 1), "ending": (2, 2)}
    assert m["decide"] == [0.1]  # line 3 (read after the gap) ends at 5.9 s; the display reaches line 4 at 6.0 s


def test_v0_clears_a_mark_the_display_shows_later_and_counts_the_flip():
    ix = _corpus()
    ups = _shows(ix, [(0, 0, 5), (2, 0, 5), (3, 0, 5), (1, 0, 5), (1, 3, 5), (2, 0, 5), (3, 0, 5)])
    tg = (0, 0, 3)
    res = bench.practice_v0(ups, ix, tg)
    assert res["line"][1][1] is not None  # marked at line 3, cleared when shown
    m = bench.practice_metrics(_item(ix, [0, 1, 2, 3]), res, ix, tg)
    assert m["false_lines"] == 0 and m["flips_line"] == 1


def test_v0_marks_a_word_the_display_moves_past():
    ix = _corpus()
    _, _, first_w = bench._dua_lines(ix, 0)
    ups = _shows(ix, [(0, 0, 3), (0, 2, 3), (0, 3, 3), (1, 1, 3), (1, 2, 3), (1, 3, 3)])
    res = bench.practice_v0(ups, ix, (0, 0, 1))
    assert set(res["word"]) == {first_w[0] + 1, first_w[1]}  # skipped inside line 0, and line 1's first word
    it = _item(ix, [0, 1], scenario="skipword", errors=[{"kind": "skipword", "t": 0.5, "word": ["d", 1]}])
    it["words"] = [w for w in it["words"] if w[1] != 1]
    m = bench.practice_metrics(it, res, ix, (0, 0, 1))
    assert m["skipword"] == (1, 1)
    assert m["false_words"] == 1  # line 1's first word was read


def test_a_line_with_the_same_words_counts_as_heard():
    lines = [" ".join(WORDS[i * 4 : (i + 1) * 4]) for i in range(6)]
    lines[4] = lines[1]  # a refrain: line 4 reads what line 1 reads
    ix = CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(lines)])})
    assert bench._same_lines(ix, 0) == [0, 1, 2, 3, 1, 5]
    # the reader reads 3, 4, 5; the display shows 3, then the earlier copy of line 4 (line 1), then 5
    ups = _shows(ix, [(3, 0, 10), (1, 0, 10), (5, 0, 10)])
    res = bench.practice_v0(ups, ix, (0, 3, 5))
    assert 4 not in res["line"]


def test_false_marks_before_the_display_finds_the_reader_are_split_out():
    ix = _corpus()
    # the reader reads lines 0-5 (2 s each); the display waits on line 2 from the start, so it is
    # right from 4.0 s, when the reader gets there, and follows on
    ups = _shows(ix, [(2, 0, 60), (3, 0, 20), (4, 0, 20), (5, 0, 20)])
    it = _item(ix, [0, 1, 2, 3, 4, 5], duration=12.0)
    res = bench.practice_v0(ups, ix, (0, 0, 5))
    m = bench.practice_metrics(it, res, ix, (0, 0, 5), ups)
    assert m["not_judged_lines"] == 2  # lines 0 and 1: before the first line the display showed
    assert m["lock_s"] == 4.0 and m["false_lines"] == 0 and m["minutes_after_lock"] == 8.0 / 60


def test_observer_leaves_the_steps_as_they_were():
    ix = _corpus()
    ups = _shows(ix, [(0, 0, 3), (2, 0, 3), (1, 0, 3)])
    before = [tuple(u) for u in ups]
    bench.practice_v0(ups, ix, (0, 0, 2))
    assert [tuple(u) for u in ups] == before
