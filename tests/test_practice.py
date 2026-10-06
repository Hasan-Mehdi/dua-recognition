"""The practice checker (src/dua_recognition/practice.py) on synthetic CTC frames: a line read
explains its stretch better than leaving it out, a line left out doesn't, and position counts
(a refrain line left out is caught though its words are heard elsewhere)."""
from dataclasses import replace

import numpy as np

from dua_recognition.align import encode
from dua_recognition.practice import (PracticeConfig, _hyp_score_jit, _hyp_score_numpy, _span_score_jit,
                                      _span_score_numpy, committed_frames, hyp_score, judge_lines, line_llr,
                                      line_score)

C = 64  # CTC columns: 0 blank, letters by code


def _letters(text):
    return encode(text.replace(" ", "")).astype(np.int64)


def _frames(*lines, gap=3, per=2, p=0.9):
    """Frames reading these lines: each letter `per` frames at probability p, `gap` blank frames
    between lines (a breath)."""
    rows = []

    def frame(col):
        x = np.full(C, np.log((1 - p) / (C - 1)))
        x[col] = np.log(p)
        return x

    for ln in lines:
        for c in ln:
            rows += [frame(int(c))] * per
        rows += [frame(0)] * gap
    return np.array(rows)


A, B, Y, Z = (_letters(t) for t in ("الحمد لله رب العالمين", "اياك نعبد", "اهدنا الصراط", "صراط الذين"))
R = _letters("يا مجير")
CFG = PracticeConfig()


def test_a_line_read_scores_above_the_line_left_out():
    read = _frames(A, B, Y)
    skipped = _frames(A, Y)
    assert line_llr(read, A, B, Y, CFG) > 8
    assert line_llr(skipped, A, B, Y, CFG) < -8


def test_a_refrain_left_out_is_caught_where_it_stands():
    # the text: Y R Z R A; the reader reads Y R Z A (the second refrain left out)
    heard = _frames(Y, R, Z, A)
    # judging line "R" between Z and A: its words were heard (after Y), but not here
    tail = heard[sum(2 * len(x) + 3 for x in (Y, R)) - 3 :]
    assert line_llr(tail, Z, R, A, CFG) < -4


def test_talk_between_lines_is_not_the_line():
    talk = _frames(_letters("سلام عليكم"))  # something else said where line B should be
    lp = np.concatenate([_frames(A), talk, _frames(Y)])
    assert line_llr(lp, A, B, Y, CFG) < 0


def test_jit_and_numpy_give_the_same_score():
    if _hyp_score_jit is None:
        return
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(C) * 0.3, size=80))
    r = np.concatenate([A, B]).astype(np.int64)
    seg_end = np.zeros(r.size, dtype=np.bool_)
    seg_end[[A.size - 1, r.size - 1]] = True
    a = _hyp_score_numpy(lp, r, seg_end, A.size, A.size, 1.0, -8.0)
    b = _hyp_score_jit(lp, r, seg_end, A.size, A.size, 1.0, -8.0)
    assert abs(a - b) < 1e-6
    assert abs(hyp_score(lp, [A, B], 1.0, -8.0) - a) < 1e-6


def test_committed_frames_take_each_frame_once():
    # 10 windows of 100 frames, one step every 0.1 s: each step commits the 5 frames ending 0.2 s before its end
    lp = np.zeros((10, 100, C), np.float32)
    for k in range(10):
        lp[k, :, 0] = np.arange(100) + k * 5  # each frame's "time index"
    f, t = committed_frames(lp, np.full(10, 100), np.arange(1, 11) * 0.1 + 2.0)
    assert f.shape[0] == 50
    assert np.all(np.diff(f[:, 0]) == 1)  # consecutive, none twice
    assert np.all(np.diff(t) > 0)


def test_judge_lines_decides_two_lines_on_and_the_rest_at_the_end():
    lines = [A, B, Y, Z]
    lp = _frames(A, B, Z)  # Y left out
    times = (np.arange(lp.shape[0]) + 1) * 0.02
    # the display: line 0, 1, then straight to 3 (decisions at once: no settling, spans from the display)
    steps = [(0.5, 0), (1.2, 1), (2.0, 3), (max(2.5, times[-1] + 0.3), 3)]
    cfg = replace(CFG, settle_s=0.0, span_min_s=0.0)
    recs = judge_lines(lp, times, steps, lines, 0, 3, cfg, lookahead=0.0)
    by = {k: (t, llr, shown) for k, t, llr, shown in recs}
    assert set(by) == {0, 1, 2, 3}
    assert by[1][0] == 2.0  # decided when the display reached line 3
    assert by[2][1] < -4 and not by[2][2]
    assert by[0][1] > 4 and by[1][1] > 4 and by[3][1] > 4


W1 = _letters("رب اغفر لي")
W2 = _letters("وارحمني يا ارحم الراحمين")
W3 = _letters("واهدني سواء السبيل")


def test_several_lines_left_out_are_each_caught():
    # the reader reads A, then jumps to Z: B, Y and W1 left out
    lp = _frames(A, Z)
    for k_line, later in ((B, [Y, W1, Z]), (Y, [W1, Z]), (W1, [Z])):
        assert line_score(lp, A, k_line, 2, later, CFG) < -8


def test_a_line_half_said_when_judged_counts_as_said():
    # the display moved on early: the reader is only through the first words of B
    half = B[: len(_letters("اياك"))]
    lp = _frames(A, half)
    assert line_score(lp, A, B, half.size, [Y, Z], CFG) > 4


def test_a_forgotten_ending_is_caught():
    lp = np.concatenate([_frames(A, B), _frames(np.zeros(0, np.int64), gap=200)])  # then 4 s of silence
    assert line_score(lp, B, Y, 2, [], CFG) < -4
    assert line_score(lp, A, B, 2, [], CFG) > 4


def test_span_score_jit_and_numpy_agree():
    if _span_score_jit is None:
        return
    rng = np.random.default_rng(1)
    lp = np.log(rng.dirichlet(np.ones(C) * 0.3, size=90))
    segs = [A, B, Y]
    r = np.concatenate(segs).astype(np.int64)
    sizes = np.array([x.size for x in segs])
    seg_last = (np.cumsum(sizes) - 1).astype(np.int64)
    seg_first = (seg_last - sizes + 1).astype(np.int64)
    seg_of = np.repeat(np.arange(3), sizes).astype(np.int64)
    opt = np.array([False, True, True])
    for end_from in (0, A.size + 2):
        a = _span_score_numpy(lp, r, seg_of, seg_first, seg_last, opt, A.size, end_from, 1.0, -8.0, -4.0)
        b = _span_score_jit(lp, r, seg_of, seg_first, seg_last, opt, A.size, end_from, 1.0, -8.0, -4.0)
        assert abs(a - b) < 1e-6


def test_the_sound_gate_holds_back_a_left_out_line_when_little_is_heard():
    from dua_recognition.practice import gated

    good = ["heard", "heard", "not sure", "heard"]
    assert gated("left out", list(good), CFG) == "left out"  # 3 of the last 4 heard
    poor = ["heard", "not sure", "not sure", "not sure"]
    assert gated("left out", list(poor), CFG) == "not sure"  # hall echo: little is heard
    assert gated("left out", [], CFG) == "not sure"  # nothing heard yet to vouch for the sound
    h = ["heard"]
    assert gated("left out", h, CFG) == "left out" and h == ["heard"]  # a left-out line isn't history
