import numpy as np

from dua_recognition.align import _ALPHABET, CorpusIndex
from dua_recognition.corpus import Dua, Segment
from dua_recognition.follower import FollowerConfig, LocalFollower

LINES = ["بسم الله الرحمن الرحيم", "الحمد لله رب العالمين", "الرحمن الرحيم مالك يوم الدين"]
N_COLS = len(_ALPHABET) + 1


def _frames(letters, per_letter=3, blanks=2):
    """Fake CTC posteriors: each letter peaks for a few frames, with blanks between."""
    rows = []
    for c in letters:
        rows += [c] * per_letter + [0] * blanks
    lp = np.full((len(rows), N_COLS), -12.0, dtype=np.float32)
    lp[np.arange(len(rows)), rows] = -0.01
    return lp


def test_follows_words_and_holds_through_a_pause():
    ix = CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(LINES)])})
    fol = LocalFollower(ix, FollowerConfig(window_s=2.0, line_confirm=1))  # one step per word
    spoken = []  # letters heard so far
    t = 0.0
    for w in range(ix.n_words):
        a, b = np.flatnonzero(ix.letter_word == w)[[0, -1]]
        spoken += list(ix.letters[a : b + 1])
        t += 0.5
        shown = fol.step(_frames(spoken)[-100:], t, hmm_word=max(0, w - 2))  # the tracker lags a little
        assert shown == w  # the word whose last letter was just heard
    # The reciter stops: only blanks arrive. The follower stays on the last word.
    silence = np.full((60, N_COLS), -12.0, dtype=np.float32)
    silence[:, 0] = -0.01
    lp = np.concatenate([_frames(spoken), silence])[-100:]
    assert fol.step(lp, t + 1.0, hmm_word=ix.n_words - 3) == ix.n_words - 1


def test_a_new_line_is_entered_once_it_wins_twice():
    ix = CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(LINES)])})
    fol = LocalFollower(ix, FollowerConfig(window_s=2.0))  # line_confirm 2 (default)
    first = int(np.flatnonzero(np.diff(ix.word_segment) != 0)[0]) + 1  # the second line's first word
    a, b = np.flatnonzero(ix.letter_word == first)[[0, -1]]
    spoken = list(ix.letters[: b + 1])
    assert fol.step(_frames(list(ix.letters[:a]))[-100:], 1.0, hmm_word=first - 1) == first - 1
    assert fol.step(_frames(spoken)[-100:], 1.1, hmm_word=first - 1) == first - 1  # heard once: held
    assert fol.step(_frames(spoken)[-100:], 1.2, hmm_word=first - 1) == first  # twice: entered


def test_reanchors_when_the_tracker_disagrees_for_long():
    ix = CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(LINES)])})
    fol = LocalFollower(ix, FollowerConfig(reset_after=1.0))
    silence = np.full((20, N_COLS), -12.0, dtype=np.float32)
    silence[:, 0] = -0.01
    assert fol.step(silence, 0.0, hmm_word=0) == 0
    last = ix.n_words - 1  # two lines on
    assert fol.step(silence, 0.5, hmm_word=last) == 0  # not yet
    assert fol.step(silence, 1.6, hmm_word=last) == last
    assert fol.step(silence, 1.8, hmm_word=None) is None  # tracker lost the du'a: hand back


def _index():
    return CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(LINES)])})


def _upto(ix, w):
    """Frames of the letters of words 0..w, followed by a short blank."""
    return _frames(list(ix.letters[: np.flatnonzero(ix.letter_word == w)[-1] + 1]))


def _blanks(n):
    lp = np.full((n, N_COLS), -12.0, dtype=np.float32)
    lp[:, 0] = -0.01
    return lp


def test_numba_end_scores_matches_numpy():
    from dua_recognition.ctc_align import end_scores, end_scores_numpy

    rng = np.random.default_rng(0)
    for i in range(500):
        T, J = int(rng.integers(1, 60)), int(rng.integers(1, 120))
        lp = np.log(rng.dirichlet(np.full(N_COLS, 0.3), size=T) + 1e-9).astype(np.float32)
        r = rng.integers(1, N_COLS, size=J)
        if i % 3 == 0:
            r[1::2] = r[: J // 2]  # doubled letters
        np.testing.assert_allclose(end_scores(lp, r), end_scores_numpy(lp, r), rtol=1e-5, atol=1e-3)


# A letter that is in none of the lines: the onset rule fires on any letters after a
# gap, while ordinary scoring can't place it and holds.
ALIEN = next(c for c in range(1, N_COLS) if c not in set(np.concatenate([_index().letters])))


def test_onset_moves_to_the_next_line_after_a_gap():
    ix = _index()
    last = int(np.flatnonzero(ix.word_segment == 1)[-1])  # last word of line 1
    lp = _upto(ix, last)
    alien = _frames([ALIEN], per_letter=3, blanks=0)
    x = np.concatenate([lp, _blanks(15), alien])
    off = LocalFollower(ix, FollowerConfig())
    assert off.step(lp, 1.0, hmm_word=last) == last
    assert off.step(x, 1.2, hmm_word=last) == last  # off: the alien letter can't move it
    on = LocalFollower(ix, FollowerConfig(onset_gap=0.2))
    assert on.step(lp, 1.0, hmm_word=last) == last
    assert on.step(x, 1.2, hmm_word=last) == last + 1  # first word of line 2
    # too short a gap: no onset
    short = LocalFollower(ix, FollowerConfig(onset_gap=0.2))
    short.step(lp, 1.0, hmm_word=last)
    assert short.step(np.concatenate([lp, _blanks(5), alien]), 1.2, hmm_word=last) == last
    # onset_confirm: only once the tracker's lead is on a later line
    conf = LocalFollower(ix, FollowerConfig(onset_gap=0.2, onset_confirm=True))
    conf.step(lp, 1.0, hmm_word=last)
    assert conf.step(x, 1.2, hmm_word=last, lead_word=last) == last
    assert conf.step(x, 1.4, hmm_word=last, lead_word=last + 2) == last + 1


def test_no_onset_mid_line():
    ix = _index()
    mid = 1  # second word of line 1 (line 1 has four words)
    lp = _upto(ix, mid)
    fol = LocalFollower(ix, FollowerConfig(onset_gap=0.2))
    assert fol.step(lp, 1.0, hmm_word=mid) == mid
    x = np.concatenate([lp, _blanks(15), _frames([ALIEN], per_letter=3, blanks=0)])
    assert fol.step(x, 1.2, hmm_word=mid) == mid


def test_confirm_steps_delays_a_backward_move_by_one_step():
    ix = _index()
    ahead, back = 6, 4
    for n, want in ((1, [back, back]), (2, [ahead, back])):
        fol = LocalFollower(ix, FollowerConfig(beta_back=0.0, confirm_steps=n, back_confirm=1))
        assert fol.step(_upto(ix, ahead), 1.0, hmm_word=ahead) == ahead
        # the audio now ends on word 4: a move back two words
        got = [fol.step(_upto(ix, back), 1.2 + 0.2 * k, hmm_word=ahead) for k in range(2)]
        assert got == want


def test_one_word_steps_back_are_ignored_and_longer_ones_confirmed():
    ix = _index()
    fol = LocalFollower(ix, FollowerConfig(beta_back=0.0, back_confirm=2))  # back_min 2 (default)
    assert fol.step(_upto(ix, 6), 1.0, hmm_word=6) == 6
    assert fol.step(_upto(ix, 5), 1.2, hmm_word=6) == 6  # one word back: noise, held
    got = [fol.step(_upto(ix, 4), 1.4 + 0.2 * k, hmm_word=6) for k in range(2)]
    assert got == [6, 4]  # two words back: once it wins twice
    # The default since 2026-10-01: four steps in a row (0.4 s at the phone's 0.1 s step).
    fol = LocalFollower(ix, FollowerConfig(beta_back=0.0))
    assert fol.step(_upto(ix, 6), 1.0, hmm_word=6) == 6
    got = [fol.step(_upto(ix, 4), 1.1 + 0.1 * k, hmm_word=6) for k in range(4)]
    assert got == [6, 6, 6, 4]


def test_leaps_to_a_word_out_of_reach_once_it_wins_twice():
    ix = _index()
    fol = LocalFollower(ix, FollowerConfig(leap_margin=4.0, leap_confirm=2))
    assert fol.step(_upto(ix, 1), 1.0, hmm_word=1) == 1
    far = 7  # the reciter is well past max_jump (2) from word 1
    x = _upto(ix, far)[-60:]
    first = fol.step(x, 1.2, hmm_word=1)
    assert first <= 3  # within reach on the first step
    assert fol.step(x, 1.4, hmm_word=1) == far


def test_end_scores_with_starts_begins_only_where_allowed():
    """A window that holds only the middle of a word: free to start anywhere it scores well
    there; allowed to start only at word beginnings it can't, unless it pays for the
    first letters it never heard."""
    from dua_recognition.ctc_align import end_scores, end_scores_numpy

    r = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)  # one word, six letters
    lp = _frames([8, 9, 10])  # its last three letters only
    starts = np.zeros(r.size, dtype=bool)
    starts[0] = True
    free, fixed = end_scores(lp, r), end_scores(lp, r, starts)
    assert free[-1] > -1.0  # the tail alone explains the audio
    assert fixed[-1] < free[-1] - 20  # from the first letter, three letters were never heard
    np.testing.assert_allclose(fixed, end_scores_numpy(lp, r, starts), rtol=1e-5, atol=1e-3)


def test_reanchors_forward_when_the_tracker_stays_well_ahead():
    """Fallen behind, then silence: the tracker four words ahead for 2 s takes the follower there."""
    ix = _index()
    fol = LocalFollower(ix, FollowerConfig(ahead_words_reanchor=4, ahead_after=2.0))
    assert fol.step(_upto(ix, 1), 1.0, hmm_word=1) == 1
    assert fol.step(_blanks(60), 1.5, hmm_word=5) == 1  # ahead, but not for long yet
    assert fol.step(_blanks(60), 3.0, hmm_word=5) == 1
    assert fol.step(_blanks(60), 3.6, hmm_word=5) == 5
    back = LocalFollower(ix, FollowerConfig(ahead_words_reanchor=4, ahead_after=2.0))
    assert back.step(_upto(ix, 6), 1.0, hmm_word=6) == 6
    assert back.step(_blanks(60), 4.0, hmm_word=2) == 6  # never backward
    assert back.step(_blanks(60), 6.0, hmm_word=2) == 6


JUMP_LINES = ["بسم الله الرحمن الرحيم", "قل هو الله احد", "الله الصمد", "لم يلد ولم يولد", "ولم يكن له كفوا احد",
              "قل اعوذ برب الفلق", "من شر ما خلق", "ومن شر غاسق اذا وقب", "ومن شر النفاثات في العقد",
              "ومن شر حاسد اذا حسد", "قل اعوذ برب الناس", "ملك الناس", "اله الناس", "من شر الوسواس الخناس",
              "الذي يوسوس في صدور الناس", "من الجنة والناس", "انا اعطيناك الكوثر", "فصل لربك وانحر",
              "ان شانئك هو الابتر", "اذا جاء نصر الله والفتح", "ورايت الناس يدخلون في دين الله افواجا",
              "فسبح بحمد ربك واستغفره انه كان توابا"]


def _jump_setup():
    ix = CorpusIndex({"d": Dua("d", "D", "", [Segment(i + 1, t) for i, t in enumerate(JUMP_LINES)])})
    seg = ix.word_segment
    line = {s: [w for w in range(ix.n_words) if seg[w] == s] for s in set(seg.tolist())}
    letters = lambda ws: list(ix.letters[np.isin(ix.letter_word, ws)])  # noqa: E731
    return ix, line, letters


def test_jumps_to_a_line_far_out_of_reach():
    """Read line 1, then jump to line 21: the jump search finds it; without it the follower stays."""
    ix, line, letters = _jump_setup()
    heard = _frames(letters(line[1]) + letters(line[21][:3]))
    for margin, moved in ((3.0, True), (0.0, False)):
        fol = LocalFollower(ix, FollowerConfig(jump_margin=margin, jump_confirm=2))
        assert fol.step(_frames(letters(line[1]))[-100:], 1.0, hmm_word=line[1][-1]) == line[1][-1]
        got = [fol.step(heard[-100:], 1.1 + 0.1 * k, hmm_word=line[1][-1]) for k in range(3)]
        assert (got[-1] in line[21]) == moved, got


def test_jump_alone_waits_for_the_tracker():
    """With jump_alone above the evidence, the follower jumps only once the tracker's word is near."""
    ix, line, letters = _jump_setup()
    heard = _frames(letters(line[1]) + letters(line[21][:3]))
    fol = LocalFollower(ix, FollowerConfig(jump_margin=3.0, jump_alone=1e9, jump_confirm=2))
    fol.step(_frames(letters(line[1]))[-100:], 1.0, hmm_word=line[1][-1])
    assert all(fol.step(heard[-100:], 1.1 + 0.1 * k, hmm_word=line[1][-1]) not in line[21] for k in range(3))
    got = [fol.step(heard[-100:], 1.5 + 0.1 * k, hmm_word=line[21][0]) for k in range(3)]
    assert got[-1] in line[21], got
