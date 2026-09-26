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
    fol = LocalFollower(ix, FollowerConfig(window_s=2.0))
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
