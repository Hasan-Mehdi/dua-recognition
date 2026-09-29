"""The token-CTC adapter (ctc_adapter.py): sub-word pieces, their own frame hop, word ownership."""
import numpy as np

from dua_recognition.align import CorpusIndex
from dua_recognition.corpus import Dua, Segment
from dua_recognition.ctc_adapter import LOG_FLOOR, TokenFollower, corpus_surface_words, expand, surface
from dua_recognition.follower import FollowerConfig

WORDS = ["بسم", "الله", "الرحمن", "الرحيم", "الحمد", "لله", "رب", "العالمين"]
DUA = Dua("d-1", "T", "", [Segment(1, " ".join(WORDS[:4])), Segment(2, " ".join(WORDS[4:]))])
BLANK = 9  # native blank id; adapter column = token + 2 below it


def test_surface_keeps_spelling_the_model_writes():
    assert surface("ٱلْحَمْدُ لِلَّهِ") == "الحمد لله"
    assert surface("رَحْمَةً") == "رحمة"  # ta marbuta kept (text.normalize folds it)
    assert surface("أَنْتَ") == "أنت"


def test_surface_words_line_up_with_the_index():
    ix = CorpusIndex({DUA.id: DUA})
    assert corpus_surface_words(ix) == WORDS


def _frames(seq, frames_per=3, blank_after=0):
    """Adapter-column log posteriors: each token id holds `frames_per` frames, then blanks."""
    width = BLANK + 1
    rows = []
    for t in seq:
        for _ in range(frames_per):
            r = np.full(width, -12.0)
            r[t + 2] = -0.01
            r[1] = -0.01  # best non-blank
            rows.append(r)
        r = np.full(width, -12.0)
        r[0] = -0.01
        rows.append(r)
    for _ in range(blank_after):
        r = np.full(width, -12.0)
        r[0] = -0.01
        rows.append(r)
    return np.array(rows, dtype=np.float32)


def test_token_follower_uses_multi_letter_pieces_and_its_own_hop():
    ix = CorpusIndex({DUA.id: DUA})
    # 8 words, 2 pieces each (ids 0-7 reused): word w = [w % 8, (w + 1) % 8]
    toks = [[w % 8, (w + 1) % 8] for w in range(len(WORDS))]
    meta = {"frame_s": 0.08, "blank": BLANK}
    f = TokenFollower(ix, meta, FollowerConfig(window_s=3.0), word_tokens=toks)
    assert f.frame_s == 0.08
    ref, owner = f._reference(0)
    assert list(owner[:4]) == [0, 0, 1, 1] and list(ref[:2]) == [2, 3]
    # audio says words 0..2: the follower should reach word 2 from an anchor on word 0
    lp = _frames([t for w in range(3) for t in toks[w]])
    assert f.step(lp, 1.0, hmm_word=0) in (1, 2)
    assert f.step(lp, 1.2, hmm_word=0) == 2
    # a long blank tail after word 2 (a pause): it holds
    lp2 = _frames([t for w in range(3) for t in toks[w]], blank_after=12)
    assert f.step(lp2, 2.0, hmm_word=0) == 2


def test_expand_restores_restricted_columns():
    stored = np.array([[[-0.1, -1.0, -2.0]]], dtype=np.float16)
    full = expand(stored, np.array([0, 1, 5]), 7)
    assert full.shape == (1, 1, 7)
    assert full[0, 0, 5] == np.float32(np.float16(-2.0)) and full[0, 0, 3] == LOG_FLOOR
