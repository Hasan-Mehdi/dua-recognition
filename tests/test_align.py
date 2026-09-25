import random

import numpy as np

from dua_recognition.align import CorpusIndex, semiglobal_end_costs
from dua_recognition.corpus import Dua, Segment


def _brute(h, r):
    m, n = len(h), len(r)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        d[i][0] = i
        for e in range(1, n + 1):
            d[i][e] = min(d[i - 1][e - 1] + (h[i - 1] != r[e - 1]), d[i - 1][e] + 1, d[i][e - 1] + 1)
    return d[m][1:]


def test_vectorized_dp_matches_textbook_dp():
    rng = random.Random(0)
    for _ in range(300):
        h = [rng.randint(1, 4) for _ in range(rng.randint(1, 8))]
        r = [rng.randint(1, 4) for _ in range(rng.randint(1, 15))]
        got = semiglobal_end_costs(np.array(h, dtype=np.int16), np.array(r, dtype=np.int16))
        assert list(got) == _brute(h, r)


def _index():
    return CorpusIndex(
        {
            "a": Dua("a", "A", "", [Segment(1, "بسم الله الرحمن الرحيم"), Segment(2, "الحمد لله رب العالمين")]),
            "b": Dua("b", "B", "", [Segment(1, "اللهم صل علي محمد وال محمد")]),
        }
    )


def test_fragment_scores_zero_where_it_ends():
    ix = _index()
    costs = ix.word_costs("رب العالمين")
    best = int(np.argmin(costs))
    assert costs[best] == 0
    assert ix.words[best].text == "العالمين" and ix.word_segment[best] == 2


def test_word_end_scoring_does_not_tie_with_next_word():
    ix = _index()
    costs = ix.word_costs("بسم الله")
    allah = next(i for i, w in enumerate(ix.words) if w.text == "الله")
    assert costs[allah] == 0 and costs[allah + 1] > 0


def test_asr_spelling_noise_still_lands():
    ix = _index()
    costs = ix.word_costs("اللهم صلي على محمد و آل محمد")  # ya/alef-maksura, split waw
    best = int(np.argmin(costs))
    assert ix.dua_ids[ix.word_dua[best]] == "b" and costs[best] <= 2
