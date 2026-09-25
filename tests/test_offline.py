from dua_recognition.align import CorpusIndex
from dua_recognition.offline import align_recording

from test_tracker import REFRAIN, _corpus


def test_smoother_labels_every_repetition_of_a_refrain():
    corpus = _corpus()
    dua = corpus["tawassul"]
    ix = CorpusIndex({"tawassul": dua})
    words = [(s.id, w) for s in dua.segments for w in s.arabic.split()]
    heard = [" ".join(w for _, w in words[max(0, i - 4) : i + 1]) for i in range(len(words))]
    costs = [ix.word_costs(h) for h in heard]
    segs = align_recording(ix, costs, 1.0)
    truth = [s for s, _ in words]
    assert sum(int(a == b) for a, b in zip(segs, truth)) / len(truth) >= 0.9
    refrain_ids = {s.id for s in dua.segments if s.arabic == REFRAIN}
    # Every refrain word lands on its own repetition, never a sibling one.
    assert all(abs(int(a) - b) <= 1 for a, b in zip(segs, truth) if b in refrain_ids)
