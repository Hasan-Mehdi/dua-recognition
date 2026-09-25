from dua_recognition.align import CorpusIndex
from dua_recognition.corpus import Dua, Segment
from dua_recognition.tracker import Tracker

OPENING = "بسم الله الرحمن الرحيم"
REFRAIN = "يا وجيها عند الله اشفع لنا عند الله"
NAMES = ["يا ابا القاسم يا رسول الله", "يا ابا الحسن يا امير المومنين", "يا فاطمه الزهراء يا بنت محمد"]


def _corpus():
    segs = [Segment(1, OPENING)]
    for name in NAMES:
        segs += [Segment(len(segs) + 1, name), Segment(len(segs) + 2, REFRAIN)]
    return {
        "tawassul": Dua("tawassul", "T", "", segs),
        "a": Dua("a", "A", "", [Segment(1, OPENING), Segment(2, "الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين")]),
        "b": Dua("b", "B", "", [Segment(1, OPENING), Segment(2, "قل هو الله احد الله الصمد لم يلد ولم يولد")]),
    }


def _recite(tracker, dua, window_words=5):
    """Feed one word per hop, each hop 'hearing' the last few words."""
    words = [(s.id, w) for s in dua.segments for w in s.arabic.split()]
    out = []
    for i in range(len(words)):
        heard = " ".join(w for _, w in words[max(0, i - window_words + 1) : i + 1])
        pos = tracker.update(heard, 1.0)
        out.append((words[i][0], pos))
    return out


def test_follows_the_right_repetition_of_a_refrain():
    corpus = _corpus()
    tracker = Tracker(CorpusIndex(corpus))
    trace = _recite(tracker, corpus["tawassul"])
    # Once the du'a is known, every refrain word is shown on *its* repetition,
    # though all three repetitions are textually identical.
    refrain_ids = {s.id for s in corpus["tawassul"].segments if s.arabic == REFRAIN}
    judged = [(truth, pos) for truth, pos in trace if truth in refrain_ids and pos.dua]
    assert judged
    assert all(pos.dua == "tawassul" and pos.segment == truth for truth, pos in judged)


def test_withholds_the_dua_until_the_text_is_distinctive():
    tracker = Tracker(CorpusIndex(_corpus()))
    pos = tracker.update(OPENING, 1.0)  # all three open this way
    assert pos.dua is None
    tracker.update("يا ابا القاسم", 1.0)
    tracker.update("يا ابا القاسم يا رسول الله", 1.0)  # only in tawassul
    assert tracker.position().dua == "tawassul"


def test_silence_does_not_move_the_position():
    corpus = _corpus()
    tracker = Tracker(CorpusIndex(corpus))
    _recite(tracker, Dua("x", "", "", corpus["tawassul"].segments[:3]))
    before = tracker.position()
    for _ in range(10):
        after = tracker.update("", 1.0)
    assert (after.dua, after.segment) == (before.dua, before.segment)


def test_finishing_a_dua_does_not_spill_into_the_next_one():
    corpus = _corpus()
    tracker = Tracker(CorpusIndex(corpus))  # "a" sits right before "b" in the index
    _recite(tracker, corpus["a"])
    for _ in range(20):  # the reciter has finished; windows hear the last words
        pos = tracker.update("مالك يوم الدين", 1.0)
    assert pos.dua == "a" and pos.segment == 2
    assert tracker.post[tracker.ix.dua_word_span[2][0]] < 1e-3  # nothing leaked into "b"
