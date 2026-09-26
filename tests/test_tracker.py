from dua_recognition.align import CorpusIndex
from dua_recognition.corpus import Dua, Segment
from dua_recognition.tracker import Tracker, TrackerConfig

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


def _stop_after_first_name(config):
    """Recite the opening and the first name, then stop: the windows keep hearing
    the last words (as a 6 s window does) while the reciter is silent."""
    corpus = _corpus()
    tracker = Tracker(CorpusIndex(corpus), config)
    words = [w for s in corpus["tawassul"].segments[:2] for w in s.arabic.split()]
    for i in range(len(words)):
        tracker.update(" ".join(words[max(0, i - 4) : i + 1]), 1.0, lead=1.0)
    tail = " ".join(words[-5:])
    return [tracker.update(tail, 1.0, lead=1.0, quiet=q) for q in (0.2, 1.2, 2.2, 3.0, 3.0, 3.0)]


def test_waits_at_the_end_of_a_line_when_the_reciter_stops():
    from dua_recognition.tracker import TrackerConfig

    shown = _stop_after_first_name(TrackerConfig())
    # Whatever the lead did at the moment they stopped, once the silence is
    # clear the display is back on the line they stopped on, at its end.
    assert all(p.dua == "tawassul" and p.segment == 2 and p.at_line_end for p in shown[2:])


def test_reciter_mode_keeps_running_on_through_a_breath():
    from dua_recognition.tracker import RECITER, TrackerConfig

    shown = _stop_after_first_name(TrackerConfig(**RECITER))
    assert all(p.segment == shown[0].segment for p in shown)  # no stepping back


KURSI = ["الله لا اله الا هو الحي القيوم", "لا تاخذه سنه ولا نوم", "له ما في السماوات وما في الارض",
         "من ذا الذي يشفع عنده الا باذنه"]


def _shared_corpus():
    # "kursi" is wholly inside "sahifa" (as Ayat al-Kursi is inside Sahifa 54), and a
    # third text opens the same way as neither.
    return {
        "kursi": Dua("kursi", "K", "", [Segment(i + 1, t) for i, t in enumerate(KURSI)]),
        "sahifa": Dua("sahifa", "S", "", [Segment(1, "اعصمني وطهرني واذهب ببليتي")]
                      + [Segment(i + 2, t) for i, t in enumerate(KURSI)]
                      + [Segment(len(KURSI) + 2, "قل هو الله احد الله الصمد")]),
        "other": Dua("other", "O", "", [Segment(1, OPENING), Segment(2, "الحمد لله رب العالمين")]),
    }


def test_a_shared_passage_is_shown_rather_than_withheld():
    corpus = _shared_corpus()
    off = Tracker(CorpusIndex(corpus), TrackerConfig(same_text_words=0))
    on = Tracker(CorpusIndex(corpus))
    # Someone reciting Sahifa, from its first line into the shared passage.
    shown_off, shown_on = [], []
    for _, pos in _recite(off, corpus["sahifa"]):
        shown_off.append(pos.dua)
    for _, pos in _recite(on, corpus["sahifa"]):
        shown_on.append(pos.dua)
    # With the rule, the shared passage never makes the du'a vanish once it was shown,
    # and it never flips to the other text that shares it.
    first = shown_on.index("sahifa")
    assert all(d == "sahifa" for d in shown_on[first:])
    assert shown_on.count(None) <= shown_off.count(None)


def test_the_passage_on_its_own_is_shown_as_itself():
    # Reciting the short text: every text it's in reads the same, so without the
    # rule nothing is ever shown; with it, the text the passage opens is shown.
    corpus = _shared_corpus()
    off = Tracker(CorpusIndex(corpus), TrackerConfig(same_text_words=0))
    assert all(pos.dua is None for _, pos in _recite(off, corpus["kursi"]))
    trace = _recite(Tracker(CorpusIndex(corpus)), corpus["kursi"])
    shown = [pos for _, pos in trace if pos.dua]
    assert len(shown) >= len(trace) - 2
    assert all(pos.dua == "kursi" and pos.same_as == ["sahifa"] for pos in shown)
