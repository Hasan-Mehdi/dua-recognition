from dua_recognition.corpus import Recording, load_all


def test_segment_at_follows_labels_and_stops_at_end():
    rec = Recording("x", "d", "r", path=None, duration_s=30, starts=[(0.0, 1), (5.0, 2), (12.5, 3)], end_s=20)
    assert rec.segment_at(0.0) == 1
    assert rec.segment_at(4.99) == 1
    assert rec.segment_at(5.0) == 2
    assert rec.segment_at(19.9) == 3
    assert rec.segment_at(20.0) is None


def test_committed_corpus_is_contiguous():
    duas = load_all()
    assert len(duas) >= 12
    for dua in duas.values():
        ids = [s.id for s in dua.segments]
        assert ids == list(range(1, len(ids) + 1)), dua.id
