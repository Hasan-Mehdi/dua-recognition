from dua_recognition.corpus import Dua, Segment
from dua_recognition.match import PassageMatcher


def _dua():
    return Dua(
        id="demo",
        name_en="Demo",
        name_ar="",
        segments=[
            Segment(1, "بسم الله الرحمن الرحيم"),
            Segment(2, "الحمد لله رب العالمين"),
            Segment(3, "الرحمن الرحيم"),
        ],
    )


def test_locates_exact_segment():
    hit = PassageMatcher(_dua()).locate("الحمد لله رب العالمين")
    assert hit is not None and hit.segment_id == 2
    assert hit.score >= 0.9


def test_locates_partial_fragment():
    hit = PassageMatcher(_dua()).locate("رب العالمين")
    assert hit is not None and hit.segment_id == 2


def test_returns_none_below_threshold():
    assert PassageMatcher(_dua()).locate("كلام مختلف تماما", min_score=0.8) is None
