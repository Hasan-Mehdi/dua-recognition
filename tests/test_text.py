from dua_recognition.text import normalize, strip_diacritics


def test_strips_diacritics():
    assert strip_diacritics("بِسْمِ") == "بسم"


def test_normalizes_alef_variants():
    assert normalize("أحمد") == "احمد"
    assert normalize("إله") == "اله"


def test_collapses_whitespace_and_drops_punctuation():
    assert normalize("بسم،  الله") == "بسم الله"


def test_alef_maksura_and_ta_marbuta():
    assert normalize("صلى") == "صلي"
    assert normalize("فاطمة") == "فاطمه"
