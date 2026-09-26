from dua_recognition.translit import for_display, transliterate, vowelled


def test_common_openings():
    assert transliterate("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ") == "Bismillāhi r-raḥmāni r-raḥīm"
    assert transliterate("الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ") == "Al-ḥamdu lillāhi rabbi l-‘ālamīn"
    assert transliterate("اللَّهُمَّ صَلِّ عَلَى مُحَمَّدٍ وَآلِ مُحَمَّد") == "Allāhumma ṣalli ‘alā Muḥammadin wa’āli Muḥammad"


def test_article_sun_letters_and_relatives():
    assert transliterate("يَا نُورَ النُّورِ") == "Yā nūra n-nūr"
    assert transliterate("مَنْ ذَا الَّذِي يَشْفَعُ عِنْدَهُ") == "Man dhā lladhī yashfa‘u ‘indah"
    assert transliterate("لَهُ مَا فِي السَّمَاوَاتِ وَمَا فِي الْأَرْضِ") == "Lahu mā fī s-samāwāti wamā fī l-arḍ"


def test_source_slips():
    # Shadda on the alif, a sukun beside a vowel, the small alif left out.
    assert transliterate("إلاّ أنْتَ") == "Illā ant"
    assert transliterate("كُلَّ شَيٍْء") == "Kulla shay’"
    assert transliterate("اللَّهُ لَا إِلَهَ إِلَّا هُوَ") == "Allāhu lā ilāha illā huwa"


def test_pause_at_line_end():
    assert transliterate("نَبِيِّ الرَّحْمَةِ") == "Nabiyyi r-raḥmah"
    assert transliterate("يَا جَليلُ يَا اللَّهُ") == "Yā jalīlu yā Allāh"


def test_unvowelled_lines_keep_the_source_reading():
    assert not vowelled("من به طه")
    assert for_display("من به طه", "min bihi taha") == "min bihi taha"
    assert for_display("يَا نُورَ النُّورِ", "YAA NOORAN NOOR") == "Yā nūra n-nūr"
