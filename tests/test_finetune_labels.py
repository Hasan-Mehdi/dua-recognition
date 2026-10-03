"""Training labels are the spelling Whisper writes (scripts/build_finetune_set.whisper_style)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_finetune_set import whisper_style  # noqa: E402


def test_labels_drop_marks_but_keep_the_letters_whisper_writes():
    assert whisper_style("ٱلرَّحْمَـٰنِ") == "الرحمن"
    assert whisper_style("رَحْمَةِ") == "رحمة"  # ta marbuta stays: Whisper writes it
    assert whisper_style("أَسْأَلُكَ") == "أسألك"


def test_persian_and_urdu_letter_forms_become_arabic_letters():
    # duas.org prints some texts with them; they used to fall out of the label (یا -> ا).
    assert whisper_style("یَا") == "يا"
    assert whisper_style("اَللّٰہُمَّ") == "اللهم"
    assert whisper_style("کُلِّ") == "كل"
