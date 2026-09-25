"""Reciter-disjoint train/test split.

Every number reported in docs/RESULTS.md comes from TEST reciters: nobody in
that set was used to tune the tracker or to fine-tune an ASR model. The split
is by reciter, not by recording, because a model that has heard someone recite
Kumayl has learned their voice and pacing, not just that recording.

Train covers all 12 du'as; test covers 9 of them (the other three were only
ever recorded by train reciters).
"""
TEST_RECITERS = frozenset(
    {
        "Abu Thar Al-Halawaji",
        "Hussein Ghareeb",
        "Hussain Al-Akraf",
        "Murtada al-Qureish",
        "Murtaza Quraish",  # the same reciter, spelled differently on another upload
        "Mohsen Farahmand Azad",
    }
)


def is_test(reciter: str) -> bool:
    return reciter in TEST_RECITERS


# Line timings that don't match the committed text (scripts/audit_labels.py):
# both Ziyarat Ashura recordings sit a steady 4 lines off from mid-way on,
# because DuaPlayer's timings were recorded against an older 108-slide split of
# a text that now has 103 lines. These recordings still count for
# identification (the du'a label is right), but not for line accuracy.
LINE_LABELS_UNRELIABLE = frozenset({"ziyarat-ashura"})
