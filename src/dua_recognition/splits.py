"""Reciter-disjoint train/test split.

Every number reported in docs/RESULTS.md comes from TEST reciters: nobody in
that set was used to tune the tracker or to fine-tune an ASR model. The split
is by reciter, not by recording, because a model that has heard someone recite
Kumayl has learned their voice and pacing, not just that recording.

Some du'as were only ever recorded by train reciters, so test covers fewer
du'as than train (docs/results lists which).

The extra test sets (data/testsets: majlis streams, ordinary voices) are held
out by venue and uploader instead: TEST_CHANNELS are YouTube channels that no
training harvester may download from.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

TEST_RECITERS = frozenset(
    {
        "Abu Thar Al-Halawaji",
        "Hussein Ghareeb",
        "Hussain Al-Akraf",
        "Murtada al-Qureish",
        "Murtaza Quraish",  # the same reciter, spelled differently on another upload
        "Mohsen Farahmand Azad",
        # duas.org recordings whose voice matched no named reciter confidently
        # (scripts/speaker_check.py): uncertain voices go to test, never train.
        "duas.org: Naqvi",
        "duas.org: Jamia Kabira reciter",
        "duas.org: Ya Mafzai reciter",
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


# DuaPlayer reciters as they appear in YouTube titles and channel names (Arabic
# and English). Harvests skip them: test reciters so held-out numbers can't
# leak, train reciters so extra hours bring new voices; the ordinary-voice
# search skips them because they're professionals.
KNOWN_RECITERS = [
    # test
    "حلواجي", "الحلواجي", "halwachi", "halawaji", "abather", "abu thar", "أباذر", "اباذر",
    "غريب", "ghareeb", "ghreeb", "gharib",
    "الأكرف", "الاكرف", "akraf",
    "قريش", "qureish", "quraish", "qurayshi",
    "فرهمند", "farahmand",
    # train
    "فاني", "fani", "قمبر", "qambar", "kambar", "العطار", "attar", "بوماد", "boumad",
    "رسولي", "rasouli", "رضوي", "rizvi",
]

# YouTube channels of the majlis test venues (scripts/fetch_testset.py). Every
# training harvester (fetch_youtube.py, find_captioned.py, fetch_fatemah.py)
# skips them, so a test venue's room, PA and voices never reach training.
TEST_CHANNELS = {
    "UC55K9YR08uUk-RW5-6MXKzw": "KSIJ Dar es Salaam",
    "UCMnrAfpbEGgb8RSFxfzL_gg": "Islamic Center of MOMIN",
    "UC3J1fl-utIACafil8MJG7Xw": "SABA Islamic Center",
    "UC9aYBYroj9pUzDtz7R-R_RA": "Imam Al-Khoei Foundation, NY",
    "UCSfQfp-yD6lZNZiW_B3lzTQ": "Hyderi Islamic Centre",
}
# Private uploaders (the ordinary-voice set) aren't named in the repo: their
# channel and video ids live in the gitignored data/testsets/<set>/holdout.json
# ({"channels": [...], "videos": [...]}), written by scripts/fetch_testset.py.
TESTSETS_DIR = Path(__file__).resolve().parents[2] / "data" / "testsets"


@functools.cache  # read once per run: harvesters call is_test_upload per video
def _holdout() -> tuple[frozenset[str], frozenset[str]]:
    channels, videos = set(TEST_CHANNELS), set()
    for f in sorted(TESTSETS_DIR.glob("*/holdout.json")):
        h = json.loads(f.read_text(encoding="utf-8"))
        channels |= set(h.get("channels", []))
        videos |= set(h.get("videos", []))
    return frozenset(channels), frozenset(videos)


def is_test_upload(video: dict) -> bool:
    """True if a YouTube entry (yt-dlp info: id, channel_id, uploader_id,
    channel_url) belongs to a test venue or a test uploader."""
    channels, videos = _holdout()
    if video.get("id") in videos:
        return True
    # By name too: metadata saved before downloads recorded channel_id has only it.
    if video.get("channel") in TEST_CHANNELS.values() or video.get("uploader") in TEST_CHANNELS.values():
        return True
    keys = {video.get("channel_id"), video.get("uploader_id")}
    url = video.get("channel_url") or video.get("uploader_url") or ""
    keys |= {url.rstrip("/").rsplit("/", 1)[-1]} if url else set()
    return bool(keys & channels)
