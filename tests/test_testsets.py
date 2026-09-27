"""The held-out test sets (corpus.TESTSETS): loading, tiers, review round-trip, harvest exclusion."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dua_recognition import corpus, splits  # noqa: E402
from dua_recognition.corpus import Dua, Segment  # noqa: E402
from dua_recognition.labels import agreement, from_srt, meta_starts, set_starts, to_srt  # noqa: E402

DUA = Dua("dua-test", "Test", "", [Segment(i, f"line {i}") for i in range(1, 6)])


def _write(root: Path, audio_id: str, **meta) -> None:
    d = root / DUA.id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{audio_id}.mp3").write_bytes(b"")
    m = {"audio_id": audio_id, "dua_id": DUA.id, "reciter": f"majlis:{audio_id}", "duration_ms": 60000,
         "slide_start_ms": {"1": 0, "2": 10000, "3": 25000, "6": 50000}}
    m.update(meta)
    (d / f"{audio_id}.json").write_text(json.dumps(m), encoding="utf-8")


@pytest.fixture
def majlis(tmp_path, monkeypatch):
    _write(tmp_path, "mj-a", auto=False, condition="majlis", venue="KSIJ")
    _write(tmp_path, "mj-b", auto=True, condition="majlis", venue="MOMIN")
    _write(tmp_path, "mj-c", auto=True, needs_review=True, condition="majlis", venue="MOMIN")
    monkeypatch.setitem(corpus.TESTSETS, "majlis", tmp_path)
    return tmp_path


def test_loader_reads_a_testset_dir(majlis):
    recs = {r.audio_id: r for r in corpus.load_recordings(DUA, cache_dir=majlis, extra=False)}
    assert set(recs) == {"mj-a", "mj-b", "mj-c"}
    a = recs["mj-a"]
    assert a.starts == [(0.0, 1), (10.0, 2), (25.0, 3)] and a.end_s == 50.0
    assert (a.condition, a.venue, a.tier) == ("majlis", "KSIJ", "gold")
    assert recs["mj-b"].tier == "silver" and recs["mj-c"].tier == "review"


def test_duaplayer_meta_defaults_to_gold_studio(tmp_path):
    _write(tmp_path, "dp-x")
    (r,) = corpus.load_recordings(DUA, cache_dir=tmp_path, extra=False)
    assert (r.auto, r.condition, r.tier) == (False, "studio", "gold")


def test_testset_source_is_all_test_and_tiers_filter(majlis):
    import evaluate as ev

    try:
        ev.use_source("majlis", "all")
        assert ev.is_test("anyone at all")
        assert {r.audio_id for r in ev.load_recordings(DUA)} == {"mj-a", "mj-b"}  # review queue left out
        ev.use_source("majlis", "gold")
        assert [r.audio_id for r in ev.load_recordings(DUA)] == ["mj-a"]
        ev.use_source("majlis", "silver")
        assert [r.audio_id for r in ev.load_recordings(DUA)] == ["mj-b"]
        assert "2 venues" in ev.describe(ev.load_recordings(DUA) + [r for r in corpus.load_recordings(
            DUA, cache_dir=majlis, extra=False) if r.audio_id == "mj-a"])
    finally:
        ev.use_source("duaplayer")
    assert not ev.is_test("Ali Fani")  # the DuaPlayer split is unchanged


def test_srt_json_srt_is_identity():
    meta = {"duration_ms": 60000, "slide_start_ms": {"1": 0, "2": 10250, "3": 25003, "6": 50000}}
    starts, end = meta_starts(meta, 5)
    text = {s.id: s.arabic for s in DUA.segments}
    srt = to_srt(starts, end, text)
    back, end2 = from_srt(srt)
    assert back == starts and end2 == end
    meta2 = set_starts(meta, back, end2, 5)
    assert meta2["slide_start_ms"] == meta["slide_start_ms"]
    assert to_srt(*meta_starts(meta2, 5), text) == srt


def test_srt_reader_takes_line_ids_from_the_text_not_cue_numbers():
    srt = ("7\r\n00:00:01,500 --> 00:00:04,000\r\n2. line 2\r\n\r\n"
           "3\n00:00:00,000 --> 00:00:01,500\n1. line 1\nsecond row\n")
    assert from_srt(srt) == ([(0.0, 1), (1.5, 2)], 4.0)


def test_agreement():
    a = [(0.0, 1), (10.0, 2)]
    assert agreement(a, a, 20.0) == 1.0
    assert agreement(a, [(0.0, 1), (15.0, 2)], 20.0) == pytest.approx(0.75)


def test_test_channel_video_is_skipped_by_harvesters():
    import fetch_youtube
    import find_captioned

    venue = next(iter(splits.TEST_CHANNELS))
    v = {"id": "abcdefghijk", "title": "Dua Kumayl live", "channel_id": venue}
    assert splits.is_test_upload(v)
    assert fetch_youtube.excluded(v)
    assert find_captioned.is_test_upload(v)
    assert not fetch_youtube.excluded({"id": "abcdefghijk", "title": "Dua Kumayl", "channel_id": "UCsomeoneelse"})
    assert splits.is_test_upload({"id": "x", "channel_url": f"https://www.youtube.com/channel/{venue}"})
    # an old download's meta.json has the channel name but no channel_id
    assert fetch_youtube.excluded({"id": "x", "title": "Dua Kumayl", "channel": splits.TEST_CHANNELS[venue]})
