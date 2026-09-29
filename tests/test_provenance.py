"""Label provenance (dua_recognition/provenance.py), the dev/final split, and eval-set cache keys."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dua_recognition import corpus, provenance  # noqa: E402
from dua_recognition.corpus import Dua, Segment  # noqa: E402

WORDS = ["بسم", "الله", "الرحمن", "الرحيم", "الحمد", "لله", "رب", "العالمين", "مالك", "يوم"]
DUA = Dua("dua-test", "Test", "", [Segment(i, " ".join(WORDS[2 * i - 2 : 2 * i])) for i in range(1, 6)])


def _write(root: Path, audio_id: str, **meta) -> Path:
    d = root / DUA.id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{audio_id}.mp3").write_bytes(b"")
    m = {"audio_id": audio_id, "dua_id": DUA.id, "reciter": f"x:{audio_id}", "duration_ms": 60000,
         "slide_start_ms": {"1": 0, "2": 10000, "3": 25000}}
    m.update(meta)
    (d / f"{audio_id}.json").write_text(json.dumps(m), encoding="utf-8")
    return d / f"{audio_id}.json"


def test_legacy_line_provenance_is_inferred_never_assumed_human(tmp_path):
    _write(tmp_path, "r-a", auto=True, teacher="whisper-turbo-dua")
    _write(tmp_path, "r-b", auto=False, reviewed="2026-09-26", teacher="whisper-turbo-dua")
    _write(tmp_path, "r-c", auto=False)  # gold tier, but nobody recorded who timed it
    recs = {r.audio_id: r for r in corpus.load_recordings(DUA, cache_dir=tmp_path, extra=False)}
    assert recs["r-a"].line_provenance["kind"] == "auto" and recs["r-a"].tier == "silver"
    assert recs["r-b"].line_provenance["kind"] == "human_reviewed" and recs["r-b"].tier == "gold"
    # The legacy tier interface is unchanged; provenance says "unknown", not human.
    assert recs["r-c"].tier == "gold" and recs["r-c"].line_provenance["kind"] == "unknown"
    assert not provenance.is_human_line(recs["r-c"].line_provenance)
    dp = tmp_path / "duaplayer"
    _write(dp, "r-d")
    (r,) = corpus.load_recordings(DUA, cache_dir=dp, extra=False)
    assert r.line_provenance["kind"] == "human_timed"


def test_line_review_does_not_promote_words(tmp_path, monkeypatch):
    import evaluate as ev

    monkeypatch.setattr(ev, "WORD_TRUTH", tmp_path / "wt")
    monkeypatch.setattr(provenance, "WORD_REVIEW", tmp_path / "wr")
    (tmp_path / "wt").mkdir()
    _write(tmp_path / "set", "u-1", auto=False, reviewed="2026-09-26",
           provenance={"line": {"kind": "human_reviewed"}, "word": {"kind": "auto"}})
    (rec,) = corpus.load_recordings(DUA, cache_dir=tmp_path / "set", extra=False)
    ix = ev.CorpusIndex({DUA.id: DUA})
    (tmp_path / "wt" / "u-1.json").write_text(json.dumps({"dua": DUA.id, "words": [[0, 0.1, 0.4, -0.2], [1, 10.2, 10.5, -0.3]]}))
    assert rec.line_provenance["kind"] == "human_reviewed"
    # default (legacy) word truth unchanged; the human tier is empty without a word review
    assert ev.load_word_truth(rec, ix) == [[0, 0.1, 0.4, -0.2], [1, 10.2, 10.5, -0.3]]
    assert ev.load_word_truth(rec, ix, tier="human") is None
    assert ev.word_truth_provenance(rec)["kind"] == "auto"


def test_partial_word_review_masks_unreviewed_words():
    review = {"intervals": [[0.0, 1.0]],
              "words": [{"w": 0, "start": 0.2, "status": "ok"},
                        {"w": 1, "start": 5.0, "status": "ok"},  # outside the reviewed interval
                        {"w": 2, "status": "omitted"},
                        {"w": 3, "start": 0.6, "start_lo": 0.5, "start_hi": 0.8, "status": "repeated", "occurrence": 1}]}
    hw = provenance.human_words(review, lo=100)
    assert [(r[0], r[6], r[7]) for r in hw] == [(100, "ok", 0), (103, "repeated", 1)]
    assert hw[1][4:6] == [0.5, 0.8]
    inside, outside = provenance.mask_auto_words([[0, 0.2, 0.4, 0], [1, 5.0, 5.3, 0]], review)
    assert [w[0] for w in inside] == [0] and [w[0] for w in outside] == [1]
    assert provenance.mask_auto_words([[0, 0.2, 0.4, 0]], None) == ([], [[0, 0.2, 0.4, 0]])


def test_split_groups_never_cross_and_stay_frozen(tmp_path):
    import testset_split as ts

    _write(tmp_path, "m-1", venue="KSIJ", channel_id="UCaaa")
    _write(tmp_path, "m-2", venue="KSIJ (reupload)", channel_id="UCaaa")  # same channel, other title
    _write(tmp_path, "m-3", venue="Other", channel_id="UCbbb")
    recs = corpus.load_recordings(DUA, cache_dir=tmp_path, extra=False)
    groups = {r.audio_id: ts.group_of("majlis", r) for r in recs}
    assert groups["m-1"] == groups["m-2"] != groups["m-3"]
    split = {}
    ts.assign(split, "majlis", sorted(set(groups.values())))
    first = {g: v["part"] for g, v in split["majlis"].items()}
    parts = {ts.part_of("majlis", r, split) for r in recs if groups[r.audio_id] == groups["m-1"]}
    assert len(parts) == 1
    ts.assign(split, "majlis", sorted(set(groups.values())) + ["venue:UCccc", "venue:UCddd"])
    assert all(split["majlis"][g]["part"] == p for g, p in first.items())  # never moved
    assert ts.part_of("majlis", corpus.Recording("z", DUA.id, "r", tmp_path / "none.mp3", 1, [], 1, venue="new"),
                      split) is None
    split = {}
    ts.assign(split, "user", ["person:user:hasan"])
    assert split["user"]["person:user:hasan"]["part"] == "dev"


def test_eval_set_cache_key_catches_config_and_input_changes(tmp_path, monkeypatch):
    import follow_eval as fe

    from dua_recognition.tracker import TrackerConfig

    for name in ("WINDOWS", "WORD_TRUTH"):
        monkeypatch.setattr(fe.ev, name, tmp_path / name.lower())
    monkeypatch.setattr(fe, "CTC", tmp_path / "ctc")
    monkeypatch.setattr(fe, "ROOT", tmp_path)
    (tmp_path / "word_truth").mkdir()
    ix = fe.ev.CorpusIndex({DUA.id: DUA})
    args = ("asr", "train", "ctc", "ctc", False)
    k0 = fe.set_key(ix, *args, TrackerConfig(), 0.5)
    assert fe.set_key(ix, *args, TrackerConfig(), 0.5) == k0
    assert fe.set_key(ix, *args, TrackerConfig(), 0.3) != k0
    fields = TrackerConfig.__dataclass_fields__
    name = next(n for n, f in fields.items() if isinstance(f.default, float))
    from dataclasses import replace

    assert fe.set_key(ix, *args, replace(TrackerConfig(), **{name: getattr(TrackerConfig(), name) + 1.0}), 0.5) != k0
    f = tmp_path / "word_truth" / "r.json"
    f.write_text("{}")
    k1 = fe.set_key(ix, *args, TrackerConfig(), 0.5)
    assert k1 != k0
    f.write_text('{"words": []}')
    os.utime(f, ns=(1, 1))
    assert fe.set_key(ix, *args, TrackerConfig(), 0.5) != k1
    other = Dua("dua-test", "Test", "", [Segment(i, " ".join(WORDS[2 * i - 2 : 2 * i]) + " نور") for i in range(1, 6)])
    assert fe.set_key(fe.ev.CorpusIndex({other.id: other}), *args, TrackerConfig(), 0.5) != \
        fe.set_key(ix, *args, TrackerConfig(), 0.5)


def test_word_provenance_legacy_and_fresh():
    assert provenance.word_provenance({"words": []})["kind"] == "auto"
    p = provenance.auto_word_provenance("m", "h", provenance.reference_hash(["a"]), {"kind": "human_reviewed"})
    assert p["kind"] == "auto" and p["line_labels"] == "human_reviewed" and p["review"] == "none"


@pytest.mark.parametrize("status", provenance.WORD_STATUS)
def test_every_status_is_handled(status):
    review = {"intervals": [[0, 1]], "words": [{"w": 0, "start": 0.5, "status": status}]}
    assert len(provenance.human_words(review, 0)) == (1 if status in provenance.SCORABLE else 0)
