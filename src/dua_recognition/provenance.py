"""Where a label came from, per granularity: line starts and word timings apart.

A recording's `auto` flag / tier says whether its *line* starts were reviewed.
It says nothing about its *word* timings: those come from the wav2vec2 forced
aligner (scripts/word_truth.py) whatever the line labels are, and that aligner
shares its acoustic model with the word follower. So the two are kept apart:

    line provenance   meta["provenance"]["line"], else inferred from the legacy
                      fields (auto / reviewed / teacher / the source directory)
    word provenance   word_truth/<id>.json "provenance", else "auto, unknown
                      revision" (files written before this module)
    word review       data/cache/word_review/<id>.json: what a human checked,
                      inside which intervals, and what was actually spoken

Missing provenance means unknown, never human. Importing a reviewed line SRT
promotes line truth only; the re-aligned words stay automatic until a word
review covers them, and only reviewed intervals enter a human-tier word score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORD_REVIEW = ROOT / "data" / "cache" / "word_review"

LINE_KINDS = ("human_timed", "human_reviewed", "human_carried", "auto", "unknown")
# Statuses a reviewer can give a word (the canonical position being followed is
# kept apart from what was actually said: `spoken` holds the verbatim words).
WORD_STATUS = ("ok", "uncertain", "substituted", "omitted", "repeated", "inserted")
SCORABLE = ("ok", "uncertain", "substituted", "repeated")  # has a heard onset for a canonical word

# Human line timings in the DuaPlayer shape, by source directory name.
_HUMAN_DIRS = {"duaplayer": "DuaPlayer annotators", "duaspro": "duas.pro (DuaPlayer mirror)"}
_CARRIED_DIRS = {"duasorg_timed": "duas.org human timings carried onto our line split by word alignment"}


def line_provenance(meta: dict, source_dir: str | Path | None = None) -> dict:
    """The line labels' provenance: explicit if recorded, else inferred from legacy fields."""
    explicit = (meta.get("provenance") or {}).get("line")
    if explicit:
        return dict(explicit)
    if meta.get("auto"):
        return {"kind": "auto", "model": meta.get("teacher") or "unknown", "review": "none",
                "second_teacher": meta.get("second_teacher") or None,
                "teacher_agreement": meta.get("teacher_agreement")}
    if meta.get("reviewed"):
        return {"kind": "human_reviewed", "reviewed": meta["reviewed"], "from_auto": meta.get("teacher")}
    name = Path(source_dir).name if source_dir else ""
    if name in _HUMAN_DIRS:
        return {"kind": "human_timed", "source": _HUMAN_DIRS[name]}
    if name in _CARRIED_DIRS:
        return {"kind": "human_carried", "source": _CARRIED_DIRS[name]}
    return {"kind": "unknown"}


def is_human_line(prov: dict) -> bool:
    return prov.get("kind") in ("human_timed", "human_reviewed")


def reference_hash(texts: list[str]) -> str:
    """Identifies the reference text a label set was aligned against."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def word_provenance(raw: dict) -> dict:
    """A word-truth file's provenance; files written before provenance existed are
    automatic labels of unknown revision (the wav2vec2 aligner, presumably)."""
    p = raw.get("provenance")
    if p:
        return dict(p)
    return {"kind": "auto", "model": "unknown (presumed models/wav2vec2-quran-dua; file predates provenance)",
            "reference_hash": None, "line_labels": "unknown", "review": "none"}


def auto_word_provenance(model: str, model_hash: str | None, ref_hash: str, line_prov: dict) -> dict:
    """What scripts/word_truth.py and user_label.py write next to fresh forced alignments."""
    return {"kind": "auto", "model": model, "model_hash": model_hash, "reference_hash": ref_hash,
            "line_labels": line_prov.get("kind", "unknown"), "review": "none"}


# -- word review overlays -------------------------------------------------------
def load_review(audio_id: str, review_dir: Path = WORD_REVIEW) -> dict | None:
    f = Path(review_dir) / f"{audio_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def inside(t: float, intervals: list[list[float]]) -> bool:
    return any(a <= t < b for a, b in intervals)


def human_words(review: dict, lo: int) -> list[list]:
    """Reviewed words as [word index (offset by `lo`), start, end, 0.0, start_lo, start_hi,
    status, occurrence]: only words with a heard onset inside a reviewed interval."""
    out = []
    iv = review.get("intervals", [])
    for w in review.get("words", []):
        if w.get("status") not in SCORABLE or w.get("start") is None or not inside(w["start"], iv):
            continue
        s = float(w["start"])
        out.append([lo + int(w["w"]), s, float(w.get("end", s)), 0.0, float(w.get("start_lo", s)),
                    float(w.get("start_hi", s)), w["status"], int(w.get("occurrence", 0))])
    out.sort(key=lambda r: r[1])
    return out


def mask_auto_words(words: list[list], review: dict | None) -> tuple[list[list], list[list]]:
    """Split automatic word truth into (inside a reviewed interval, outside one)."""
    if not review:
        return [], list(words)
    iv = review.get("intervals", [])
    a = [w for w in words if inside(w[1], iv)]
    return a, [w for w in words if not inside(w[1], iv)]
