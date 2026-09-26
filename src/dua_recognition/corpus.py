"""Load du'a reference texts, and the labelled recordings used for evaluation."""
from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "duas"
CACHE_DIR = ROOT / "data" / "duaplayer"
# More human-timed recordings in the same format (scripts/fetch_duaspro.py).
# Opt-in (extra=True or DUA_EXTRA_SOURCES=1) so results stay comparable with
# the DuaPlayer-only numbers until they're re-baselined.
EXTRA_CACHE_DIRS = [ROOT / "data" / "duaspro", ROOT / "data" / "duasorg_timed"]


@dataclass(frozen=True)
class Segment:
    id: int
    arabic: str
    transliteration: str = ""
    translation: str = ""


@dataclass
class Dua:
    id: str
    name_en: str
    name_ar: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [s.arabic for s in self.segments]


def _segment(d: dict, extra: dict | None = None) -> Segment:
    extra = extra or {}
    return Segment(
        id=d.get("segment_id", d.get("id")),
        arabic=d["arabic"],
        transliteration=d.get("transliteration") or extra.get("tl", ""),
        translation=d.get("translation_en") or d.get("translation") or extra.get("en", ""),
    )


def load_dua(path: str | Path, cache_dir: str | Path = CACHE_DIR) -> Dua:
    """Load one du'a. Translations are merged in from the local DuaPlayer cache
    when scripts/fetch_duaplayer.py has been run; they aren't committed."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    extra_path = Path(cache_dir) / raw["dua_id"] / "slides.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {}
    return Dua(
        id=raw["dua_id"],
        name_en=raw.get("dua_name_en", raw["dua_id"]),
        name_ar=raw.get("dua_name_ar", ""),
        segments=[_segment(s, extra.get(str(s.get("segment_id")))) for s in raw["segments"]],
    )


def load_all(data_dir: str | Path = DATA_DIR) -> dict[str, Dua]:
    """Load every *.json du'a under data_dir, keyed by dua id."""
    out: dict[str, Dua] = {}
    for path in sorted(Path(data_dir).glob("*.json")):
        dua = load_dua(path)
        out[dua.id] = dua
    return out


@dataclass
class Recording:
    """A recitation with a human-labelled start time for every segment."""

    audio_id: str
    dua_id: str
    reciter: str
    path: Path
    duration_s: float
    starts: list[tuple[float, int]]  # (start seconds, segment id), ascending
    end_s: float  # labels stop here; anything after is unlabelled

    def segment_at(self, t: float) -> int | None:
        """Ground-truth segment being recited at time t (seconds)."""
        if t >= self.end_s:
            return None
        i = bisect.bisect_right([s for s, _ in self.starts], t) - 1
        return self.starts[i][1] if i >= 0 else None


def load_recordings(dua: Dua, cache_dir: str | Path = CACHE_DIR, repaired: bool = False,
                    extra: bool | None = None) -> list[Recording]:
    """Labelled recordings of a du'a.

    repaired: use `<audio>.repaired.json` timings where scripts/audit_labels.py
    wrote them (human boundary times, line numbers re-derived after a text
    re-split). Off by default: those labels are partly machine-made.
    extra: also read EXTRA_CACHE_DIRS (default: the DUA_EXTRA_SOURCES env var).
    """
    if extra is None:
        extra = os.environ.get("DUA_EXTRA_SOURCES", "") not in ("", "0")
    dirs = [Path(cache_dir)] + (EXTRA_CACHE_DIRS if extra else [])
    out = []
    n = len(dua.segments)
    metas = sorted(p for d in dirs for p in (d / dua.id).glob("*-*.json"))
    for meta_path in metas:
        if meta_path.name.endswith(".repaired.json"):
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        audio = meta_path.with_suffix(".mp3")
        if not audio.exists():
            continue
        fixed = meta_path.with_name(meta_path.stem + ".repaired.json")
        if repaired and fixed.exists():
            m["slide_start_ms"] = json.loads(fixed.read_text(encoding="utf-8"))["slide_start_ms"]
        timing = {int(k): v / 1000 for k, v in m["slide_start_ms"].items()}
        starts = sorted((t, k) for k, t in timing.items() if k <= n)
        # A timing key past the last segment marks the end of the recitation.
        after = [t for k, t in timing.items() if k > n]
        end = min(after) if after else m["duration_ms"] / 1000
        out.append(
            Recording(
                audio_id=m["audio_id"],
                dua_id=dua.id,
                reciter=m["reciter"],
                path=audio,
                duration_s=m["duration_ms"] / 1000,
                starts=starts,
                end_s=end,
            )
        )
    return out
