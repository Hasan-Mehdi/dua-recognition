"""Load du'a reference texts and expose their segments."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "duas"


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


def _segment(d: dict) -> Segment:
    return Segment(
        id=d.get("segment_id", d.get("id")),
        arabic=d["arabic"],
        transliteration=d.get("transliteration", ""),
        translation=d.get("translation_en", d.get("translation", "")),
    )


def load_dua(path: str | Path) -> Dua:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dua(
        id=raw["dua_id"],
        name_en=raw.get("dua_name_en", raw["dua_id"]),
        name_ar=raw.get("dua_name_ar", ""),
        segments=[_segment(s) for s in raw["segments"]],
    )


def load_all(data_dir: str | Path = DATA_DIR) -> dict[str, Dua]:
    """Load every *.json du'a under data_dir, keyed by dua id."""
    out: dict[str, Dua] = {}
    for path in sorted(Path(data_dir).glob("*.json")):
        dua = load_dua(path)
        out[dua.id] = dua
    return out
