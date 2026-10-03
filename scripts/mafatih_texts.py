#!/usr/bin/env python
"""Training-only texts from Mafatih al-Jinan, for labelling harvested recordings of
du'as and ziyarat the 505-text corpus doesn't have (Ziyarat Arbaeen, Nahiya, the
fifteen Munajat, ...).

Source: github.com/kazemcodes/MafatihDecoder (the book decoded from a Mafatih app):
per chapter, articles of items typed Title / About / Text (vowelled Arabic, cut at the
printed book's line breaks) / Translated (Persian). A block = a Title and the Arabic
Text items after it. Blocks whose words are mostly in the corpus already (or in a
held-out du'a) are dropped. The app's corpus is not touched: these only give the
labeller (harvest_label.py) more texts to recognise, and the training export more
labelled audio.

    python scripts/mafatih_texts.py      # -> data/harvest/extra_texts.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.corpus import load_all  # noqa: E402
from dua_recognition.text import normalize  # noqa: E402

SRC = Path(r"D:\dua-data\texts")
FILES = ["babe1.json", "Babe2.json", "Babe3.json", "Khateme1.json", "Khateme2.json", "Molhaghate1.json",
         "Molhaghate2.json"]
OUT = ROOT / "data" / "harvest" / "extra_texts.json"
HELD_OUT = ("dua-tawassul", "ziyarat-ashura")


def _items(d: dict):
    for sec in d.get("sections", []):
        for art in sec.get("articles", []):
            yield from art.get("items", [])
        yield from sec.get("articleItems", [])


def blocks() -> list[dict]:
    out = []
    for fn in FILES:
        d = json.loads((SRC / fn).read_text(encoding="utf-8"))
        title, segs = None, []

        def flush():
            words = sum(len(normalize(s).split()) for s in segs)
            if title and words >= 25:
                out.append({"title": title, "chapter": fn.split(".")[0], "segments": list(segs)})

        for it in _items(d):
            t, c = it.get("type"), (it.get("content") or "").strip()
            if t == "Title":
                flush()
                title, segs = c, []
            elif t == "Text" and c:
                # Strip the book's brackets and verse marks; keep the vowelled Arabic.
                c = re.sub(r"[()«»\[\]{}*]", " ", c)
                segs.append(re.sub(r"\s+", " ", c).strip())
        flush()
    return out


def main() -> None:
    duas = load_all()
    corpus = set()
    held = set()
    for k, dua in duas.items():
        w = [x for s in dua.segments for x in normalize(s.arabic).split()]
        g = set(zip(w, w[1:], w[2:]))
        corpus |= g
        if k in HELD_OUT:
            held |= g
    kept, dropped = [], {"in corpus": 0, "held-out": 0, "repeat": 0}
    seen: set = set()
    # Longest first, so a repeat of part of a longer text is the one dropped.
    for i, b in sorted(enumerate(blocks()), key=lambda ib: -sum(len(s.split()) for s in ib[1]["segments"])):
        w = [x for s in b["segments"] for x in normalize(s).split()]
        g = list(zip(w, w[1:], w[2:]))
        if not g:
            continue
        if sum(x in held for x in g) / len(g) >= 0.2:
            dropped["held-out"] += 1
            continue
        if sum(x in corpus for x in g) / len(g) >= 0.5:
            dropped["in corpus"] += 1
            continue
        if sum(x in seen for x in g) / len(g) >= 0.8:
            dropped["repeat"] += 1
            continue
        seen |= set(g)
        kept.append({"dua_id": f"mafatih-{i:04d}", "dua_name_en": "", "dua_name_ar": b["title"],
                     "chapter": b["chapter"], "source": "github.com/kazemcodes/MafatihDecoder",
                     "segments": [{"segment_id": j, "arabic": s} for j, s in enumerate(b["segments"])],
                     "words": len(w)})
    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(kept)} extra texts, {sum(k['words'] for k in kept)} words; dropped {dropped} -> {OUT}")
    for k in sorted(kept, key=lambda k: -k["words"])[:25]:
        print(f"  {k['dua_id']} {k['words']:5d}  {k['dua_name_ar'][:60]}")


if __name__ == "__main__":
    main()
