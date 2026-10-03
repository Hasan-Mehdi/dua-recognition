#!/usr/bin/env python
"""How often each text is recited: harvest recordings per corpus text -> data/dua_popularity.json.

The tracker's prior over du'as (TrackerConfig.popularity) leans on this: when someone
opens with words many texts share (السلام عليك يا أبا عبد الله, اللهم إني أسألك), the
texts people actually recite are the likelier ones. Uploads are a fair measure of what
gets recited: Kumayl 794 recordings, Ziyarat Ashura 580, most duas.org-only texts none.

A recording counts for a text only if its span there reads at least MIN_OWN placed lines
no other text reads (a line holding a PASSAGE-word run found in no other text). The
labeller has to pick one of the texts that share a passage word for word, so a recording
of Ayat al-Kursi alone would otherwise count for whichever of its three texts it picked
(Sahifa 54 got 126 that way). The Mafatih blocks the corpus has taken in count under
their corpus ids (their book lines are checked the same way).

    python scripts/dua_popularity.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from mafatih_corpus import EXTRA, TEXTS  # noqa: E402

from dua_recognition.corpus import load_all  # noqa: E402
from dua_recognition.text import normalize  # noqa: E402

LABELS = ROOT / "data" / "harvest" / "labels"
OUT = ROOT / "data" / "dua_popularity.json"
PASSAGE = 8  # words: a run this long found in another text is a shared passage
MIN_OWN = 3  # placed lines of the text's own words a recording needs to count for it


def _own_lines(texts: dict[str, list[tuple[int, list[str]]]]) -> dict[str, set[int]]:
    """text -> its lines holding a PASSAGE-word run no other text has."""
    words = {k: [w for _, ws in segs for w in ws] for k, segs in texts.items()}
    seen: dict[tuple, set[str]] = {}
    for k, w in words.items():
        for e in range(PASSAGE - 1, len(w)):
            seen.setdefault(tuple(w[e - PASSAGE + 1 : e + 1]), set()).add(k)
    out = {}
    for k, segs in texts.items():
        w, own, i = words[k], set(), 0
        for seg, ws in segs:
            i += len(ws)
            if any(len(seen[tuple(w[e - PASSAGE + 1 : e + 1])]) == 1 for e in range(max(PASSAGE - 1, i - len(ws)), i)):
                own.add(seg)
        out[k] = own
    return out


def main() -> None:
    corpus = load_all()
    texts = {k: [(s.id, normalize(s.arabic).split()) for s in d.segments] for k, d in corpus.items()}
    block = {t[0]: t[1] for t in TEXTS}
    # The labels of the added Mafatih texts name the book's lines, not ours.
    for b in json.loads(EXTRA.read_text(encoding="utf-8")):
        if b["dua_id"] in block:
            texts[b["dua_id"]] = [(s["segment_id"], normalize(s["arabic"]).split()) for s in b["segments"]]
    own = _own_lines(texts)
    recs: Counter = Counter()
    for p in LABELS.glob("*/*.json"):
        if p.name.endswith(".captions.json"):
            continue
        try:
            lab = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        got = set()
        for sp in lab.get("spans", []):
            d = sp.get("dua")
            if sp.get("usable") and d in own and sum(ln["ok"] and ln["seg"] in own[d] for ln in sp["lines"]) >= MIN_OWN:
                got.add(block.get(d, d))
        recs.update(got)
    out = {t: recs.get(t, 0) for t in sorted(corpus)}
    OUT.write_text(json.dumps({"note": "harvest recordings per corpus text, counted where they read the text's own "
                                       "lines (scripts/dua_popularity.py)", "recordings": out},
                              ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{sum(v > 0 for v in out.values())} of {len(out)} texts recorded, {sum(out.values())} recordings -> "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
