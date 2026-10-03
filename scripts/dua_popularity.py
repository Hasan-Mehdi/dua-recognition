#!/usr/bin/env python
"""How often each text is recited: harvest recordings per corpus text -> data/dua_popularity.json.

The tracker's prior over du'as (TrackerConfig.popularity) leans on this: when someone
opens with words many texts share (السلام عليك يا أبا عبد الله, اللهم إني أسألك), the
texts people actually recite are the likelier ones. A recording counts once per text it
holds a usable span of (harvest_label.py labels; the Mafatih blocks the corpus has taken
in counted under their corpus ids). Uploads are a fair measure of what gets recited:
Kumayl 794 recordings, Ziyarat Ashura 580, most duas.org-only texts none.

    python scripts/dua_popularity.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mafatih_corpus import TEXTS  # noqa: E402

LABELS = ROOT / "data" / "harvest" / "labels"
OUT = ROOT / "data" / "dua_popularity.json"


def main() -> None:
    block = {t[0]: t[1] for t in TEXTS}
    texts = sorted(p.stem for p in (ROOT / "data" / "duas").glob("*.json"))
    recs: Counter = Counter()
    for p in LABELS.glob("*/*.json"):
        if p.name.endswith(".captions.json"):
            continue
        try:
            lab = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        recs.update({block.get(sp["dua"], sp["dua"]) for sp in lab.get("spans", []) if sp.get("usable") and sp["dua"]})
    out = {t: recs.get(t, 0) for t in texts}
    OUT.write_text(json.dumps({"note": "harvest recordings per corpus text (scripts/dua_popularity.py)",
                               "recordings": out}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{sum(v > 0 for v in out.values())} of {len(out)} texts recorded, {sum(out.values())} recordings -> "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
