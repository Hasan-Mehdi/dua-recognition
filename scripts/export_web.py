#!/usr/bin/env python
"""Write web/corpus.json so the front end can run with no server (on-device mode).

Includes translations when the local DuaPlayer cache has them
(scripts/fetch_duaplayer.py), and a transliteration for every line
(translit.py); like the cache itself, the output isn't committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.corpus import load_all, web_json  # noqa: E402


def corpus_json() -> list[dict]:
    return web_json(load_all())


if __name__ == "__main__":
    out = ROOT / "web" / "corpus.json"
    tmp = out.with_suffix(".json.tmp")  # then a rename: a page loading meanwhile never reads half a file
    tmp.write_text(json.dumps(corpus_json(), ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
    print("then: python -m http.server -d web 8080  (models go in web/models/, see scripts/export_onnx.py)")
