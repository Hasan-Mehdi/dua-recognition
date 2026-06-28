#!/usr/bin/env python
"""Download du'a recitations listed in a sources file into data/raw/.

Sources file is CSV: `url,dua_id,reciter` (one per line). Every download is
appended to data/manifest.csv so the corpus stays auditable. Audio rights are
your responsibility — log the source and only keep what you're allowed to use.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"


def download(url: str, dua_id: str) -> Path:
    out_dir = RAW / dua_id
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "yt-dlp", "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            url,
        ],
        check=True,
    )
    return out_dir


def log(url: str, dua_id: str, reciter: str) -> None:
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["url", "dua_id", "reciter", "license"])
        w.writerow([url, dua_id, reciter, "unknown"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", help="CSV of url,dua_id,reciter")
    args = ap.parse_args()

    for row in csv.reader(Path(args.sources).read_text(encoding="utf-8").splitlines()):
        if not row or row[0].strip().lower() in ("", "url"):
            continue
        url, dua_id, reciter = (row + ["", "", ""])[:3]
        download(url.strip(), dua_id.strip())
        log(url.strip(), dua_id.strip(), reciter.strip())


if __name__ == "__main__":
    main()
