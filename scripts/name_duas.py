#!/usr/bin/env python
"""Give every text a name a person would recognise.

The scraped texts kept their source block titles, and many of those name
nothing: 50 are "Instruction Block", and the Ramadan pages call their du'as by
who narrated them ("Syed Ibn Baqi" x24, "Holy Prophet's Dua" x24) with no day.
This rewrites dua_name_en in data/duas/*.json:

  - an attribution title on a Ramadan day page -> "Ramadan Day 13 · Sayyid Ibn Baqi"
  - a title that names nothing -> "<page title> · <opening words>", e.g.
    "Maghrib and Isha Taqeebat · Allāhumma hādhihī ṣalātī"
  - DuaPlayer's "Day 13" / "Dua Night 22" -> "Ramadan Day 13" / "Ramadan Night 22"
  - any name still shared by two texts gets its opening words too.

Opening words come from translit.py, the same reading the app shows. Page
titles come from the raw duas.org pages (scripts/fetch_duasorg.py download).

    python scripts/name_duas.py [--pages data/duasorg] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dua_recognition.translit import transliterate  # noqa: E402

DUAS = ROOT / "data" / "duas"

# Titles that name nothing on their own.
EMPTY = re.compile(r"^(instruction block|new dua|method|recommended acts|continue.*)?$", re.I)
# Who a Ramadan day du'a is narrated from, as the pages title them -> how we say it.
NARRATORS = [
    (re.compile(r"syed?\s*(ibn\s*)?baqi", re.I), "from Sayyid Ibn Baqi"),
    (re.compile(r"holy prophet", re.I), "from the Prophet (s)"),
    (re.compile(r"zainul abideen|sajjad", re.I), "from Imam Zayn al-Abidin (a)"),
    (re.compile(r"abi qarrih", re.I), "from Muhammad ibn Abi Qurrah"),
    (re.compile(r"book of companions", re.I), "from the Book of Companions"),
    (re.compile(r"^dua 2\b", re.I), "Second du'a"),
    (re.compile(r"^day \d+ - dua$|^night dua$", re.I), ""),
]


def page_titles(pages: Path) -> dict[str, str]:
    out = {}
    for p in pages.glob("*.json"):
        if p.name.endswith("-timing.json"):
            continue
        try:
            title = json.loads(p.read_text(encoding="utf-8")).get("title") or ""
        except json.JSONDecodeError:
            continue
        out[p.stem.lower()] = clean_title(title)
    return out


# Source slips and lower-cased names in titles.
FIXES = [(r"\bArarah\b", "Arafah"), (r"\beffor\b", "effort"), (r"\bdiffult\b", "difficult"),
         (r"\bgods\b", "God's"), (r"\bgod\b", "God"), (r"\bquran\b", "Qur'an"), (r"\bramadan\b", "Ramadan"),
         (r"\bsatan\b", "Satan"), (r"\bfriday\b", "Friday"), (r"\barafah\b", "Arafah")]


def fix_words(t: str) -> str:
    for pat, good in FIXES:
        t = re.sub(pat, good, t)
    return t


def clean_title(t: str) -> str:
    t = re.sub(r"\s*\(.*?\)\s*", " ", t)  # "(duas and amaal after ...)"
    t = re.sub(r"\s+", " ", t).strip(" -:")
    return t[:1].upper() + t[1:]


def opening(raw: dict, words: int = 4) -> str:
    """The first few words of the text's reading, as the app shows it."""
    tl = next((t for t in (transliterate(s["arabic"]).rstrip(",.;:!? ") for s in raw["segments"]) if t), "")
    ws = tl.split()
    return " ".join(ws[:words]) + ("…" if len(ws) > words else "")


def ramadan_day(page_id: str) -> str | None:
    m = re.match(r"ramadan-day-0?(\d+)$", page_id)
    return m and m.group(1)


# Texts whose source title is wrong for what they hold.
OVERRIDES = {
    # From the Namaz-e-Wahshat page, but the text is Ayat al-Kursi itself (2:255-257).
    "duasorg-namaz-e-wahshat": "Ayat al-Kursi",
}


def new_name(raw: dict, titles: dict[str, str]) -> str:
    did, name = raw["dua_id"], raw.get("dua_name_en", "").strip()
    if did in OVERRIDES:
        return OVERRIDES[did]
    m = re.match(r"^dua-ramadan-(\d+)(-night)?$", did)
    if m:
        return f"Ramadan {'Night' if m.group(2) else 'Day'} {m.group(1)}"
    m = re.match(r"^Sahifa 0?(\d+): (.*)$", name)
    if m:
        return fix_words(f"Sahifa {m.group(1)}: {m.group(2)[:1].upper()}{m.group(2)[1:]}")
    if not did.startswith("duasorg-"):
        return name
    page_id = re.sub(r"\.html?$", "", raw.get("source", "").rsplit("/", 1)[-1]).lower()
    page = fix_words(titles.get(page_id, page_id.replace("-", " ").capitalize()))
    day = ramadan_day(page_id)
    if day:
        for pat, who in NARRATORS:
            if pat.search(name):
                return f"Ramadan Day {day} · {who or opening(raw)}"
        if EMPTY.match(name):
            return f"Ramadan Day {day} · {opening(raw)}"
    if EMPTY.match(name):
        return f"{page} · {opening(raw)}"
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default=str(ROOT / "data" / "duasorg"), help="raw duas.org pages")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    titles = page_titles(Path(args.pages))
    if not titles:
        sys.exit(f"no duas.org pages in {args.pages} (scripts/fetch_duasorg.py download)")
    raws = {p: json.loads(p.read_text(encoding="utf-8")) for p in sorted(DUAS.glob("*.json"))}
    names = {p: new_name(r, titles) for p, r in raws.items()}
    # Still shared: tell them apart by how they open, then (if they open alike) by number.
    count = collections.Counter(names.values())
    for p, n in list(names.items()):
        if count[n] > 1 and "·" not in n:
            names[p] = f"{n} · {opening(raws[p])}"
    count = collections.Counter(names.values())
    seen: collections.Counter = collections.Counter()
    for p, n in names.items():
        if count[n] > 1:
            seen[n] += 1
            names[p] = f"{n} ({seen[n]})"
    count = collections.Counter(names.values())
    changed = 0
    for p, raw in raws.items():
        old = raw.get("dua_name_en", "")
        if names[p] == old:
            continue
        changed += 1
        print(f"{raw['dua_id']}: {old!r} -> {names[p]!r}")
        if not args.dry_run:
            raw["dua_name_en"] = names[p]
            p.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    dup = {n: k for n, k in count.items() if k > 1}
    print(f"{changed} renamed; names still shared: {dup or 'none'}")


if __name__ == "__main__":
    main()
