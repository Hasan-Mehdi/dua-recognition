#!/usr/bin/env python
"""archive.org side of the recitation harvest.

archive.org holds whole uploaded du'a libraries (Mafatih sets, a reciter's
complete du'as, Ramadan collections) behind open search and metadata APIs.
Most "du'a" items there are lectures or Quran, so files are picked by name:
an audio file is fetched only when its own name, or its item's title, names
a du'a/ziyarat text (FILE_HINT) and no test reciter (harvest.excluded).

    python scripts/harvest_archive.py discover    # items -> data/harvest/archive/items.jsonl
    python scripts/harvest_archive.py files       # file lists -> files.jsonl
    python scripts/harvest_archive.py download -j 4
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from harvest import HARVEST, append, excluded, read_jsonl  # noqa: E402

OUT = HARVEST / "archive"
HEAD = {"User-Agent": "dua-recognition research harvest (contact: repo owner)"}
QUERIES = [
    "كميل", "التوسل", "الندبة", "زيارة عاشوراء", "الجوشن", "الافتتاح", "ابي حمزة الثمالي", "الثمالي",
    "الصحيفة السجادية", "مفاتيح الجنان", "زيارة وارث", "زيارة امين الله", "الجامعة الكبيرة", "زيارة آل ياسين",
    "دعاء العهد", "دعاء الفرج", "حديث الكساء", "دعاء الصباح", "دعاء السمات", "المشلول", "دعاء المجير",
    "دعاء عرفة", "مكارم الاخلاق", "المناجاة الشعبانية", "المناجاة", "ادعية رمضان", "ادعية شهر رمضان",
    "دعاء ابي حمزة", "دعاء يستشير", "زيارة الامام الحسين", "زيارة الامام الرضا", "زيارة الجامعة",
    "دعاء البهاء", "دعاء السحر", "دعاء ادريس", "دعاء النور", "دعاء العديلة", "تعقيبات", "ادعية الايام",
    "ادعية الاسبوع", "باسم الكربلائي دعاء", "مهدي سماواتي", "كمیل", "توسل", "ندبه", "عاشورا", "جوشن",
    "افتتاح", "ابوحمزه", "مفاتیح", "زیارت", "دعای", "صحیفه سجادیه", "مناجات",
    "kumail", "kumayl", "komeil", "komail", "tawassul", "tawasul", "nudba", "nudbah", "ziyarat", "ziarat",
    "ziyarah", "jawshan", "joshan", "iftitah", "abu hamza", "thumali", "sumali", "sajjadiya", "sajjadia",
    "mafatih", "mafateeh", "arafah dua", "dua arafa", "dua sabah", "dua ahad", "dua faraj", "hadith kisa",
    "hadees e kisa", "samavati", "karbalai dua", "shia dua", "shia duas", "duas", "dua collection",
    "ramadan duas", "munajat", "mashlool", "mujeer", "simat", "samaat", "yastashir", "ziyarat warith",
    "aminullah", "aale yaseen", "jamia kabira", "ahlulbayt", "imam mahdi dua", "dua e",
]
# A file (or its item) must name a du'a/ziyarat text. Bare "دعاء" is too broad on
# archive.org (Sunni du'a lectures), so it needs one of the text names with it.
FILE_HINT = re.compile(
    r"كميل|كمیل|كمیل|کمیل|توسل|ندب|عاشور|جوشن|افتتاح|ثمالي|ثمالی|حمز[هة]|سجادي|سجادی|مفاتيح|مفاتیح|وارث|"
    r"امين ?الله|امین ?الله|جامع[هة]|ياسين|یاسین|عهد|الفرج|فرج|كساء|کسا|الصباح|صباح|سمات|مشلول|مجير|مجیر|"
    r"عرف[هة]|مكارم|مکارم|مناجا|يستشير|یستشیر|البهاء|بهاء|سحر|ادريس|ادریس|النور|عديل|عدیل|تعقيب|تعقیب|"
    r"رمضان|زيار|زیار|رجب|شعبان|"
    r"kumail|kumayl|komeil|komail|kumeil|tawass?ul|nudba|ashura|ashoora|jawshan|joshan|iftitah|thumali|sumali|"
    r"hamza|sajjad|mafat|warith|waris|aminallah|aminullah|ameenallah|yaseen|yasin|jamia|jame[ae]|ahad\b|ahd\b|"
    r"faraj|kisa|kisaa|sabah|simat|samaat|mashlool|mujeer|mujir|arafa|makarim|munajat|yastash|baha|sahar|"
    r"idris|noor|adeela|taqeeb|ramadan|ramzan|rajab|shaban|ziyar|ziar|zyarat", re.I)
AUDIO_EXT = (".mp3", ".ogg", ".m4a", ".opus", ".wav", ".flac", ".aac", ".wma", ".amr")
# archive.org's du'a-named items are mostly lectures about du'as, Quran mushafs
# (Ramadan taraweeh) and fatwas; none of those are recitations of a du'a text.
JUNK = re.compile(r"شرح|مصحف|سورة|سوره|تلاو|تراويح|فتاو|فتوى|درس|دروس|محاضر|خطب|قاعدة جليلة|قاعدة الجلية|الفوزان|"
                  r"ختمة|تفسير|الطحاوية|العثيمين|صوتيات الشيخ|أناشيد|اناشيد|نشيد|مؤتمر|الوسيلة|مبحث|أسئلة|اسئلة|"
                  r"القبور|برنامج|حلقة|الحلقة|مسلسل|lecture|khutba|sermon|surah|quran|qur'an|tafsir|tafseer|fatwa|"
                  r"lesson|class|nasheed|episode", re.I)
DUA_START = re.compile(r"^\W*(دعاء|دعای|زيارة|زیارت|مناجاة|مناجات|حديث الكساء|dua|du'a|doa|ziyarat|ziarat|munajat)", re.I)


def get(url, params=None, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(10 * (k + 1))
        except Exception:  # noqa: BLE001
            time.sleep(5 * (k + 1))
    return None


def cmd_discover(args) -> None:
    have = {r["identifier"] for r in read_jsonl(OUT / "items.jsonl")}
    for q in QUERIES:
        d = get("https://archive.org/advancedsearch.php", {
            "q": f"(title:({q}) OR description:({q}) OR subject:({q})) AND mediatype:(audio OR movies)",
            "fl[]": ["identifier", "title", "item_size", "downloads", "creator"], "rows": 2000, "output": "json"})
        docs = (d or {}).get("response", {}).get("docs", [])
        new = 0
        for doc in docs:
            if doc["identifier"] in have:
                continue
            have.add(doc["identifier"])
            append(OUT / "items.jsonl", {**doc, "query": q})
            new += 1
        print(f"{q}: {len(docs)} items, {new} new", flush=True)
        time.sleep(1)


def cmd_files(args) -> None:
    done = {r["identifier"] for r in read_jsonl(OUT / "listed.jsonl")}
    items = [r for r in read_jsonl(OUT / "items.jsonl") if r["identifier"] not in done]
    print(f"{len(items)} items to list", flush=True)
    lock = threading.Lock()
    q = list(reversed(items))
    stats = Counter()

    def worker():
        while True:
            with lock:
                if not q:
                    return
                it = q.pop()
            ident = it["identifier"]
            md = get(f"https://archive.org/metadata/{ident}")
            files = (md or {}).get("files") or []
            title = str((md or {}).get("metadata", {}).get("title") or it.get("title") or "")
            item_hint = bool(FILE_HINT.search(title))
            n = 0
            for f in files:
                name = f.get("name", "")
                if not name.lower().endswith(AUDIO_EXT) or f.get("source") == "derivative" and not name.lower().endswith(".mp3"):
                    continue
                if f.get("source") == "derivative" and any(
                        g.get("name", "").rsplit(".", 1)[0] == name.rsplit(".", 1)[0] and g.get("source") == "original"
                        and g.get("name", "").lower().endswith(".mp3") for g in files):
                    continue  # keep the original mp3, not its derivative copy
                base = Path(name).name
                if not (FILE_HINT.search(base) or (item_hint and len(files) < 400)):
                    continue
                try:
                    length = float(f.get("length") or 0)
                except ValueError:
                    length = 0.0 if ":" not in str(f.get("length")) else sum(
                        float(x) * 60 ** i for i, x in enumerate(reversed(str(f["length"]).split(":"))))
                size = int(f.get("size") or 0)
                append(OUT / "files.jsonl", {"identifier": ident, "name": name, "size": size, "length": length,
                                             "item_title": title[:200], "query": it.get("query")})
                n += 1
            append(OUT / "listed.jsonl", {"identifier": ident, "files": n})
            with lock:
                stats["items"] += 1
                stats["files"] += n
                if stats["items"] % 100 == 0:
                    print(f"  {stats['items']} items listed, {stats['files']} files kept", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {dict(stats)}", flush=True)


def fid(r: dict) -> str:
    import hashlib
    return "ia-" + hashlib.sha1(f"{r['identifier']}/{r['name']}".encode()).hexdigest()[:16]


def cmd_download(args) -> None:
    have = {r["id"] for r in read_jsonl(OUT / "index.jsonl")}
    failed = Counter(r["id"] for r in read_jsonl(OUT / "failed.jsonl"))
    rows, why, seen = [], Counter(), set()
    for r in read_jsonl(OUT / "files.jsonl"):
        k = fid(r)
        if k in have or k in seen or failed[k] >= 2:
            continue
        seen.add(k)
        reason = excluded({"title": f"{r['item_title']} {r['name']}", "id": k})
        if reason:
            why[reason.split(" (")[0]] += 1
            continue
        base = Path(r["name"]).name
        if JUNK.search(f"{r['item_title']} {r['name']}"):
            why["lecture/quran"] += 1
            continue
        # The file must name a text itself, unless the whole item is one du'a/ziyarat.
        if not (FILE_HINT.search(base) or DUA_START.search(r["item_title"])):
            why["unnamed file"] += 1
            continue
        if r["length"] and not (args.min_s <= r["length"] <= args.max_s):
            why["duration"] += 1
            continue
        if r["size"] > args.max_mb * 1e6:
            why["size"] += 1
            continue
        rows.append(r)
    # Spread over items so one big library doesn't take the whole night.
    by_item: dict[str, list] = {}
    for r in rows:
        by_item.setdefault(r["identifier"], []).append(r)
    order = []
    while by_item:
        for k in list(by_item):
            order.append(by_item[k].pop(0))
            if not by_item[k]:
                del by_item[k]
    if args.limit:
        order = order[: args.limit]
    print(f"plan: {len(order)} files, {sum(r['length'] for r in order) / 3600:.0f} h known length; "
          f"skipped {dict(why)}", flush=True)
    audio = OUT / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    q = list(reversed(order))
    lock = threading.Lock()
    stats = Counter()
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not q:
                    return
                r = q.pop()
            k = fid(r)
            ext = Path(r["name"]).suffix.lower()
            dst = audio / f"{k}{ext}"
            url = f"https://archive.org/download/{quote(r['identifier'])}/{quote(r['name'])}"
            try:
                with requests.get(url, headers=HEAD, timeout=90, stream=True) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"http {resp.status_code}")
                    tmp = dst.with_suffix(ext + ".part")
                    with tmp.open("wb") as f:
                        for block in resp.iter_content(1 << 20):
                            f.write(block)
                    tmp.replace(dst)
            except Exception as e:  # noqa: BLE001
                append(OUT / "failed.jsonl", {"id": k, "err": str(e)[:200]})
                stats["fail"] += 1
                continue
            append(OUT / "index.jsonl", {"id": k, "audio": dst.name, "identifier": r["identifier"], "name": r["name"],
                                         "title": f"{r['item_title']} / {Path(r['name']).name}", "duration": r["length"],
                                         "channel": r["identifier"], "channel_id": f"ia:{r['identifier']}",
                                         "url": url, "group": "archive", "query": r.get("query")})
            with lock:
                stats["ok"] += 1
                stats["s"] += r["length"]
                if stats["ok"] % 50 == 0:
                    print(f"  {stats['ok']} files, {stats['s'] / 3600:.1f} h, fail {stats['fail']}, "
                          f"{(time.time() - t0) / 60:.0f} min", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {dict(stats)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "files", "download"])
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=20)
    ap.add_argument("--max-s", type=float, default=3 * 3600)
    ap.add_argument("--max-mb", type=float, default=400)
    args = ap.parse_args()
    {"discover": cmd_discover, "files": cmd_files, "download": cmd_download}[args.stage](args)


if __name__ == "__main__":
    main()
