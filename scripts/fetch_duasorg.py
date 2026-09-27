#!/usr/bin/env python
"""Add du'a texts and timed recordings from duas.org.

duas.org serves every page as JSON (data_v2/<page>.json): one or more du'as,
each a list of line segments (Arabic, transliteration, translation), usually
with one recording (mp3.duas.org). A dozen pages also have human line
timings (data_v2/<page>-timing.json, one start time per rendered segment).

    python scripts/fetch_duasorg.py download            # page JSON + timing files
    python scripts/fetch_duasorg.py texts --out DIR      # new texts -> DIR
    python scripts/fetch_duasorg.py lines               # their translations + readings -> data/lines/
    python scripts/fetch_duasorg.py timed               # timings -> data/duasorg_timed/
    python scripts/fetch_duasorg.py audio               # recordings, if the disk has room
    python scripts/fetch_duasorg.py untimed --texts DIR  # untimed recordings -> data/untimed/duasorg/

A du'a we already have (same opening, or its words mostly contained in one of
ours) isn't added again. Its recording is still useful: timings made against
duas.org's line split are carried over to ours by aligning the two texts word
by word.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_duaspro import _bare, _opening  # noqa: E402

DUAS = ROOT / "data" / "duas"
PAGES = ROOT / "data" / "duasorg"
TIMED = ROOT / "data" / "duasorg_timed"
LINES = ROOT / "data" / "lines"  # corpus.LINES_DIR
SITE = "https://www.duas.org"
UA = {"User-Agent": "dua-recognition/0.2 (research)"}
CONTINUED = re.compile(r"^\s*continue", re.I)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def cmd_download(args) -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    index = json.loads(_get(f"{SITE}/data_v2/search_index.json"))
    items = index if isinstance(index, list) else next(v for v in index.values() if isinstance(v, list))
    timed = {i["id"] for i in items if (i.get("meta") or {}).get("hasTiming")}
    for page in sorted({i["id"] for i in items}):
        for name in [f"{page}.json"] + ([f"{page}-timing.json"] if page in timed else []):
            path = PAGES / name
            if path.exists():
                continue
            try:
                path.write_bytes(_get(f"{SITE}/data_v2/{name}"))
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: {e}")
            time.sleep(0.8)


def pages() -> dict[str, dict]:
    out = {}
    for p in sorted(PAGES.glob("*.json")):
        if p.name.endswith("-timing.json"):
            continue
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return out


def blocks(page: dict) -> list[dict]:
    """The page's du'as, with "Continued..." blocks folded into the one before."""
    out: list[dict] = []
    for d in page.get("duas") or []:
        kept = [s for s in d.get("segments") or []
                if s.get("type") != "instruction" and (s.get("arabic") or "").strip()]
        segs = [s["arabic"].strip() for s in kept]
        extra = [{"tl": (s.get("transliteration") or "").strip(), "en": (s.get("translation") or "").strip()}
                 for s in kept]
        if out and CONTINUED.match(d.get("title") or ""):
            out[-1]["segments"] += segs
            out[-1]["extra"] += extra
            continue
        out.append({"id": d.get("id"), "title": (d.get("title") or "").strip(), "segments": segs,
                    "extra": extra, "audio": d.get("audio") or "", "youtube": d.get("youtube") or ""})
    return out


def corpus_texts(extra_dirs: list[Path]) -> dict[str, list[str]]:
    out = {}
    for d in [DUAS] + extra_dirs:
        for p in d.glob("*.json"):
            raw = json.loads(p.read_text(encoding="utf-8"))
            out[raw["dua_id"]] = [s["arabic"] for s in raw["segments"]]
    return out


def same_as(segs: list[str], texts: dict[str, list[str]], openings: dict[str, str]) -> str | None:
    """Our id for this text, if we already have it."""
    hit = openings.get(_opening(segs))
    if hit:
        return hit
    grams = _grams(segs)
    if not grams:
        return None
    best, best_score = None, 0.0
    for did, t in texts.items():
        ours = _grams_cached(did, t)
        # Most of this text's letter 5-grams appear in one of ours, and the two are
        # of comparable length: the same du'a in another edition or spelling.
        contained = len(grams & ours) / len(grams)
        size = min(len(grams), len(ours)) / max(len(grams), len(ours))
        score = contained if size > 0.6 else 0
        if score > best_score:
            best, best_score = did, score
    return best if best_score > 0.7 else None


def _grams(segs: list[str], n: int = 5) -> set[str]:
    # Spaces dropped: editions split "و انت" / "وانت" differently.
    s = "".join(_bare(x).replace(" ", "") for x in segs)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


_GRAM_CACHE: dict[str, set[str]] = {}


def _grams_cached(did: str, segs: list[str]) -> set[str]:
    if did not in _GRAM_CACHE:
        _GRAM_CACHE[did] = _grams(segs)
    return _GRAM_CACHE[did]


def cmd_texts(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = corpus_texts([out_dir])
    openings = {_opening(t): did for did, t in texts.items()}
    added = dup = short = 0
    for page_id, page in pages().items():
        bl = [b for b in blocks(page) if len(b["segments"]) >= args.min_segments]
        for i, b in enumerate(bl):
            hit = same_as(b["segments"], texts, openings)
            if hit:
                dup += 1
                continue
            did = f"duasorg-{page_id}" + (f"-{i + 1}" if len(bl) > 1 else "")
            did = re.sub(r"[^a-z0-9-]+", "-", did.lower()).strip("-")
            (out_dir / f"{did}.json").write_text(json.dumps({
                "dua_id": did,
                "dua_name_en": b["title"] or page_id.replace("-", " "),
                "dua_name_ar": "",
                "source": f"{SITE}/{page_id}.html",
                "segments": [{"segment_id": k + 1, "arabic": s} for k, s in enumerate(b["segments"])],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            texts[did] = b["segments"]
            openings[_opening(b["segments"])] = did
            added += 1
        short += sum(1 for b in blocks(page) if 0 < len(b["segments"]) < args.min_segments)
    print(f"{added} new texts, {dup} already in the corpus, {short} too short (< {args.min_segments} lines)")


def cmd_lines(args) -> None:
    """Each duas.org text's translation and reading, line by line, into data/lines/.

    A text is one block of its page (texts stage), so the block with the same
    lines gives them in order; a text edited since falls back to matching each
    line's letters anywhere on the page."""
    LINES.mkdir(parents=True, exist_ok=True)
    all_pages = pages()
    texts = full = lines = found = 0
    for p in sorted(DUAS.glob("duasorg-*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        page = all_pages.get(raw.get("source", "").rsplit("/", 1)[-1].removesuffix(".html"))
        if page is None:
            print(f"  {raw['dua_id']}: page not downloaded")
            continue
        ours = [s["arabic"] for s in raw["segments"]]
        bl = blocks(page)
        block = next((b for b in bl if b["segments"] == ours), None)
        if block:
            extra = block["extra"]
        else:
            by_letters = {}
            for b in bl:
                for s, e in zip(b["segments"], b["extra"]):
                    by_letters.setdefault(_bare(s), e)
            extra = [by_letters.get(_bare(s), {}) for s in ours]
        out = {str(s["segment_id"]): e for s, e in zip(raw["segments"], extra) if e.get("en") or e.get("tl")}
        (LINES / f"{raw['dua_id']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        texts += 1
        full += len(out) == len(ours)
        lines += len(ours)
        found += len(out)
    print(f"{texts} texts, {full} with every line; {found}/{lines} lines")


def cmd_prune(args) -> None:
    """Drop staged texts that would only confuse identification.

    A fragment (most of it inside another text: a long du'a split over several
    blocks) adds nothing; a compilation (a page that swallows most of a du'a we
    already have) competes with that du'a for the same recitation.
    """
    staged_dir = Path(args.out)
    ours = corpus_texts([])
    staged = {p.stem: [s["arabic"] for s in json.loads(p.read_text(encoding="utf-8"))["segments"]]
              for p in staged_dir.glob("*.json")}
    grams = {k: _grams(v) for k, v in {**ours, **staged}.items()}
    dropped: dict[str, str] = {}
    # Compilations first: a page bundling several du'as people recite on their own.
    for a in staged:
        ga = grams[a]
        inside = [b for b, gb in grams.items()
                  if b != a and gb and len({**ours, **staged}[b]) >= 5 and len(ga & gb) / len(gb) >= args.overlap
                  and len(gb) < 0.8 * len(ga)]
        if len(inside) >= 2:
            dropped[a] = "compilation of " + ", ".join(sorted(inside)[:4])
    # Shortest first, so of two copies of one text the longer survives.
    for a in sorted((k for k in staged if k not in dropped), key=lambda k: len(grams[k])):
        ga = grams[a]
        if not ga:
            dropped[a] = "empty"
            continue
        for b, gb in grams.items():
            if b == a or b in dropped or not gb:
                continue
            inter = len(ga & gb)
            if inter / len(ga) >= args.overlap:
                dropped[a] = f"fragment of {b}"
                break
            if b in ours and len(ours[b]) >= 10 and inter / len(gb) >= args.overlap:
                dropped[a] = f"contains {b}"
                break
    for a, why in sorted(dropped.items()):
        (staged_dir / f"{a}.json").unlink()
        print(f"  dropped {a}: {why}")
    print(f"{len(dropped)} dropped, {len(staged) - len(dropped)} staged texts remain")


def carry_over(src_segs: list[str], src_starts: list[float], dst_segs: list[str]) -> dict[int, float]:
    """Start times on another line split of the same text, via a word alignment.

    A destination line starts at its first word; that word's time is taken
    from the source line it falls in, interpolated by position within it.
    """
    src_words, src_pos = [], []
    for k, s in enumerate(src_segs):
        w = _bare(s).split()
        src_words += w
        src_pos += [(k, j, len(w)) for j in range(len(w))]
    dst_words, dst_first = [], {}
    for k, s in enumerate(dst_segs):
        w = _bare(s).split()
        if w:
            dst_first[len(dst_words)] = k + 1
        dst_words += w
    word_map = {}
    sm = difflib.SequenceMatcher(None, dst_words, src_words, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for x in range(n):
            word_map[a + x] = b + x
    out = {}
    for di, seg_id in dst_first.items():
        si = word_map.get(di)
        if si is None:
            continue
        k, j, n = src_pos[si]
        t0 = src_starts[k]
        t1 = src_starts[k + 1] if k + 1 < len(src_starts) else t0
        out[seg_id] = round(1000 * (t0 + (t1 - t0) * j / max(n, 1)))
    # keep only a monotone chain; misaligned repeats would otherwise run backwards
    chain, last = {}, -1
    for seg_id in sorted(out):
        if out[seg_id] >= last:
            chain[seg_id] = out[seg_id]
            last = out[seg_id]
    return chain


def cmd_timed(args) -> None:
    texts = corpus_texts([Path(d) for d in args.texts])
    openings = {_opening(t): did for did, t in texts.items()}
    all_pages = pages()
    for tf in sorted(PAGES.glob("*-timing.json")):
        page_id = tf.name[: -len("-timing.json")]
        page = all_pages.get(page_id)
        if not page:
            continue
        starts = json.loads(tf.read_text(encoding="utf-8"))
        # Timings index the rendered segments of the whole page, instructions
        # included; only use pages where that count is unambiguous.
        rendered = [s for d in page.get("duas") or [] for s in d.get("segments") or []]
        arabic = [s for s in rendered if s.get("type") != "instruction" and (s.get("arabic") or "").strip()]
        bl = [b for b in blocks(page) if b["segments"]]
        if len(bl) != 1 or len(starts) not in (len(rendered), len(arabic)) or not bl[0]["audio"]:
            print(f"  {page_id}: {len(starts)} times for {len(rendered)} segments / {len(bl)} du'as, skipped")
            continue
        if len(starts) == len(rendered):
            keep = {id(s) for s in arabic}
            starts = [t for s, t in zip(rendered, starts) if id(s) in keep]
        b = bl[0]
        did = same_as(b["segments"], texts, openings)
        if not did:
            print(f"  {page_id}: text not in the corpus yet (run texts first), skipped")
            continue
        timing = carry_over(b["segments"], [float(t) for t in starts], texts[did])
        coverage = len(timing) / len(texts[did])
        if coverage < args.min_coverage:
            print(f"  {page_id} -> {did}: only {coverage:.0%} of lines placed, skipped")
            continue
        audio_id = f"duasorg-{page_id}"
        out = TIMED / did
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "audio_id": audio_id, "dua_id": did,
            "reciter": args.reciters.get(page_id, reciter_from_url(b["audio"])),
            "duration_ms": None, "slide_start_ms": timing,
            "audio_url": b["audio"], "source": f"{SITE}/{page_id}.html",
            "timing_note": "duas.org human timings, carried over to this line split by word alignment",
        }
        (out / f"{audio_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {page_id} -> {did}: {len(timing)}/{len(texts[did])} lines timed ({meta['reciter']})")


def reciter_from_url(url: str) -> str:
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).stem.lower()
    for key, who in (("ali_fani", "Ali Fani"), ("halwachi", "Abu Thar Al-Halawaji"), ("buhamed", "Mula Ali Boumad"),
                     ("bouahmed", "Mula Ali Boumad"), ("abdulhay", "AbdulHai Qambar"), ("naqvi", "Naqvi")):
        if key in name:
            return who
    return f"duas.org:{name}"


def cmd_audio(args) -> None:
    for meta_path in sorted(TIMED.glob("*/*.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        mp3 = meta_path.with_suffix(".mp3")
        if mp3.exists():
            continue
        if shutil.disk_usage(TIMED).free < args.min_free_gb * 1e9:
            print(f"less than {args.min_free_gb} GB free, stopping")
            return
        url = urllib.parse.quote(m["audio_url"], safe=":/%")
        mp3.write_bytes(_get(url))
        m["duration_ms"] = duration_ms(mp3)
        meta_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {m['dua_id']}: {mp3.stat().st_size / 1e6:.0f} MB, {m['duration_ms'] / 60000:.1f} min")
        time.sleep(1)


def cmd_untimed(args) -> None:
    """Every duas.org recording of a du'a in the corpus, for the offline aligner to label."""
    texts = corpus_texts([Path(d) for d in args.texts])
    openings = {_opening(t): did for did, t in texts.items()}
    timed_urls = {json.loads(p.read_text(encoding="utf-8"))["audio_url"] for p in TIMED.glob("*/*.json")}
    out_root = ROOT / "data" / "untimed" / "duasorg"
    seen = set()
    for page_id, page in pages().items():
        for b in blocks(page):
            url = b["audio"]
            if not url or url in seen or url in timed_urls or len(b["segments"]) < args.min_segments:
                continue
            seen.add(url)
            did = same_as(b["segments"], texts, openings)
            if not did:
                continue
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).stem)
            out = out_root / did
            mp3 = out / f"{name}.mp3"
            if (out / f"{name}.json").exists():
                continue
            if shutil.disk_usage(out_root.parent).free < args.min_free_gb * 1e9:
                print(f"less than {args.min_free_gb} GB free, stopping")
                return
            out.mkdir(parents=True, exist_ok=True)
            try:
                mp3.write_bytes(_get(urllib.parse.quote(url, safe=":/%")))
            except Exception as e:  # noqa: BLE001
                print(f"  {url}: {e}")
                continue
            (out / f"{name}.json").write_text(json.dumps({
                "dua_id": did, "audio": mp3.name, "audio_url": url, "reciter": reciter_from_url(url),
                "duration_s": duration_ms(mp3) / 1000, "source": f"{SITE}/{page_id}.html",
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {did}: {name} ({mp3.stat().st_size / 1e6:.0f} MB)", flush=True)
            time.sleep(1)


def duration_ms(path: Path) -> int:
    import subprocess
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return int(float(r.stdout.strip()) * 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "texts", "lines", "prune", "timed", "audio", "untimed"])
    ap.add_argument("--out", default=str(DUAS), help="texts: where new texts go")
    ap.add_argument("--texts", nargs="*", default=[], help="timed: extra text dirs to match against")
    ap.add_argument("--min-segments", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--min-free-gb", type=float, default=15)
    ap.add_argument("--overlap", type=float, default=0.8, help="prune: containment that counts as a copy")
    args = ap.parse_args()
    args.reciters = {}
    {"download": cmd_download, "texts": cmd_texts, "lines": cmd_lines, "prune": cmd_prune, "timed": cmd_timed, "audio": cmd_audio, "untimed": cmd_untimed}[args.stage](args)


if __name__ == "__main__":
    main()
