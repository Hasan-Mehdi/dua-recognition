#!/usr/bin/env python
"""Word-onset review bundles: export excerpts for a human to check, import the answers.

The word truth (scripts/word_truth.py) is forced-aligned by the same wav2vec2 the
follower listens to, so it can't settle whether the follower is late or the
labels are. A reviewer can. Two kinds of bundle, never mixed:

    diagnostic   deliberately hard cases, 10 per stratum (below): where the
                 labels are most likely wrong. Not a label-quality estimate.
    audit        uniformly random words: an estimate of automatic label quality.

Strata (development recordings only: DuaPlayer train reciters + the testsets'
development groups, scripts/testset_split.py):

    within_line   an ordinary word inside a line (well aligned)
    after_pause   a line's first word after >= 0.3 s without an aligned word
    long_vowel    the longest words per letter (drawn-out vowels)
    repeat_skip   suspected repeats/skips: a > 1 s hole inside a line, or a word
                  over 3x its line's median length per letter (heuristic: the
                  reviewer decides)
    low_level     the quietest excerpts (bottom decile of RMS)
    low_conf      words from lines the aligner scored < -1.5 (rejected by the evals)

    python scripts/review_bundle.py export --kind diagnostic --name diag1
    python scripts/review_bundle.py export --kind audit --name audit1 --n 60
    python scripts/review_bundle.py import data/cache/reliability/rel-20260926/review/diag1
    python scripts/review_bundle.py status data/cache/reliability/rel-20260926/review/diag1

Each bundle folder (gitignored, private audio): excerpts/NNN.wav (16 kHz),
excerpts/NNN.txt (Audacity labels: the candidate word and its neighbours),
annotations.csv (one row per excerpt; fill the reviewed_* / status / spoken
columns) and README.md with the instructions. `import` writes
data/cache/word_review/<audio_id>.json (dua_recognition/provenance.py); only the
reviewed words become human word truth, and only inside their reviewed intervals.
"""
from __future__ import annotations

import os
import argparse
import csv
import datetime
import json
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from dua_recognition import provenance  # noqa: E402
from dua_recognition.corpus import TESTSETS  # noqa: E402

RUN = ROOT / "data" / "cache" / "reliability" / os.environ.get("DUA_REL_RUN", "rel-20260927")
SR = 16000
PRE, POST = 1.5, 1.5  # seconds of audio around the candidate word
STRATA = ["within_line", "after_pause", "long_vowel", "repeat_skip", "low_level", "low_conf"]
EDIT = ["reviewed_start", "start_lo", "start_hi", "reviewed_end", "status", "occurrence", "spoken", "notes"]
FIELDS = ["excerpt", "bundle_kind", "stratum", "audio_id", "dua", "w", "word", "line", "line_text",
          "excerpt_start", "candidate_start", "candidate_end", "auto_score", "word_provenance",
          "line_provenance"] + EDIT


def dev_recordings(ix):
    """(recording, source) for development data: DuaPlayer train + testsets' dev groups."""
    from testset_split import load_split, part_of

    out = []
    for dua in ev.load_all().values():
        for r in ev._load_recordings(dua):
            if not ev._is_test(r.reciter):
                out.append((r, "duaplayer-train"))
    split = load_split()
    for name, path in TESTSETS.items():
        if not path.exists():
            continue
        for dua in ev.load_all().values():
            for r in ev._load_recordings(dua, cache_dir=path, extra=False):
                if part_of(name, r, split) == "dev":
                    out.append((r, name))
    return out


def candidates(ix, recs, rng):
    """Every aligned word with the features the strata need (all words: rejected ones too)."""
    from faster_whisper.audio import decode_audio

    rows = []
    for rec, source in recs:
        f = ev.WORD_TRUTH / f"{rec.audio_id}.json"
        if not f.exists() or rec.dua_id not in ix.dua_ids:
            continue
        raw = json.loads(f.read_text(encoding="utf-8"))
        words = ev.load_word_truth(rec, ix)
        y = decode_audio(str(rec.path), sampling_rate=SR)
        by_line = defaultdict(list)
        for w in words:
            by_line[int(ix.word_segment[w[0]])].append(w)
        for seg, ws in by_line.items():
            per_letter = [(b - a) / max(1, len(ix.words[w].text)) for w, a, b, _ in ws]
            med = float(np.median(per_letter))
            for k, (w, a, b, score) in enumerate(ws):
                prev_end = ws[k - 1][2] if k else None
                hole = k and a - ws[k - 1][2] > 1.0
                seg_audio = y[int(max(0, a - PRE) * SR): int((b + POST) * SR)]
                rows.append({
                    "rec": rec, "source": source, "w": w, "a": a, "b": b, "score": score, "line": seg,
                    "first": k == 0, "gap": (a - prev_end) if prev_end is not None else None,
                    "per_letter": per_letter[k], "hole": bool(hole), "stretched": per_letter[k] > 3 * med,
                    "rms": float(20 * np.log10(np.sqrt(np.mean(seg_audio ** 2)) + 1e-9)) if seg_audio.size else -120,
                    "wprov": provenance.word_provenance(raw), "y": y,
                })
    return rows


def pick(rows, rng, per: int = 10) -> tuple[list[tuple[str, dict]], dict]:
    """10 per stratum, no word twice, spread over recordings; shortfalls reported, never padded."""
    good = [r for r in rows if r["score"] >= -1.5]
    per_letter_q90 = np.percentile([r["per_letter"] for r in good], 90) if good else np.inf
    rms_q10 = np.percentile([r["rms"] for r in rows], 10) if rows else -np.inf
    pools = {
        "within_line": [r for r in good if not r["first"] and r["gap"] is not None and r["gap"] < 0.15],
        "after_pause": [r for r in good if r["first"] and (r["gap"] is None or r["gap"] >= 0.3)],
        "long_vowel": [r for r in good if r["per_letter"] >= per_letter_q90],
        "repeat_skip": [r for r in rows if r["hole"] or r["stretched"]],
        "low_level": [r for r in rows if r["rms"] <= rms_q10],
        "low_conf": [r for r in rows if r["score"] < -1.5],
    }
    used, out, short = set(), [], {}
    for name in STRATA:
        pool = pools[name][:]
        rng.shuffle(pool)
        # round-robin over recordings so one long recording can't fill a stratum
        by_rec = defaultdict(list)
        for r in pool:
            by_rec[r["rec"].audio_id].append(r)
        chosen, keys = [], sorted(by_rec)
        while len(chosen) < per and any(by_rec[k] for k in keys):
            for k in keys:
                while by_rec[k]:
                    r = by_rec[k].pop()
                    key = (r["rec"].audio_id, r["w"], round(r["a"], 2))
                    if key not in used:
                        used.add(key)
                        chosen.append(r)
                        break
                if len(chosen) >= per:
                    break
        out += [(name, r) for r in chosen]
        if len(chosen) < per:
            short[name] = f"{len(chosen)}/{per} (pool {len(pools[name])})"
    return out, short


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def export(kind: str, name: str, n: int, seed: int) -> None:
    ix = ev.CorpusIndex(ev.load_all())
    rng = random.Random(seed)
    rows = candidates(ix, dev_recordings(ix), rng)
    if kind == "diagnostic":
        chosen, short = pick(rows, rng)
    else:
        chosen, short = [("random", r) for r in rng.sample(rows, min(n, len(rows)))], {}
    out = RUN / "review" / name
    if out.exists():
        sys.exit(f"{out} exists: bundles are never overwritten")
    (out / "excerpts").mkdir(parents=True)
    with (out / "annotations.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, FIELDS)
        wr.writeheader()
        for i, (stratum, r) in enumerate(chosen, 1):
            rec, w = r["rec"], r["w"]
            a0 = max(0.0, r["a"] - PRE)
            write_wav(out / "excerpts" / f"{i:03d}.wav", r["y"][int(a0 * SR): int((r["b"] + POST) * SR)])
            # Audacity labels: the candidate and its aligned neighbours, relative to the excerpt
            truth = ev.load_word_truth(rec, ix)
            near = [t for t in truth if t[2] > a0 and t[1] < r["b"] + POST]
            (out / "excerpts" / f"{i:03d}.txt").write_text(
                "".join(f"{max(0, t[1] - a0):.3f}\t{t[2] - a0:.3f}\t{'>> ' if t[0] == w else ''}{ix.words[t[0]].text}\n"
                        for t in near), encoding="utf-8")
            seg = next(s for s in ix.duas[ix.word_dua[w]].segments if s.id == r["line"])
            wr.writerow({"excerpt": f"{i:03d}", "bundle_kind": kind, "stratum": stratum, "audio_id": rec.audio_id,
                         "dua": rec.dua_id, "w": w - ix.dua_word_span[ix.word_dua[w]][0],
                         "word": ix.words[w].text, "line": r["line"], "line_text": seg.arabic,
                         "excerpt_start": f"{a0:.3f}", "candidate_start": f"{r['a']:.3f}",
                         "candidate_end": f"{r['b']:.3f}", "auto_score": f"{r['score']:.2f}",
                         "word_provenance": r["wprov"].get("kind"),
                         "line_provenance": rec.line_provenance.get("kind"), **{k: "" for k in EDIT}})
    manifest = {"kind": kind, "name": name, "seed": seed, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "excerpts": len(chosen), "per_stratum": {s: sum(1 for x, _ in chosen if x == s) for s in
                                                         sorted({x for x, _ in chosen})},
                "shortfall": short, "candidate_words": len(rows),
                "recordings": sorted({r["rec"].audio_id for _, r in chosen})}
    (out / "bundle.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "README.md").write_text(README.format(kind=kind, n=len(chosen)), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


README = """# Word-onset review ({kind}, {n} excerpts)

Private audio: keep this folder local; never upload or commit it.

For each row of `annotations.csv`, open `excerpts/NNN.wav` in Audacity
(File > Import > Labels: `excerpts/NNN.txt` shows the machine's word boundaries;
the target word is marked `>>`). Times in the CSV are seconds in the *whole
recording*: excerpt time + `excerpt_start`.

Fill in:

- `status`: ok (the target word is spoken where the label says, give or take) /
  uncertain (spoken, but the onset can't be heard precisely) / substituted (a
  different word was said in its place) / omitted (not said) / repeated (said
  more than once; review the occurrence given in `occurrence`, 0 = first) /
  inserted (the excerpt's speech is not in the text).
- `reviewed_start`: where the word's first sound starts (recording seconds).
  For a breath before the word, the word starts after the breath.
- `start_lo`, `start_hi`: the range you'd accept as the onset if it can't be
  pinned down (leave blank if exact within ~50 ms).
- `reviewed_end` (optional), `spoken` (what was actually said, if not the text),
  `notes`.

Then: `python scripts/review_bundle.py import <this folder>`.
"""


def f_or_none(x: str) -> float | None:
    return float(x) if x and x.strip() else None


def import_bundle(folder: Path, reviewer: str) -> None:
    rows = list(csv.DictReader((folder / "annotations.csv").open(encoding="utf-8-sig")))
    done = [r for r in rows if r["status"].strip()]
    bad = [r["excerpt"] for r in done if r["status"].strip() not in provenance.WORD_STATUS]
    if bad:
        sys.exit(f"unknown status in excerpts {bad}; use one of {provenance.WORD_STATUS}")
    by_rec = defaultdict(list)
    for r in done:
        by_rec[r["audio_id"]].append(r)
    provenance.WORD_REVIEW.mkdir(parents=True, exist_ok=True)
    for aid, rs in by_rec.items():
        review = provenance.load_review(aid) or {"audio_id": aid, "words": [], "intervals": [], "sources": []}
        for r in rs:
            st = r["status"].strip()
            s = f_or_none(r["reviewed_start"])
            if st in provenance.SCORABLE and s is None:
                s = float(r["candidate_start"]) if st in ("ok", "uncertain") else None
            lo, hi = f_or_none(r["start_lo"]), f_or_none(r["start_hi"])
            word = {"w": int(r["w"]), "status": st, "occurrence": int(r["occurrence"] or 0),
                    "spoken": r["spoken"], "stratum": r["stratum"], "bundle": folder.name,
                    "auto_start": float(r["candidate_start"])}
            if s is not None:
                word.update(start=s, start_lo=lo if lo is not None else s, start_hi=hi if hi is not None else s)
                if f_or_none(r["reviewed_end"]) is not None:
                    word["end"] = float(r["reviewed_end"])
                # Only this word's own onset range counts as reviewed: its neighbours stay automatic.
                review["intervals"].append([round(min(word["start_lo"], s) - 0.01, 3),
                                            round(max(word["start_hi"], s) + 0.01, 3)])
            review["words"] = [x for x in review["words"] if not (x["w"] == word["w"] and x.get("occurrence", 0)
                                                                     == word["occurrence"])] + [word]
        review["sources"] = sorted(set(review.get("sources", [])) | {folder.name})
        review["reviewer"] = reviewer
        review["updated"] = datetime.date.today().isoformat()
        (provenance.WORD_REVIEW / f"{aid}.json").write_text(json.dumps(review, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
    print(f"{len(done)}/{len(rows)} excerpts reviewed, {len(by_rec)} recordings -> {provenance.WORD_REVIEW}")


def status(folder: Path) -> None:
    rows = list(csv.DictReader((folder / "annotations.csv").open(encoding="utf-8-sig")))
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["stratum"]][0] += 1
        by[r["stratum"]][1] += bool(r["status"].strip())
    for k, (n, d) in sorted(by.items()):
        print(f"{k:12s} {d}/{n} reviewed")
    offs = [float(r["reviewed_start"]) - float(r["candidate_start"]) for r in rows
            if r["status"].strip() in ("ok", "uncertain") and r["reviewed_start"].strip()]
    if offs:
        print(f"auto onset minus reviewed: median {-np.median(offs):+.3f} s, |err| p90 "
              f"{np.percentile(np.abs(offs), 90):.3f} s (n={len(offs)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--kind", choices=["diagnostic", "audit"], required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--n", type=int, default=60, help="audit: number of random words")
    e.add_argument("--seed", type=int, default=20260926)
    i = sub.add_parser("import")
    i.add_argument("folder", type=Path)
    i.add_argument("--reviewer", default="hasan")
    s = sub.add_parser("status")
    s.add_argument("folder", type=Path)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.kind, args.name, args.n, args.seed)
    elif args.cmd == "import":
        import_bundle(args.folder, args.reviewer)
    else:
        status(args.folder)


if __name__ == "__main__":
    main()
