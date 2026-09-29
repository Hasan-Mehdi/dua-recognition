#!/usr/bin/env python
"""How much of the supplied audio each evaluation actually scores, and why the rest is out.

A recognizer that helps choose its own test material can hide its failures:
teacher disagreement sends audio to review, and word truth under a line-score
threshold is dropped from word scores. This reports, per source and split:
recordings/hours supplied, labelled, and scored (lines, words, line transitions),
exclusions by reason, and the provenance of the line and word labels.

    python scripts/label_coverage.py            # table
    python scripts/label_coverage.py --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from dua_recognition import provenance  # noqa: E402
from dua_recognition.corpus import TESTSETS  # noqa: E402
from dua_recognition.splits import LINE_LABELS_UNRELIABLE  # noqa: E402

MIN_SCORE = -1.5  # follow_eval / word_eval keep word truth from lines scoring at least this
CTC = ROOT / "data" / "cache" / "ctc" / "wav2vec2-quran-dua"


def groups():
    from testset_split import load_split, part_of

    split = load_split()
    duas = ev.load_all()
    for dua in duas.values():
        for r in ev._load_recordings(dua, extra=False):
            yield ("duaplayer", "test (regression)" if ev._is_test(r.reciter) else "train (dev)"), r
        for name, path in TESTSETS.items():
            if path.exists():
                for r in ev._load_recordings(dua, cache_dir=path, extra=False):
                    yield (name, part_of(name, r, split) or "unassigned"), r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    ix = ev.CorpusIndex(ev.load_all())
    agg = defaultdict(lambda: defaultdict(float))
    excl = defaultdict(Counter)
    prov_line, prov_word = defaultdict(Counter), defaultdict(Counter)
    for key, r in groups():
        a = agg[key]
        a["recordings"] += 1
        a["supplied_h"] += r.duration_s / 3600
        a["labelled_h"] += r.end_s / 3600
        prov_line[key][r.line_provenance.get("kind", "unknown")] += 1
        if r.tier == "review":
            excl[key]["line labels in the review queue (teachers disagree)"] += 1
            continue
        if r.dua_id in LINE_LABELS_UNRELIABLE:
            excl[key]["line labels unreliable (text re-split)"] += 1
        else:
            a["line_scored_h"] += r.end_s / 3600
            a["transitions"] += max(0, len(r.starts) - 1)
        truth = ev.load_word_truth(r, ix)
        if truth is None:
            excl[key]["no word truth"] += 1
            continue
        raw = json.loads((ev.WORD_TRUTH / f"{r.audio_id}.json").read_text(encoding="utf-8"))
        prov_word[key][provenance.word_provenance(raw)["kind"] + (
            "" if raw.get("provenance") else " (legacy file, revision unknown)")] += 1
        review = provenance.load_review(r.audio_id)
        if review:
            a["human_words"] += len(provenance.human_words(review, 0))
        a["words_aligned"] += len(truth)
        good = [w for w in truth if w[3] >= MIN_SCORE]
        a["words_scored"] += len(good)
        excl[key]["words from lines scoring < -1.5 (dropped from word scores)"] += len(truth) - len(good)
        if good:
            a["word_scored_h"] += (min(r.end_s, good[-1][2] + 2.0) - good[0][1]) / 3600
        expected = sum(1 for w in range(*ix.dua_word_span[ix.dua_ids.index(r.dua_id)])
                       if any(s <= r.end_s for s, sid in r.starts if sid == ix.word_segment[w]))
        excl[key]["words the aligner did not place"] += max(0, expected - len(truth))
        if not (CTC / f"{r.audio_id}_w3_h0.2.npz").exists():
            excl[key]["no CTC dump (not in follower scores)"] += 1
    out = {f"{k[0]} / {k[1]}": {"totals": {m: round(v, 3) for m, v in agg[k].items()},
                                "exclusions": dict(excl[k]), "line_provenance": dict(prov_line[k]),
                                "word_provenance": dict(prov_word[k])} for k in sorted(agg)}
    for name, d in out.items():
        t = d["totals"]
        print(f"\n## {name}\n{int(t.get('recordings', 0))} recordings; supplied {t.get('supplied_h', 0):.2f} h, "
              f"labelled {t.get('labelled_h', 0):.2f} h, line-scored {t.get('line_scored_h', 0):.2f} h "
              f"({int(t.get('transitions', 0))} transitions), word-scored {t.get('word_scored_h', 0):.2f} h "
              f"({int(t.get('words_scored', 0))} of {int(t.get('words_aligned', 0))} aligned words; "
              f"{int(t.get('human_words', 0))} human-reviewed)")
        for k, v in d["exclusions"].items():
            print(f"  excluded: {k}: {v}")
        print(f"  line labels: {d['line_provenance']}; word labels: {d['word_provenance']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
