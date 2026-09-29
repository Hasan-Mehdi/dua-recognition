#!/usr/bin/env python
"""Turn an ordinary person's recording into a labelled test recording (auto labels, to review).

The DuaPlayer test reciters are professionals; the people who will use the app
aren't. This takes one recording (any format ffmpeg reads, phone video
included) and writes it in the DuaPlayer shape into a held-out set
(corpus.TESTSETS), so every evaluation script reads it with `--source <set>`:

    data/<set dir>/<dua_id>/<audio_id>.mp3     the audio (16 kHz mono)
    data/<set dir>/<dua_id>/<audio_id>.json    DuaPlayer meta: line start times ("auto": true)
    data/<set dir>/<dua_id>/<audio_id>.srt     the line texts on those times, to review
    data/cache/word_truth/<audio_id>.json      forced-aligned word timings (as scripts/word_truth.py)
    data/cache/windows/<tag>/<audio_id>_w6_h1.jsonl   the teacher windows (reused by evaluate.py)

Steps: teacher windows (6 s, 1 s hop; the fine-tuned turbo) -> which du'a (the
tracker over the whole corpus, unless --dua is given) -> line starts from the
offline smoother -> word timings from the wav2vec2 forced aligner, line by
line. A second teacher (stock large-v3) labels the same audio; the share of
seconds both put on the same line is "teacher_agreement", and a recording
under --min-agreement goes to the review queue instead of the silver tier.

Review: open the .srt in Subtitle Edit (waveform view) next to the .mp3, fix
the cue start times, save, then

    python scripts/user_label.py --import-srt data/testsets/amateur/<dua>/<id>.srt

which writes the times back, sets "auto": false (gold) and redoes the word
timings. The audio is a person's voice: every TESTSETS dir is gitignored.

    python scripts/user_label.py my_test.mp4 --reciter hasan
    python scripts/user_label.py take.m4a --dua dua-tawassul --reciter guest3 --condition headset
    python scripts/user_label.py clip.webm --set amateur --reciter yt07 --source-url URL --condition phone
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402
from align_offline import placed_mask, starts_from  # noqa: E402
from transcribe_windows import cache_path  # noqa: E402

from dua_recognition.align import CorpusIndex  # noqa: E402
from dua_recognition.asr import _looks_hallucinated, speech_in_tail, transcribe_batch  # noqa: E402
from dua_recognition.corpus import TESTSETS  # noqa: E402
from dua_recognition.labels import agreement, from_srt, meta_starts, set_starts, to_srt  # noqa: E402
from dua_recognition.offline import align_recording  # noqa: E402
from dua_recognition.tracker import Tracker  # noqa: E402

SR = 16000
TEACHER = ("models/whisper-turbo-dua-ct2", "whisper-turbo-dua")
SECOND = ("large-v3", "large-v3")
PREFIX = {"user": "user", "amateur": "am", "majlis": "mj"}


def identify(ix: CorpusIndex, costs) -> tuple[str | None, float]:
    """The du'a the live tracker is locked on for most of the recording."""
    tr = Tracker(ix)
    seen = Counter()
    for c in costs:
        p = tr.update_costs(c, 1.0)
        if p.dua is not None:
            seen[p.dua] += 1
    if not seen:
        return None, 0.0
    dua, n = seen.most_common(1)[0]
    return dua, n / len(costs)


def teacher_rows(y, audio_id: str, model: str, tag: str, device: str | None) -> list[tuple[float, str]]:
    """6 s / 1 s windows of one teacher, cached like transcribe_windows.py."""
    from dua_recognition.asr import load_model

    win = cache_path(tag, audio_id, 6.0, 1.0)
    if win.exists():
        rows = [json.loads(line) for line in win.read_text(encoding="utf-8").splitlines() if line]
    else:
        model = str(ROOT / model) if (ROOT / model).exists() else model
        load_model(model, device)
        times = [float(k + 1) for k in range(int(y.size / SR))]
        texts = []
        for b in range(0, len(times), 16):
            chunk = times[b : b + 16]
            texts += transcribe_batch([y[int(max(0.0, t - 6.0) * SR) : int(t * SR)] for t in chunk], model=model)
        rows = [{"t": t, "text": x, "speech": speech_in_tail(y[int(max(0.0, t - 6.0) * SR) : int(t * SR)])}
                for t, x in zip(times, texts)]
        win.parent.mkdir(parents=True, exist_ok=True)
        win.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return [(r["t"], "" if _looks_hallucinated(r["text"]) else r["text"]) for r in rows]


def line_starts(dua, rows) -> tuple[list[tuple[float, int]], float]:
    """(line starts, share of voiced windows placed) from the offline smoother on one du'a."""
    ix = CorpusIndex({dua.id: dua})
    costs = [ix.word_costs(x) if x else None for _, x in rows]
    segs = align_recording(ix, costs, 1.0)
    placed = placed_mask(ix, rows, costs)
    voiced = [p for (_, x), p in zip(rows, placed) if x]
    return starts_from([t for t, _ in rows], segs, placed), sum(voiced) / max(1, len(voiced))


def word_timings(full: CorpusIndex, dua_id: str, audio: Path, starts, end_s: float,
                 line_prov: dict | None = None) -> list[list]:
    """Forced-aligned words. Always automatic labels: a reviewed line SRT promotes the
    *line* truth only (dua_recognition/provenance.py); words need their own review."""
    from faster_whisper.audio import decode_audio
    from word_truth import aligner_provenance, align_words, load_aligner

    y = decode_audio(str(audio), sampling_rate=SR)
    words = align_words(load_aligner(), full, y, starts, end_s, full.dua_ids.index(dua_id))
    prov = aligner_provenance(full.duas[full.dua_ids.index(dua_id)], line_prov or {"kind": "unknown"})
    ev.WORD_TRUTH.mkdir(parents=True, exist_ok=True)
    (ev.WORD_TRUTH / f"{audio.stem}.json").write_text(json.dumps({"dua": dua_id, "words": words, "provenance": prov}),
                                                    encoding="utf-8")
    return words


def import_srt(srt: Path) -> None:
    """A reviewed .srt back into its meta: human line times, "auto": false (gold)."""
    meta_path = srt.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    duas = ev.load_all()
    dua = duas[meta["dua_id"]]
    n = len(dua.segments)
    old, _ = meta_starts(meta, n)
    starts, end = from_srt(srt.read_text(encoding="utf-8-sig"))
    if not starts:
        sys.exit(f"no cues with a '<line>. text' body in {srt}")
    bad = [s for _, s in starts if not 1 <= s <= n]
    if bad:
        sys.exit(f"line ids outside 1..{n}: {bad}")
    meta = set_starts(meta, starts, end, n)
    today = datetime.date.today().isoformat()
    line_prov = {"kind": "human_reviewed", "reviewed": today, "from_auto": meta.get("teacher"), "tool": "srt"}
    meta.update(auto=False, needs_review=False, reviewed=today,
                auto_vs_reviewed=round(agreement(old, starts, end), 3))
    # Line truth only: the words re-aligned below stay automatic (provenance.py).
    meta.setdefault("provenance", {})["line"] = line_prov
    meta["provenance"]["word"] = {"kind": "auto", "review": "none"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{meta['audio_id']}: {len(starts)} lines, gold now; auto labels matched the review on "
          f"{meta['auto_vs_reviewed']:.0%} of seconds")
    words = word_timings(CorpusIndex(duas), dua.id, srt.with_suffix(".mp3"), starts, end, line_prov)
    print(f"  word timings redone: {len(words)} words (automatic; word review is separate)")


def second_opinion(name: str, tag: str, min_agreement: float) -> None:
    """teacher_agreement for every auto-labelled recording of a set labelled elsewhere
    (segment_streams.py --all-test), from the second teacher's cached windows
    (transcribe_windows.py --source NAME --model large-v3 --tag large-v3)."""
    from dua_recognition.corpus import load_recordings

    duas = ev.load_all()
    for dua in duas.values():
        for rec in load_recordings(dua, cache_dir=TESTSETS[name], extra=False):
            meta_path = rec.path.with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows = ev.load_rows(tag, rec, 6.0, 1.0)
            if not meta.get("auto") or not rows:
                continue
            starts2, _ = line_starts(dua, rows)
            agree = round(agreement(rec.starts, starts2, rec.end_s), 3)
            meta.update(second_teacher=tag, teacher_agreement=agree, needs_review=agree < min_agreement)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {rec.audio_id:40s} {dua.id:28s} agreement {agree:.0%}"
                  + ("  -> review queue" if meta["needs_review"] else ""), flush=True)


def sanity(tag: str, second_tag: str, min_agreement: float) -> None:
    """The labeller on the human-timed DuaPlayer test recordings: how often its lines match the
    human ones, and whether the two-teacher gate keeps the good labels and sends the bad to review."""
    from dua_recognition.corpus import load_recordings
    from dua_recognition.splits import LINE_LABELS_UNRELIABLE, is_test

    rows_out = []
    for dua in ev.load_all().values():
        if dua.id in LINE_LABELS_UNRELIABLE:
            continue
        for rec in load_recordings(dua):
            r1, r2 = ev.load_rows(tag, rec, 6.0, 1.0), ev.load_rows(second_tag, rec, 6.0, 1.0)
            if not is_test(rec.reciter) or not r1 or not r2:
                continue
            s1, _ = line_starts(dua, r1)
            s2, _ = line_starts(dua, r2)
            h1, h2 = agreement(rec.starts, s1, rec.end_s), agreement(rec.starts, s2, rec.end_s)
            both = agreement(s1, s2, rec.end_s)
            rows_out.append((rec, h1, h2, both))
            print(f"  {dua.id:28s} {rec.reciter[:22]:22s} vs human: {tag} {h1:.1%}  {second_tag} {h2:.1%}  "
                  f"teachers agree {both:.1%}", flush=True)
    if not rows_out:
        sys.exit("no test recordings with both teachers' windows")
    w = [r.end_s for r, *_ in rows_out]

    def mean(xs, ws):
        return sum(x * v for x, v in zip(xs, ws)) / max(sum(ws), 1e-9)

    print(f"\n{len(rows_out)} test recordings, {sum(w) / 3600:.1f} h (time-weighted): {tag} vs human "
          f"{mean([h for _, h, _, _ in rows_out], w):.1%}, {second_tag} vs human {mean([h for _, _, h, _ in rows_out], w):.1%}")
    for name, keep in (("silver (agree >= gate)", True), ("review (agree < gate)", False)):
        sel = [(r, h) for r, h, _, b in rows_out if (b >= min_agreement) == keep]
        if sel:
            print(f"  {name}: {len(sel)} recordings, {tag} vs human {mean([h for _, h in sel], [r.end_s for r, _ in sel]):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, nargs="?")
    ap.add_argument("--sanity", action="store_true",
                    help="score the two-teacher labeller against the DuaPlayer test recordings' human timings")
    ap.add_argument("--import-srt", type=Path, metavar="SRT", help="read a reviewed .srt back (-> gold)")
    ap.add_argument("--agree", choices=list(TESTSETS), metavar="SET",
                    help="only compute teacher_agreement for a set's auto labels from --second-tag windows")
    ap.add_argument("--set", choices=["user", "amateur"], default="user", help="which held-out set (corpus.TESTSETS)")
    ap.add_argument("--dua", help="du'a id, if known (skips identification)")
    ap.add_argument("--reciter", default="user", help="a name or code for the person (stays local)")
    ap.add_argument("--condition", default="phone", choices=["studio", "phone", "headset", "room", "majlis"])
    ap.add_argument("--source-url", default="", help="where the recording came from (stays local)")
    ap.add_argument("--venue", default="", help="uploader channel id or place (caps and leak checks group by it)")
    ap.add_argument("--model", default=TEACHER[0], help="teacher ASR")
    ap.add_argument("--tag", default=TEACHER[1], help="window-cache tag of --model")
    ap.add_argument("--second", default=SECOND[0], help="second-opinion ASR ('' to skip)")
    ap.add_argument("--second-tag", default=SECOND[1])
    ap.add_argument("--min-agreement", type=float, default=0.85)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.import_srt:
        import_srt(args.import_srt)
        return
    if args.agree:
        second_opinion(args.agree, args.second_tag, args.min_agreement)
        return
    if args.sanity:
        sanity(args.tag, args.second_tag, args.min_agreement)
        return
    if not args.audio:
        ap.error("an audio file, or --import-srt")

    from faster_whisper.audio import decode_audio

    out = TESTSETS[args.set]
    stem = re.sub(r"[^A-Za-z0-9]+", "-", args.audio.stem).strip("-").lower()
    audio_id = f"{PREFIX[args.set]}-{re.sub(r'[^A-Za-z0-9]+', '', args.reciter).lower()}-{stem}"
    tmp = out / "_incoming" / f"{audio_id}.mp3"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.audio), "-vn", "-ac", "1", "-ar", str(SR),
                    "-b:a", "96k", str(tmp)], check=True)
    y = decode_audio(str(tmp), sampling_rate=SR)
    dur = y.size / SR

    rows = teacher_rows(y, audio_id, args.model, args.tag, args.device)
    duas = ev.load_all()
    full = CorpusIndex(duas)
    if args.dua:
        dua_id, share = args.dua, 1.0
    else:
        dua_id, share = identify(full, ev.LazyCosts(full, [x for _, x in rows]))
        if dua_id is None:
            sys.exit("could not tell which du'a this is; pass --dua")
    dua = duas[dua_id]
    print(f"{audio_id}: {dur:.0f} s, du'a {dua_id} ({dua.name_en}){'' if args.dua else f', locked {share:.0%} of the time'}")

    starts, coverage = line_starts(dua, rows)
    print(f"  {len(starts)} line starts; teacher placed {coverage:.0%} of voiced windows")
    if not starts:
        sys.exit("no lines placed: wrong du'a, or the recording is too hard for the teacher")
    agree = None
    if args.second:
        starts2, _ = line_starts(dua, teacher_rows(y, audio_id, args.second, args.second_tag, args.device))
        agree = round(agreement(starts, starts2, dur), 3)
        print(f"  second teacher ({args.second_tag}) on the same line {agree:.0%} of seconds")

    d = out / dua_id
    d.mkdir(parents=True, exist_ok=True)
    mp3 = d / f"{audio_id}.mp3"
    tmp.replace(mp3)
    meta = {"audio_id": audio_id, "dua_id": dua_id, "reciter": f"{args.set}:{args.reciter}",
            "duration_ms": int(dur * 1000), "slide_start_ms": {str(s): int(t * 1000) for t, s in starts},
            "auto": True, "condition": args.condition, "venue": args.venue, "source_file": args.audio.name,
            "source_url": args.source_url, "teacher": args.tag, "placed_fraction": round(coverage, 3),
            "second_teacher": args.second_tag if args.second else "", "teacher_agreement": agree,
            "needs_review": agree is not None and agree < args.min_agreement}
    (d / f"{audio_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / f"{audio_id}.srt").write_text(to_srt(starts, dur, {s.id: s.arabic for s in dua.segments}), encoding="utf-8")
    words = word_timings(full, dua_id, mp3, starts, dur, {"kind": "auto", "model": args.tag})
    good = [w for w in words if w[3] >= -1.5]
    print(f"  {len(words)} words aligned, {len(good)} with a line score >= -1.5")
    print(f"  wrote {d / audio_id}.{{mp3,json,srt}} and word truth"
          + ("; teachers disagree -> review queue" if meta["needs_review"] else ""))


if __name__ == "__main__":
    main()
