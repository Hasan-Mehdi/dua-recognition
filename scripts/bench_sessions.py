#!/usr/bin/env python
"""The scenario bench on Hasan's own debug sessions, as he recorded them (data/sessions).

bench.py's user lane holds 7 sessions (to 2026-10-01), cut into its scenarios. This takes every
session whose reading the teacher can time: each run of at least `MIN_LINES` consecutive lines it
times word by word is one item, played as recorded (his pauses, his phone, his room), and every gap
of 2 s or more between his words is a pause the highlight should hold still through. Items, model
outputs and results live in data/testbed/sessions, so the bench's own grid and baselines are
untouched; every bench.py command runs here unchanged.

    python scripts/bench_sessions.py build      # GPU: reference transcripts, the teacher's alignment, items
    python scripts/bench_sessions.py asr --model models/whisper-base-ph-kd2-ct2 --ctc-model models/ctc-stream-base-v1
    python scripts/bench_sessions.py asr ... --ctc-model ctc-student-base-v6u   # the window student on real 2 s windows
    python scripts/bench_sessions.py truth      # the perfect ear's rows (score --asr truth), after one asr pass
    python scripts/bench_sessions.py score --name x --split all --display stream --same-text 8 ...

Truth: session_eval.truth, the teacher (wav2vec2-quran-dua-voices) forced along the lines that turbo's
transcripts say were read, against today's texts (cached per texts and du'a in data/cache/session_truth). The
du'a: of those the page showed or the transcripts point to that turbo's windows place within 15 points
of the best, the one whose reading the teacher times most of in runs of whole lines.
Automatic, not human, and it flatters the letter model (the student learned from this teacher).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import bench  # noqa: E402

# Everything under its own root (set at import, so the scorer's worker processes see it too).
bench.BENCH = bench.BENCH / "sessions"
MIN_LINES = 4  # (bench.py's sources: 6; his Ayat al-Kursi lines are short)
PAUSE_S = 2.0
TRUTH_MODEL = "wav2vec2-quran-dua-voices"
V6U = "ctc-student-base-v6u"  # the window student on real 2 s windows (no silence padded to its 3 s)


def _texts_hash(duas) -> str:
    body = json.dumps({k: [s.arabic for s in d.segments] for k, d in sorted(duas.items())}, ensure_ascii=False)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]


def _candidates(log: dict, texts: list[str], index) -> list[str]:
    """Du'as the page showed (or was told to follow) and the four the transcripts point to most. A window
    votes for every text that reads it equally well: Ayat al-Kursi is two long lines of an Eid al-Mubahila
    text and twelve short ones of Namaz-e-Wahshat, and only the short lines make runs to score."""
    from collections import Counter

    from dua_recognition.align import encode

    shown = [e["dua"] for e in log.get("events", []) if e.get("type") == "dua" and e.get("dua")]
    votes = Counter()
    for t in texts:
        if not t or encode(t).size < 8:
            continue
        c = index.word_costs(t)
        if c is not None:
            for d in {int(index.word_dua[w]) for w in np.flatnonzero(c <= float(c.min()) + 1e-6)}:
                votes[index.dua_ids[d]] += 1
    out = ([log["chosen"]] if log.get("chosen") else []) + [d for d, _ in Counter(shown).most_common(3)]
    out += [d for d, _ in votes.most_common(4)]
    return [d for k, d in enumerate(out) if d in index.dua_ids and d not in out[:k]]


def cmd_build() -> None:
    import session_eval as se
    from session_report import REF_MODEL, SESSIONS, SR, clock_ratio, read_session, reference, reference_lines

    from dua_recognition.align import CorpusIndex
    from dua_recognition.corpus import load_all

    duas = load_all()
    index = CorpusIndex(duas)
    se.TRUTH_MODEL = TRUTH_MODEL
    cache = se.CACHE / f"texts-{_texts_hash(duas)}"  # today's texts (word ids shift with them), per du'a
    # session_eval.truth loads the teacher for every session it times, and the old copies weren't
    # freed (11 GB after 25 sessions): one copy for them all.
    import dua_recognition.ctc as dc

    ctc_model, loaded = dc.CtcModel, {}
    dc.CtcModel = lambda name, *a, **k: loaded.get(name) or loaded.setdefault(name, ctc_model(name, *a, **k))
    out_dir = bench.BENCH / "items" / "session"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_items, minutes, used = 0, 0.0, 0
    for path in sorted(SESSIONS.glob("*.wav")):
        sid = path.stem
        try:
            audio, log = read_session(path)
        except ValueError as e:
            print(f"{sid}: {e}")
            continue
        if len(audio) / SR < 20:
            print(f"{sid}: {len(audio) / SR:.0f} s, too short")
            continue
        if (clock_ratio(log) or 1.0) > 1.5:
            print(f"{sid}: audio clock {clock_ratio(log):.2f}, the microphone captured twice")
            continue
        _, texts = reference(sid, audio, str(ROOT / REF_MODEL))
        placed = sorted(((float(ok.mean()), d, segs, ok) for d in _candidates(log, texts, index)
                         for segs, ok in [reference_lines(duas, d, texts)]), key=lambda x: -x[0])
        if not placed or placed[0][0] < 0.25 or placed[0][3].sum() < 10:
            print(f"{sid}: no du'a the transcripts place ({placed[0][1] if placed else '-'} "
                  f"{placed[0][0] if placed else 0:.0%})")
            continue
        # Of the texts turbo's windows place nearly as well (and the one he chose), the one the teacher
        # times the most reading time of in runs of whole lines.
        best = None
        for share, d, segs, ok in placed:
            if share < placed[0][0] - 0.15 and d != log.get("chosen"):
                continue
            se.CACHE = cache / d
            tr = se.truth(sid, audio, duas, index, d, segs, ok)
            runs = [r for r in bench._line_runs(sorted(tr, key=lambda w: w[1]), index) if len(r) >= MIN_LINES]
            secs = sum(r[-1]["to"] - r[0]["from"] for r in runs)
            if best is None or secs > best[0]:
                best = (secs, share, d, tr, runs)
        _, share, dua, tr, runs = best
        if not runs:
            print(f"{sid}: {dua} ({share:.0%} placed), {len(tr)} words timed, no run of {MIN_LINES} lines")
            continue
        used += 1
        for k, run in enumerate(runs):
            src = {"sid": f"session:{sid}:{dua}:{k}", "lane": "user", "audio": str(path), "dua": dua, "voice": "user:hasan",
                   "lines": run, "tags": {"condition": "phone", "engine": log.get("engine"), "date": sid[:8],
                                          "n_lines": len(run)}}
            p = bench.Prog(src, index)
            p.play(run)
            ws = sorted(p.words, key=lambda w: w[2])
            for a, b in zip(ws, ws[1:]):
                if b[2] - a[3] >= PAUSE_S:
                    p.still.append([round(a[3] + 0.3, 3), round(b[2] - 0.2, 3), "pause"])
            if p.t < 20.0:
                continue
            it = p.item("session", [], 1.0, extra={"seed": 0})
            (out_dir / f"{it['id']}.json").write_text(json.dumps(it, ensure_ascii=False), encoding="utf-8")
            n_items += 1
            minutes += p.t / 60
        print(f"{sid}: {duas[dua].name_en} ({share:.0%} placed), {len(tr)} words timed, "
              f"{len(runs)} runs of {[len(r) for r in runs]} lines", flush=True)
    print(f"{n_items} items, {minutes:.0f} min, from {used} sessions")


def cmd_truth() -> None:
    """Rows of the perfect ear: each second, the true words of the 8 s before it (as data/testbed/asr/truth)."""
    import evaluate as ev

    ix = ev.CorpusIndex(ev.load_all())
    out = bench.BENCH / "asr" / "truth"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((bench.BENCH / "items").glob("*/*.json")):
        it = json.loads(f.read_text(encoding="utf-8"))
        quiet_from = next((p for p in sorted((bench.BENCH / "asr").glob(f"*/{it['id']}.json")) if p.parent.name != "truth"),
                          None)
        if quiet_from is None:
            sys.exit(f"{it['id']}: run an asr pass first (the rows take its quiet)")
        quiet = json.loads(quiet_from.read_text(encoding="utf-8"))["quiet"]
        lo = {d: ix.dua_word_span[ix.dua_ids.index(d)][0] for d in {w[0] for w in it["words"]}}
        ws = [(ix.words[lo[d] + w].text, a, b) for d, w, a, b in it["words"]]
        rows = [[s, " ".join(t for t, a, b in ws if a >= s - 8 and b <= s)] for s in range(1, len(quiet) + 1)]
        (out / f"{it['id']}.json").write_text(json.dumps({"rows": rows, "quiet": quiet}, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"perfect-ear rows in {out}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        return cmd_build()
    if cmd == "truth":
        return cmd_truth()
    if cmd == "asr" and V6U in " ".join(sys.argv):
        import dua_recognition.ctc_student as cs

        load = cs.load_ctc

        def load_ctc(name):
            if Path(name).name != V6U:
                return load(name)
            m = cs.StudentCtc(Path(name).with_name("ctc-student-base-v6"))
            m.window_s = 2.0  # real 2 s windows, as the page's -w2 export runs them
            return m

        cs.load_ctc = load_ctc
    bench.main()


if __name__ == "__main__":
    main()
