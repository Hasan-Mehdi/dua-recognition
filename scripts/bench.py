#!/usr/bin/env python
"""One test bench for the whole app: every way people read, every place they read in.

Until now each problem got its own benchmark (pauses, repeats, jumps), all cut from the same
16 studio recordings by 6 reciters. This builds one grid instead and scores the phone's
display (Whisper + tracker for the du'a and line, the CTC word follower for the highlight)
on every cell of it, in terms a reader would notice: is the highlight on the line I'm
reading, does it jump where I'm not, does it get lost, does it follow me when I go back,
does it stay put while I talk, does it find the du'a, does it stay quiet for a du'a it
doesn't know.

Voices (lanes):
    studio   DuaPlayer test reciters (6 voices, human line times)
    majlis   du'a nights streamed from other centres (crowd, PA echo; data/testsets/majlis)
    harvest  uploaders the harvest keeps out of all training (harvest_label.py export's
             val side: sha1(uploader) % 100 < 3); labels are the CTC teacher's forced
             alignment, lines it placed confidently only
    mafatih  the same uploaders reading the texts added from Mafatih (mafatih_corpus.py:
             Sabah, 'Adeelah, 'Asharat, Munajat 2-15), in the reading scenarios only;
             `sources --lane mafatih` adds them without touching the rest of the grid

Scenarios: how they read (flow, start mid-du'a, pauses, talking in between, salawat in
between, repeating a line, going back 1-3 lines, skipping ahead, jumping around, stumbling
and restarting a line, switching du'a, a du'a not in the corpus, slow chant, fast) and
where (room, hall, people talking, fan, another recitation nearby, far from the phone,
overdriven mic, phone-call codec, low-bitrate codec, and a combination).

    python scripts/bench.py sources                  # test sources -> data/testbed/sources.jsonl
    python scripts/bench.py build                    # scenario programs + truth -> items/
    python scripts/bench.py asr                      # (GPU) Whisper rows, quiet, CTC frames per item
    python scripts/bench.py score --name base        # replay the phone's display, the grid
    python scripts/bench.py score --name x --fw "jump_margin=5" --compare base

Every item is a program over source audio (spans, room tone, inserted clips, effects), so
the audio is rebuilt exactly from the program; nothing but the model outputs is cached.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BENCH = ROOT / "data" / "testbed"  # junction to D:\dua-data\testbed (C: is nearly full)
SR = 16000
WINDOW = 6.0  # Whisper window, s (the page's)
ASR_TAG = "whisper-base-syn-v5-ctx8ft"
CTC_TAG = "ctc-student-base-v6"
CTC_WINDOW, CTC_HOP = 2.0, 0.1
MIN_LINES = 6  # a source is a run of at least this many consecutive, fully timed lines
MORE_OOC = 40  # out-of-corpus readings from uploaders outside the held-out side (sources_harvest)
MAFATIH_MIN = 3  # voices per added text in lane mafatih, from other uploaders where the held-out side is short
SALAWAT = "اللهم صل على محمد وآل محمد"

# What "usable" means, per metric (score prints pass/fail against these).
BARS = {
    "on_line": (">=", 0.90),  # share of reading time the highlight is on the line being read
    "jumps_10": ("<=", 0.2),  # from the reader's line to one 2+ away (or another du'a) they aren't on, per 10 min
    "early_10": ("<=", 1.0),  # to the next line before the reader gets there, per 10 min
    "wrong_place": ("<=", 0.03),  # reading time with the highlight 2+ lines away (after the du'a is found)
    "lost_10": ("<=", 0.5),  # episodes of 4+ s off the reader's line, per 10 min
    "event_3s": (">=", 0.80),  # back / skip / jump / repeat / restart / switch followed within 3 s
    "stay": (">=", 0.95),  # share of pause / talk / salawat time the highlight stays put
    "found_10s": (">=", 0.95),  # du'a on screen within 10 s of the first word
    "found_d10s": (">=", 0.95),  # ...of the first words no other text reads (--same-text)
    "ooc_shown": ("<=", 0.05),  # unknown du'a: share of time some du'a is on screen anyway
}


# ---------------------------------------------------------------- audio helpers
def decode(path: str | Path, a: float | None = None, b: float | None = None) -> np.ndarray:
    """16 kHz mono float32 of path[a:b] (ffmpeg seek; whole file without a/b)."""
    cmd = ["ffmpeg", "-v", "error"]
    if a is not None:
        cmd += ["-ss", f"{max(0.0, a):.3f}"]
    if a is not None and b is not None:
        cmd += ["-t", f"{b - max(0.0, a):.3f}"]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def ffmpeg_filter(y: np.ndarray, af: str) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", af,
                        "-f", "f32le", "-"], input=y.astype(np.float32).tobytes(), capture_output=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def ffmpeg_codec(y: np.ndarray, enc: list[str], ext: str) -> np.ndarray:
    """Encode with a lossy codec and decode back to 16 kHz (phone call, Bluetooth, VoIP)."""
    with tempfile.TemporaryDirectory() as d:
        src, mid = Path(d) / "in.f32", Path(d) / f"mid.{ext}"
        src.write_bytes(y.astype(np.float32).tobytes())
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(src),
                        *enc, str(mid)], check=True, capture_output=True)
        out = decode(mid)
    n = len(y)
    return np.r_[out, np.zeros(max(0, n - len(out)), np.float32)][:n]


def frame_db(y: np.ndarray, F: int = 512) -> np.ndarray:
    fr = y[: y.size // F * F].reshape(-1, F).astype(np.float64)
    return 10 * np.log10((fr * fr).mean(axis=1) + 1e-12)


def speech_power(y: np.ndarray) -> float:
    """Mean power of the voiced frames (within 15 dB of the 95th-percentile frame)."""
    db = frame_db(y)
    if not db.size:
        return 1e-6
    sel = db >= np.percentile(db, 95) - 15
    return float(np.mean(10 ** (db[sel] / 10)))


def match_level(x: np.ndarray, ref_power: float, rel_db: float = 0.0) -> np.ndarray:
    return (x * np.sqrt(ref_power / max(speech_power(x), 1e-12)) * 10 ** (rel_db / 20)).astype(np.float32)


def room_tone(y: np.ndarray, seconds: float) -> np.ndarray:
    """The quietest second of `y` (away from its ends when it is long enough), tiled."""
    k = 31
    db = frame_db(y)
    if db.size < k + 2:
        return np.zeros(int(seconds * SR), np.float32)
    smooth = np.convolve(db, np.ones(k) / k, mode="valid")
    edge = int(3 / 0.032) if smooth.size > int(8 / 0.032) else 0
    lo, hi = edge, max(edge + 1, smooth.size - edge)
    i = lo + int(np.argmin(smooth[lo:hi]))
    snip = y[i * 512 : (i + k) * 512]
    reps = int(np.ceil(seconds * SR / max(1, snip.size)))
    return np.tile(snip, reps)[: int(seconds * SR)].astype(np.float32)


def rir(rt60: float, drr_db: float, rng: np.random.Generator) -> np.ndarray:
    """A synthetic room impulse response: direct path + exponentially decaying noise tail."""
    n = int(rt60 * 1.2 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.004 * SR)] = 0.0
    tail /= np.sqrt(np.sum(tail ** 2)) + 1e-12
    h = tail * 10 ** (-drr_db / 20)
    h[0] += 1.0
    return h.astype(np.float32)


def reverb(y: np.ndarray, rt60: float, drr_db: float, rng) -> np.ndarray:
    from scipy.signal import fftconvolve

    out = fftconvolve(y, rir(rt60, drr_db, rng))[: len(y)].astype(np.float32)
    return match_level(out, speech_power(y))


def colored_noise(n: int, rng, kind: str = "pink") -> np.ndarray:
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    s = np.fft.rfft(w)
    s[1:] /= np.sqrt(f[1:]) if kind == "pink" else f[1:] / 50.0  # brown: 1/f
    s[0] = 0
    x = np.fft.irfft(s, n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def add_noise(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y) / max(1, len(noise)))))
    noise = noise[: len(y)]
    p_n = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    g = np.sqrt(speech_power(y) / p_n / 10 ** (snr_db / 10))
    return (y + g * noise).astype(np.float32)


# ---------------------------------------------------------------- sources
def _line_runs(words: list, ix, max_gap: float = 6.0) -> list[list[dict]]:
    """Consecutive whole lines (every word timed, line numbers +1 each, in time order)."""
    by_seg: dict[tuple, list] = {}
    for w in words:
        d = int(ix.word_dua[w[0]])
        by_seg.setdefault((d, int(ix.word_segment[w[0]])), []).append(w)
    lines = []
    for (d, seg), ws in by_seg.items():
        ws = sorted(ws, key=lambda w: w[0])
        lo, hi = ix.dua_word_span[d]
        n_line = int(np.sum(ix.word_segment[lo:hi] == seg))
        if len(ws) != n_line or ws[-1][0] - ws[0][0] + 1 != len(ws):
            continue
        if any(b[1] < a[1] for a, b in zip(ws, ws[1:])) or ws[-1][2] - ws[0][1] > 20.0:
            continue
        lines.append({"seg": seg, "words": [[int(w[0]), round(float(w[1]), 3), round(float(w[2]), 3)] for w in ws],
                      "from": round(float(ws[0][1]) - 0.05, 3), "to": round(float(ws[-1][2]) + 0.1, 3)})
    lines.sort(key=lambda l: l["words"][0][1])
    runs, cur = [], []
    for ln in lines:
        if cur and (ln["seg"] != cur[-1]["seg"] + 1 or ln["from"] < cur[-1]["to"] - 0.3
                    or ln["from"] - cur[-1]["to"] > max_gap):
            runs.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        runs.append(cur)
    return runs


def sources_testsets(ix, which: str) -> list[dict]:
    import evaluate as ev
    from pause_eval import good_words

    ev.use_source(which if which != "studio" else "duaplayer", "all")
    duas = ev.load_all()
    out = []
    for dua in duas.values():
        if dua.id in ev.LINE_LABELS_UNRELIABLE:
            continue
        for rec in ev.load_recordings(dua):
            if which == "studio" and not ev.is_test(rec.reciter):
                continue
            words = good_words(rec, ix)
            if not words:
                continue
            runs = sorted(_line_runs(words, ix), key=len, reverse=True)
            for k, run in enumerate(runs[:2]):
                if len(run) < MIN_LINES:
                    continue
                out.append({"sid": f"{which}:{rec.audio_id}:{k}", "lane": which, "audio": str(rec.path),
                            "dua": dua.id, "voice": f"{which}:{rec.reciter}", "lines": run,
                            "tags": {"condition": rec.condition}})
    return out


def sources_harvest(ix) -> tuple[list[dict], list[dict]]:
    """Held-out uploaders' runs of confidently placed lines; and spans of du'as the app's
    corpus doesn't have (training-only Mafatih texts) as out-of-corpus readings. The
    Mafatih texts the corpus has since taken in (mafatih_corpus.py) are lane "mafatih":
    their alignments carried over word by word to the corpus's own lines. The held-out
    side has few readings of texts the app lacks, so the out-of-corpus readings add one
    each from up to MORE_OOC other uploaders, and an added text with fewer than
    MAFATIH_MIN held-out runs gets runs of other uploaders up to that (all listed in
    test_voices.json by `sources`, so training exports leave them out too; the phone's
    models predate the harvest)."""
    from harvest_label import HARVEST, LABELS, _meta_index, _reciter, read_jsonl
    from mafatih_corpus import book_words

    from dua_recognition.text import normalize

    added = book_words()
    metas = _meta_index()
    voices = {(r["platform"], r["id"]): r for r in read_jsonl(HARVEST / "voices.jsonl")}
    side = lambda rc: int(hashlib.sha1(rc.encode()).hexdigest()[:8], 16) % 100 < 3  # noqa: E731
    out, ooc, more, more_mf = [], [], {}, {}

    def ooc_source(res, si, sp, meta, audio, rc, good):
        return {"sid": f"harvest:{res['platform']}:{res['id']}:{si}", "lane": "harvest", "audio": str(audio),
                "dua": None, "text_id": sp["dua"], "voice": rc,
                "lines": [{"seg": ln["seg"], "words": [], "from": ln["start"] - 0.05, "to": ln["end"] + 0.1}
                          for ln in good],
                "tags": {"platform": res["platform"], "title": meta.get("title", "")[:120]}}

    def mafatih_runs(res, si, sp, add, meta, audio, rc):
        lo = ix.dua_word_span[ix.dua_ids.index(add["dua"])][0]
        timed: dict[int, list] = {}  # two book words can be one of ours (a separate وَ joined on)
        for ln in sp["lines"]:
            o = add["off"].get(ln["seg"])
            if not ln["ok"] or o is None or add["words"][o:o + len(ln["words"])] != ln["words"]:
                continue
            for j, (a, b) in enumerate(ln["times"]):
                w = add["local"].get(o + j)
                if w is not None:
                    t = timed.setdefault(lo + w, [a, b])
                    t[0], t[1] = min(t[0], a), max(t[1], b)
        words = [[w, a, b, 0.0] for w, (a, b) in timed.items()]
        runs = sorted(_line_runs(sorted(words, key=lambda w: w[1]), ix), key=len, reverse=True)
        return [{"sid": f"harvest:{res['platform']}:{res['id']}:{si}.{k}", "lane": "mafatih", "audio": str(audio),
                 "dua": add["dua"], "voice": rc, "lines": run,
                 "tags": {"platform": res["platform"], "title": meta.get("title", "")[:120]}}
                for k, run in enumerate(runs[:2]) if len(run) >= MIN_LINES]

    for lab in sorted(LABELS.glob("*/*.json")):
        if lab.name.endswith(".captions.json"):
            continue
        res = json.loads(lab.read_text(encoding="utf-8"))
        key = (res["platform"], res["id"])
        if key not in metas:
            continue
        meta, audio = metas[key]
        rc = _reciter(res["platform"], meta)
        if voices.get(key, {}).get("verdict") != "ok":
            continue
        if not side(rc):
            for si, sp in enumerate(res["spans"]):
                add = added.get(sp["dua"])
                if add and sp["usable"] and rc not in more_mf.get(add["dua"], {}):
                    runs = mafatih_runs(res, si, sp, add, meta, audio, rc)
                    if runs:
                        more_mf.setdefault(add["dua"], {})[rc] = runs[0]
            if rc in more or any(sp["dua"] in added for sp in res["spans"]):
                continue
            for si, sp in enumerate(res["spans"]):
                good = [ln for ln in sp["lines"] if ln["ok"]]
                if sp["usable"] and sp["dua"] and sp["dua"] not in ix.dua_ids and len(good) >= 8:
                    more[rc] = ooc_source(res, si, sp, meta, audio, rc, good)
                    break
            continue
        for si, sp in enumerate(res["spans"]):
            if not sp["usable"]:
                continue
            add = added.get(sp["dua"])
            if add:
                out += mafatih_runs(res, si, sp, add, meta, audio, rc)
                continue
            if sp["dua"] not in ix.dua_ids:
                good = [ln for ln in sp["lines"] if ln["ok"]]
                if len(good) >= 8:
                    ooc.append(ooc_source(res, si, sp, meta, audio, rc, good))
                continue
            d = ix.dua_ids.index(sp["dua"])
            lo, hi = ix.dua_word_span[d]
            words = []
            for ln in sp["lines"]:
                if not ln["ok"]:
                    continue
                ws = np.flatnonzero(ix.word_segment[lo:hi] == ln["seg"]) + lo
                if len(ws) != len(ln["words"]):
                    continue
                if [normalize(ix.words[int(w)].text) for w in ws] != ln["words"] and \
                        [ix.words[int(w)].text for w in ws] != ln["words"]:
                    continue
                words += [[int(w), a, b, 0.0] for w, (a, b) in zip(ws, ln["times"])]
            runs = sorted(_line_runs(sorted(words, key=lambda w: w[1]), ix), key=len, reverse=True)
            for k, run in enumerate(runs[:2]):
                if len(run) < MIN_LINES:
                    continue
                out.append({"sid": f"harvest:{res['platform']}:{res['id']}:{si}.{k}", "lane": "harvest",
                            "audio": str(audio), "dua": sp["dua"], "voice": rc, "lines": run,
                            "tags": {"platform": res["platform"], "title": meta.get("title", "")[:120]}})
    order = lambda rc: hashlib.sha1(rc.encode()).hexdigest()  # noqa: E731
    ooc += [more[rc] for rc in sorted(more, key=order)[:MORE_OOC]]
    for dua, cands in sorted(more_mf.items()):
        need = MAFATIH_MIN - len({s["voice"] for s in out if s["lane"] == "mafatih" and s["dua"] == dua})
        out += [cands[rc] for rc in sorted(cands, key=order)[:max(0, need)]]
    return out, ooc


def sources_user(ix) -> list[dict]:
    """Hasan's own phone sessions (data/sessions) with word truth from the CTC teacher
    (session_eval.py --truth-model wav2vec2-quran-dua-voices): his voice through his phone."""
    out = []
    truth_dir = ROOT / "data" / "cache" / "session_truth" / "wav2vec2-quran-dua-voices"
    for f in sorted(truth_dir.glob("*.json")):
        tr = json.loads(f.read_text())
        wavs = list((ROOT / "data" / "sessions").glob(f"{f.stem}.wav"))
        if not tr or not wavs:
            continue
        runs = sorted(_line_runs(sorted(tr, key=lambda w: w[1]), ix), key=len, reverse=True)
        if not runs or len(runs[0]) < MIN_LINES:
            continue
        d = ix.dua_ids[ix.word_dua[runs[0][0]["words"][0][0]]]
        out.append({"sid": f"user:{f.stem}:0", "lane": "user", "audio": str(wavs[0]), "dua": d,
                    "voice": "user:hasan", "lines": runs[0], "tags": {"condition": "phone"}})
    return out


def _salawat_clips(ix, sources: list[dict]) -> list[dict]:
    """Spoken salawat lines (the reader's own voice where possible): every source line whose
    words are exactly the salawat."""
    from dua_recognition.text import normalize

    target = normalize(SALAWAT).split()
    out = []
    for s in sources:
        for ln in s["lines"]:
            if ln["words"] and [normalize(ix.words[w[0]].text) for w in ln["words"]] == target:
                out.append({"sid": s["sid"], "audio": s["audio"], "voice": s["voice"], "from": ln["from"],
                            "to": ln["to"]})
    return out


def cmd_sources(args) -> None:
    import evaluate as ev

    from dua_recognition.align import encode

    ix = ev.CorpusIndex(ev.load_all())

    def tag(srcs):
        for s in srcs:
            letters = sum(len(encode(ix.words[w[0]].text)) for ln in s["lines"] for w in ln["words"])
            inside = sum(ln["words"][-1][2] - ln["words"][0][1] for ln in s["lines"])
            s["tags"]["rate"] = round(letters / max(inside, 1e-6), 2)
            s["tags"]["n_lines"] = len(s["lines"])

    BENCH.mkdir(parents=True, exist_ok=True)
    if args.lane:  # just these sources; the rest of sources.jsonl (and the grid built on it) kept as is
        hv, ooc = sources_harvest(ix)
        if args.lane == "ooc":
            srcs, keep = ooc, [s for s in load_sources() if s["dua"]]
        else:
            srcs = [s for s in hv if s["lane"] == args.lane]
            keep = [s for s in load_sources() if s["lane"] != args.lane]
            tag(srcs)
        with (BENCH / "sources.jsonl").open("w", encoding="utf-8") as f:
            for s in keep + srcs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        tv = json.loads((BENCH / "test_voices.json").read_text(encoding="utf-8"))
        recs = {(r["source"], r["rec"]) for r in tv["recordings"]} | {tuple(s["sid"].split(":")[1:3]) for s in srcs}
        tv["recordings"] = [{"source": p, "rec": r} for p, r in sorted(recs)]
        (BENCH / "test_voices.json").write_text(json.dumps(tv, indent=0), encoding="utf-8")
        h = sum(ln["to"] - ln["from"] for s in srcs for ln in s["lines"]) / 3600
        print(f"{args.lane}: {len(srcs)} sources, {len({s['voice'] for s in srcs})} voices, "
              f"{len({s['dua'] or s['text_id'] for s in srcs})} texts, {h:.1f} h; test recordings listed {len(recs)}")
        return
    srcs = sources_testsets(ix, "studio") + sources_testsets(ix, "majlis") + sources_user(ix)
    hv, ooc = sources_harvest(ix)
    srcs += hv
    tag(srcs)
    with (BENCH / "sources.jsonl").open("w", encoding="utf-8") as f:
        for s in srcs + ooc:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    sal = _salawat_clips(ix, srcs)
    (BENCH / "salawat.json").write_text(json.dumps(sal, ensure_ascii=False, indent=0), encoding="utf-8")
    # The harvest session's export drops every recording listed here (and voice matches).
    recs = sorted({tuple(s["sid"].split(":")[1:3]) for s in hv + ooc})
    (BENCH / "test_voices.json").write_text(json.dumps(
        {"note": "bench.py sources: held-out harvest recordings used as test audio",
         "recordings": [{"source": p, "rec": r} for p, r in recs]}, indent=0), encoding="utf-8")
    by = {}
    for s in srcs:
        by.setdefault(s["lane"], []).append(s)
    for lane, ss in by.items():
        h = sum(ln["to"] - ln["from"] for s in ss for ln in s["lines"]) / 3600
        print(f"{lane:8s} {len(ss):4d} runs, {len({s['voice'] for s in ss}):3d} voices, "
              f"{len({s['dua'] for s in ss}):3d} du'as, {h:.1f} h of lines; rate median "
              f"{np.median([s['tags']['rate'] for s in ss]):.1f} letters/s")
    print(f"out of corpus: {len(ooc)} spans ({len({s['text_id'] for s in ooc})} texts); salawat clips {len(sal)}; "
          f"test recordings listed {len(recs)}")


def load_sources() -> list[dict]:
    return [json.loads(x) for x in (BENCH / "sources.jsonl").read_text(encoding="utf-8").splitlines() if x]


# ---------------------------------------------------------------- programs
class Prog:
    """A reading built from source spans. Ops (in output order):
        ["src", a, b]                  source audio a..b (source time)
        ["tone", s]                    s seconds of the source's room tone
        ["clip", kind, ref, rel_db]    a clip from elsewhere (talk / salawat), level-matched
    Truth is kept in output time: words [dua, local word, start, end], events, still spans."""

    def __init__(self, src: dict, ix):
        self.src, self.ix = src, ix
        self.ops, self.t = [], 0.0
        self.words, self.events, self.still = [], [], []
        self.next_event = None  # the kind of move the next play() makes (back, skip, repeat...)

    def _local(self, w: int) -> list:
        d = int(self.ix.word_dua[w])
        return [self.ix.dua_ids[d], int(w - self.ix.dua_word_span[d][0])]

    def play(self, lines: list[dict], a: float | None = None, b: float | None = None, words: list | None = None,
             event: str | None = None) -> None:
        """Source audio from the first line's start to the last line's end (or a..b), with the
        words inside it as truth."""
        a = lines[0]["from"] if a is None else a
        b = lines[-1]["to"] if b is None else b
        d = self.t - a
        ws = words if words is not None else [w for ln in lines for w in ln["words"]]
        if event and ws:
            self.events.append({"t": round(ws[0][1] + d, 3), "kind": event, "word": self._local(ws[0][0])})
        for w, wa, wb in ws:
            if a - 0.01 <= wa and wb <= b + 0.01:
                self.words.append(self._local(w) + [round(wa + d, 3), round(wb + d, 3)])
        self.ops.append(["src", round(a, 3), round(b, 3)])
        self.t += b - a

    def tone(self, s: float, kind: str = "pause") -> None:
        self.ops.append(["tone", round(s, 3)])
        if kind:
            self.still.append([round(self.t, 3), round(self.t + s, 3), kind])
        self.t += s

    def clip(self, kind: str, ref: dict, dur: float, rel_db: float = 0.0) -> None:
        self.ops.append(["clip", kind, ref, rel_db])
        self.still.append([round(self.t, 3), round(self.t + dur, 3), kind])
        self.t += dur

    def item(self, scenario: str, fx: list | None = None, tempo: float = 1.0, extra: dict | None = None) -> dict:
        sid = self.src["sid"]
        iid = f"{scenario}-" + hashlib.sha1(sid.encode()).hexdigest()[:10]
        k = 1.0 / tempo
        return {"id": iid, "scenario": scenario, "sid": sid, "lane": self.src["lane"], "voice": self.src["voice"],
                "dua": self.src["dua"], "audio": self.src["audio"], "ops": self.ops, "fx": fx or [], "tempo": tempo,
                "words": [[d, w, round(a * k, 3), round(b * k, 3)] for d, w, a, b in self.words],
                "events": [dict(e, t=round(e["t"] * k, 3)) for e in self.events],
                "still": [[round(a * k, 3), round(b * k, 3), kind] for a, b, kind in self.still],
                "duration": round(self.t * k, 3), "tags": self.src["tags"], **(extra or {})}


def _take(lines: list[dict], i0: int, max_s: float) -> list[dict]:
    out = []
    for ln in lines[i0:]:
        if out and ln["to"] - out[0]["from"] > max_s:
            break
        out.append(ln)
    return out


def sc_flow(src, ix, rng, max_s=240.0, **_):
    p = Prog(src, ix)
    p.play(_take(src["lines"], 0, max_s))
    return p


def sc_midstart(src, ix, rng, **_):
    L = src["lines"]
    if len(L) < 12:
        return None
    i0 = rng.randrange(len(L) // 3, 2 * len(L) // 3)
    p = Prog(src, ix)
    p.play(_take(L, i0, 120.0), event="start")
    return p


def _runs_with(src, ix, rng, every: tuple[int, int], max_s: float, between):
    """Read lines in order in runs of n lines (n drawn from `every`); after each run,
    between(p, i, n) may insert things (and set p.next_event) and returns the line to go on
    from (None: the next one). Stops after max_s seconds of output."""
    L = src["lines"]
    p = Prog(src, ix)
    i = 0
    while i < len(L) and p.t < max_s:
        n = rng.randint(*every)
        run = L[i : i + n]
        p.play(run, event=p.next_event)
        p.next_event = None
        last = i + len(run) - 1
        if last >= len(L) - 1 or p.t >= max_s:
            break
        nxt = between(p, last, len(run))
        i = last + 1 if nxt is None else nxt
    return p


def sc_pause(src, ix, rng, **_):
    state = {"long": False}

    def between(p, i, n):
        if not state["long"] and p.t > 60:
            state["long"] = True
            p.tone(15.0)
        else:
            p.tone(4.0)
        return None

    return _runs_with(src, ix, rng, (3, 3), 180.0, between)


def sc_talk(src, ix, rng, talk_clips=None, **_):
    def between(p, i, n):
        c = rng.choice(talk_clips)
        dur = min(6.0, c["dur"])
        p.tone(0.5, "talk")
        p.clip("talk", {"clip": c["clip"], "dur": round(dur, 3)}, dur, -2.0)
        p.tone(0.5, "talk")
        return None

    return _runs_with(src, ix, rng, (3, 5), 180.0, between)


def sc_salawat(src, ix, rng, salawat=None, **_):
    own = [c for c in salawat if c["voice"] == src["voice"]] or [c for c in salawat if c["sid"].split(":")[0] == src["lane"]]
    own = own or salawat
    if not own:
        return None

    def between(p, i, n):
        c = rng.choice(own)
        dur = c["to"] - c["from"]
        p.tone(0.4, "salawat")
        p.clip("salawat", {"audio": c["audio"], "from": c["from"], "to": c["to"]}, dur, 0.0)
        p.tone(0.4, "salawat")
        return None

    return _runs_with(src, ix, rng, (3, 5), 180.0, between)


def sc_repeat(src, ix, rng, **_):
    def between(p, i, n):
        p.tone(rng.uniform(0.3, 0.8), "breath")
        p.next_event = "repeat"
        return i  # the same line again

    return _runs_with(src, ix, rng, (3, 3), 150.0, between)


def sc_back(src, ix, rng, **_):
    def between(p, i, n):
        k = min(rng.choice([1, 1, 2, 2, 3]), n - 2)  # lands after the run's first line: always progress
        if k < 1 or i - k < 0:
            return None
        p.tone(rng.uniform(0.4, 1.2), "breath")
        p.next_event = "back"
        return i - k

    return _runs_with(src, ix, rng, (4, 6), 180.0, between)


def sc_skip(src, ix, rng, **_):
    L = src["lines"]

    def between(p, i, n):
        k = rng.choice([1, 1, 2, 3, 4])
        if i + 1 + k >= len(L):
            return None
        p.tone(rng.uniform(0.4, 1.2), "breath")
        p.next_event = "skip"
        return i + 1 + k

    return _runs_with(src, ix, rng, (3, 5), 180.0, between)


def sc_jumps(src, ix, rng, **_):
    L = src["lines"]
    if len(L) < 8:
        return None
    p = Prog(src, ix)
    p.play(L[:3])
    cur, total = 2, 0.0
    while p.t < 150.0:
        choices = [i for i in range(len(L)) if i not in (cur, cur + 1)]
        i = rng.choice(choices)
        n = rng.choice([1, 2, 2, 3])
        run = L[i : i + n]
        p.tone(rng.uniform(0.3, 1.2), "breath")
        p.play(run, event="jump")
        cur = i + len(run) - 1
    return p


def sc_stumble(src, ix, rng, **_):
    L = src["lines"]

    def between(p, i, n):
        nxt = L[i + 1]
        if len(nxt["words"]) < 3:
            return None
        k = rng.choice([1, 2])
        part = nxt["words"][:k]
        p.tone(0.3, None)
        p.play([nxt], a=nxt["from"], b=part[-1][2] + 0.05, words=part)
        p.tone(rng.uniform(0.3, 0.7), "breath")
        p.next_event = "restart"
        return i + 1

    return _runs_with(src, ix, rng, (3, 5), 180.0, between)


def sc_switch(src, ix, rng, others=None, **_):
    cands = [o for o in others if o["dua"] != src["dua"] and o["lane"] == src["lane"] and len(o["lines"]) >= 8]
    if not cands:
        return None
    o = rng.choice(cands)
    p = Prog(src, ix)
    p.play(_take(src["lines"], 0, 75.0))
    p.tone(2.5)
    q = Prog(o, ix)
    lines = _take(o["lines"], 0, 75.0)
    d = p.t - lines[0]["from"]
    p.events.append({"t": round(lines[0]["words"][0][1] + d, 3), "kind": "switch",
                     "word": q._local(lines[0]["words"][0][0])})
    for w, wa, wb in (w for ln in lines for w in ln["words"]):
        p.words.append(q._local(w) + [round(wa + d, 3), round(wb + d, 3)])
    p.ops.append(["other", o["audio"], lines[0]["from"], lines[-1]["to"]])
    p.t += lines[-1]["to"] - lines[0]["from"]
    return p


def sc_ooc(src, ix, rng, **_):
    p = Prog(src, ix)
    lines = _take(src["lines"], 0, 120.0)
    p.ops.append(["src", round(lines[0]["from"], 3), round(lines[-1]["to"], 3)])
    p.t = lines[-1]["to"] - lines[0]["from"]
    return p


def sc_tempo(factor):
    def f(src, ix, rng, **_):
        p = Prog(src, ix)
        p.play(_take(src["lines"], 0, 150.0 * factor))
        return p

    return f


def sc_cond(src, ix, rng, **_):
    p = Prog(src, ix)
    p.play(_take(src["lines"], 0, 150.0))
    return p


def sc_combo(src, ix, rng, **_):
    L = src["lines"]

    def between(p, i, n):
        r = rng.random()
        if r < 0.35:
            p.tone(rng.uniform(3.0, 8.0))
            return None
        k = min(rng.choice([1, 2]), n - 2)
        if k < 1 or i - k < 0:
            return None
        p.tone(rng.uniform(0.4, 1.2), "breath")
        p.next_event = "back"
        return i - k

    return _runs_with(src, ix, rng, (3, 6), 180.0, between)


# scenario -> (program builder, effects, lanes it runs on, tempo)
SCENARIOS = {
    "flow": (sc_flow, [], ("user", "studio", "majlis", "harvest", "mafatih"), 1.0),
    "midstart": (sc_midstart, [], ("studio", "majlis", "harvest", "mafatih"), 1.0),
    "pause": (sc_pause, [], ("user", "studio", "majlis", "harvest", "mafatih"), 1.0),
    "talk": (sc_talk, [], ("user", "studio", "harvest", "mafatih"), 1.0),
    "salawat": (sc_salawat, [], ("studio", "harvest"), 1.0),
    "repeat": (sc_repeat, [], ("user", "studio", "majlis", "harvest", "mafatih"), 1.0),
    "back": (sc_back, [], ("user", "studio", "majlis", "harvest", "mafatih"), 1.0),
    "skip": (sc_skip, [], ("user", "studio", "majlis", "harvest", "mafatih"), 1.0),
    "jumps": (sc_jumps, [], ("studio", "harvest"), 1.0),
    "stumble": (sc_stumble, [], ("studio", "harvest"), 1.0),
    "switch": (sc_switch, [], ("studio", "harvest"), 1.0),
    "ooc": (sc_ooc, [], ("harvest",), 1.0),
    "slow": (sc_tempo(0.7), [], ("studio", "harvest"), 0.7),
    "fast": (sc_tempo(1.3), [], ("studio", "harvest"), 1.3),
    "room": (sc_cond, [["reverb", 0.5, 3.0]], ("studio", "harvest"), 1.0),
    "hall": (sc_cond, [["reverb", 1.4, -4.0]], ("studio", "harvest"), 1.0),
    "babble": (sc_cond, [["babble", 10.0]], ("studio", "harvest"), 1.0),
    "fan": (sc_cond, [["noise", "brown", 8.0]], ("studio", "harvest"), 1.0),
    "bgrecite": (sc_cond, [["bgrecite", -10.0]], ("studio", "harvest"), 1.0),
    "far": (sc_cond, [["reverb", 0.8, -8.0], ["noise", "pink", 15.0], ["gain", -22.0]], ("studio", "harvest"), 1.0),
    "clip": (sc_cond, [["clip", 15.0]], ("studio", "harvest"), 1.0),
    "phonecall": (sc_cond, [["codec", "amr"]], ("studio", "harvest"), 1.0),
    "lowbitrate": (sc_cond, [["codec", "opus12"]], ("studio", "harvest"), 1.0),
    "combo": (sc_combo, [["reverb", 0.6, 0.0], ["babble", 15.0]], ("studio", "majlis", "harvest"), 1.0),
}


def _talk_clips(n: int = 400) -> list[dict]:
    rows = [json.loads(x) for x in (ROOT / "data" / "cache" / "finetune" / "cv_ar_test.jsonl").read_text(
        encoding="utf-8").splitlines() if x]
    rng = random.Random(1)
    rng.shuffle(rows)
    out = []
    for r in rows:
        f = ROOT / r["clip"]
        if not f.exists():
            continue
        dur = np.load(f, mmap_mode="r").shape[0] / SR
        if dur >= 3.0:
            out.append({"clip": r["clip"], "dur": round(float(dur), 3)})
        if len(out) >= n:
            break
    return out


def cmd_build(args) -> None:
    import evaluate as ev

    ix = ev.CorpusIndex(ev.load_all())
    srcs = load_sources()
    sal = json.loads((BENCH / "salawat.json").read_text(encoding="utf-8"))
    talk = _talk_clips()
    known = [s for s in srcs if s["dua"]]
    ooc = [s for s in srcs if not s["dua"]]
    by_lane = {}
    for s in known:
        by_lane.setdefault(s["lane"], []).append(s)
    # Harvest: one run per uploader per scenario, a different run of that uploader where it has
    # several, so the grid as a whole hears as many recordings as possible.
    hv_by_voice = {}
    for s in by_lane.get("harvest", []):
        hv_by_voice.setdefault(s["voice"], []).append(s)
    only = set(args.scenarios) if args.scenarios else None
    n_new = 0
    for si, (name, (fn, fx, lanes, tempo)) in enumerate(SCENARIOS.items()):
        if only and name not in only:
            continue
        out_dir = BENCH / "items" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pool = []
        if name == "ooc":
            rng = random.Random(7)
            seen = set()
            for s in sorted(ooc, key=lambda s: s["sid"]):
                if s["voice"] not in seen:
                    seen.add(s["voice"])
                    pool.append(s)
            rng.shuffle(pool)
            pool = pool[: args.harvest_per_cell]
        else:
            for lane in lanes:
                if lane == "harvest":
                    voices = sorted(hv_by_voice)
                    rng = random.Random(si)
                    rng.shuffle(voices)
                    for v in voices[: args.harvest_per_cell]:
                        runs = sorted(hv_by_voice[v], key=lambda s: s["sid"])
                        pool.append(runs[si % len(runs)])
                else:
                    pool += by_lane.get(lane, [])
        for s in pool:
            rng = random.Random(f"{name}|{s['sid']}")
            p = fn(s, ix, rng, talk_clips=talk, salawat=sal, others=known)
            if p is None or p.t < 20.0:
                continue
            fx_full = []
            for e in fx:
                e = list(e)
                if e[0] == "babble":
                    e.append([c["clip"] for c in rng.sample(talk, 6)])
                if e[0] == "bgrecite":
                    others = [o for o in known if o["dua"] != s["dua"] and o["voice"] != s["voice"]]
                    o = rng.choice(others)
                    e.append({"audio": o["audio"], "from": o["lines"][0]["from"]})
                fx_full.append(e)
            it = p.item(name, fx_full, tempo, extra={"seed": rng.randrange(1 << 30)})
            if name == "ooc":
                it["text_id"] = s["text_id"]
            f = out_dir / f"{it['id']}.json"
            if not f.exists() or args.rebuild:
                f.write_text(json.dumps(it, ensure_ascii=False), encoding="utf-8")
                n_new += 1
        n = len(list(out_dir.glob("*.json")))
        print(f"{name:10s} {n:4d} items", flush=True)
    print(f"{n_new} items written")


def split_of(voice: str) -> str:
    """dev (tune on it) or test (report on it): half the voices each, by a hash of the voice."""
    return "dev" if int(hashlib.sha1(voice.encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def load_items(scenarios=None, split: str = "all") -> list[dict]:
    from mafatih_corpus import TEXTS

    taken_in = {t[0] for t in TEXTS}  # no longer unknown du'as: lane "mafatih" reads them
    out = []
    for f in sorted((BENCH / "items").glob("*/*.json")):
        if scenarios and f.parent.name not in scenarios:
            continue
        it = json.loads(f.read_text(encoding="utf-8"))
        if it.get("text_id") in taken_in:
            continue
        if split == "all" or split_of(it["voice"]) == split:
            out.append(it)
    return out


# ---------------------------------------------------------------- rendering
_AUDIO_CACHE: dict = {}


def _source_window(path: str, a: float, b: float) -> tuple[np.ndarray, float]:
    """Decoded path[a-12 s : b+12 s] (cached for the item being rendered) and its start time."""
    key = (path, round(a, 2), round(b, 2))
    if key not in _AUDIO_CACHE:
        if len(_AUDIO_CACHE) > 4:
            _AUDIO_CACHE.clear()
        t0 = max(0.0, a - 12.0)
        _AUDIO_CACHE[key] = (decode(path, t0, b + 12.0), t0)
    return _AUDIO_CACHE[key]


def render(it: dict) -> np.ndarray:
    srcs = [op for op in it["ops"] if op[0] == "src"]
    a0 = min((op[1] for op in srcs), default=0.0)
    b0 = max((op[2] for op in srcs), default=0.0)
    y_src, t0 = _source_window(it["audio"], a0, b0)
    tone = room_tone(y_src, 20.0)
    ref_p = speech_power(y_src[int(max(0, a0 - t0) * SR) : int((b0 - t0) * SR)])
    pieces, tone_at = [], 0
    for op in it["ops"]:
        if op[0] == "src":
            pieces.append(y_src[int((op[1] - t0) * SR) : int((op[2] - t0) * SR)])
        elif op[0] == "tone":
            n = int(op[1] * SR)
            seg = np.tile(tone, int(np.ceil((tone_at + n) / tone.size)) + 1)[tone_at : tone_at + n]
            tone_at = (tone_at + n) % tone.size
            pieces.append(seg)
        elif op[0] == "clip":
            ref = op[2]
            if "clip" in ref:
                x = np.load(ROOT / ref["clip"]).astype(np.float32)
                x = x / 32768.0 if np.abs(x).max() > 2.0 else x
                x = x[: int(ref["dur"] * SR)]
            else:
                x = decode(ref["audio"], ref["from"], ref["to"])
            x = match_level(x, ref_p, op[3])
            n = int(round((ref.get("dur") or (ref["to"] - ref["from"])) * SR))
            x = np.r_[x, np.zeros(max(0, n - x.size), np.float32)][:n]
            pieces.append(x + tone[: x.size] if tone.size >= x.size else x)
        elif op[0] == "other":
            y2 = decode(op[1], op[2], op[3])
            pieces.append(match_level(y2, ref_p))
    y = np.concatenate(pieces + [tone[: int(1.5 * SR)]]).astype(np.float32)
    rng = np.random.default_rng(it.get("seed", 0))
    for e in it["fx"]:
        if e[0] == "reverb":
            y = reverb(y, e[1], e[2], rng)
        elif e[0] == "noise":
            y = add_noise(y, colored_noise(len(y), rng, e[1]), e[2])
        elif e[0] == "babble":
            parts = []
            for c in e[2]:
                x = np.load(ROOT / c).astype(np.float32)
                x = x / 32768.0 if np.abs(x).max() > 2.0 else x
                x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
                parts.append(np.tile(x, int(np.ceil(len(y) / max(1, len(x)))))[: len(y)])
            nz = np.sum(parts, axis=0) if parts else np.zeros_like(y)
            y = add_noise(y, np.roll(nz, rng.integers(0, len(y))), e[1])
        elif e[0] == "bgrecite":
            ref = e[2]
            x = decode(ref["audio"], ref["from"], ref["from"] + len(y) / SR + 1.0)
            x = np.r_[x, np.zeros(max(0, len(y) - x.size), np.float32)][: len(y)]
            y = (y + match_level(x, speech_power(y), e[1])).astype(np.float32)
        elif e[0] == "gain":
            y = (y * 10 ** (e[1] / 20)).astype(np.float32)
        elif e[0] == "clip":
            pk = np.percentile(np.abs(y), 99.9) + 1e-9
            y = np.clip(y / pk * 10 ** (e[1] / 20), -1.0, 1.0).astype(np.float32)
        elif e[0] == "codec":
            if e[1] == "amr":
                y = ffmpeg_codec(y, ["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "7.95k"], "amr")
            elif e[1] == "opus12":
                y = ffmpeg_codec(y, ["-c:a", "libopus", "-b:a", "12k", "-application", "voip"], "ogg")
    if it.get("tempo", 1.0) != 1.0:
        y = ffmpeg_filter(y, f"atempo={it['tempo']}")
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def quiet_track(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """follow_eval.item_quiet over rendered audio: (times, seconds without voice)."""
    F = 512
    fr = y[: y.size // F * F].reshape(-1, F).astype(np.float64)
    db = 10 * np.log10((fr * fr).mean(axis=1) + 1e-12)
    sound = db >= np.percentile(db, 95) - 15.0
    q = np.empty(db.size)
    run, last = 0, 0
    for i, snd in enumerate(sound):
        run = run + 1 if snd else 0
        if run >= 3:
            last = i + 1
        q[i] = (i + 1 - last) * F / SR
    return ((np.arange(db.size) + 1) * F / SR).astype(np.float32), q.astype(np.float32)


# ---------------------------------------------------------------- model passes
def asr_paths(it: dict, asr_tag: str, ctc_tag: str) -> tuple[Path, Path, Path]:
    return (BENCH / "asr" / asr_tag / f"{it['id']}.json",
            BENCH / "ctc" / f"{ctc_tag}_w{CTC_WINDOW:g}_h{CTC_HOP:g}" / f"{it['id']}.npz",
            BENCH / "quiet" / f"{it['id']}.npz")


def cmd_asr(args) -> None:
    os.environ.setdefault("DUA_ASR_DEVICE", args.device)
    from dua_recognition.asr import _quiet, _tail_activity, transcribe_batch
    from dua_recognition.ctc_student import load_ctc

    items = load_items(args.scenarios)
    asr_tag, ctc_tag = Path(args.model).name.removesuffix("-ct2"), Path(args.ctc_model).name
    todo = []
    for it in items:
        fa, fc, fq = asr_paths(it, asr_tag, ctc_tag)
        if not (fa.exists() and fc.exists() and fq.exists()):
            todo.append(it)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} of {len(items)} items need model passes ({asr_tag}, {ctc_tag})", flush=True)
    ctc = None
    t_start, done_s = time.time(), 0.0
    for k, it in enumerate(todo):
        fa, fc, fq = asr_paths(it, asr_tag, ctc_tag)
        for f in (fa, fc, fq):
            f.parent.mkdir(parents=True, exist_ok=True)
        y = render(it)
        if not fq.exists():
            qt, qv = quiet_track(y)
            np.savez(fq, t=qt, q=qv)
        if not fa.exists():
            times = list(range(1, int(len(y) / SR) + 1))
            wins = [y[int(max(0, s - WINDOW) * SR) : int(s * SR)] for s in times]
            texts = []
            for b0 in range(0, len(wins), 16):
                texts += transcribe_batch(wins[b0 : b0 + 16], model=args.model)
            quiet = []
            for w in wins:
                probs, db, floor = _tail_activity(w, 3.0)
                quiet.append(3.0 if probs is None else round(_quiet(probs, db, floor, 6.0), 3))
            fa.write_text(json.dumps({"rows": [[s, x] for s, x in zip(times, texts)], "quiet": quiet},
                                     ensure_ascii=False), encoding="utf-8")
        if not fc.exists():
            ctc = ctc or load_ctc(args.ctc_model)
            end = len(y) / SR
            times = [round((j + 1) * CTC_HOP, 3) for j in range(int((end + 1e-6) // CTC_HOP))]
            arr, lens, _, _ = ctc.windows(y, times, CTC_WINDOW, 32)
            np.savez(fc, lp=arr, n_frames=lens, t=np.array(times))
        done_s += len(y) / SR
        el = time.time() - t_start
        print(f"[{k + 1}/{len(todo)}] {it['id']:28s} {len(y) / SR:6.0f} s audio  "
              f"({done_s / max(el, 1e-9):.1f}x real time)", flush=True)


# ---------------------------------------------------------------- scoring
_W: dict = {}


def _worker_init(cfg_kw: dict, fw_kw: dict, asr_tag: str, ctc_tag: str, delay: float, f_delay: float,
                 display: str = "follower", sc_kw: dict | None = None, same_text: int = 0) -> None:
    from dataclasses import replace

    import evaluate as ev
    from dua_recognition.follower import FollowerConfig
    from dua_recognition.tracker import TrackerConfig

    _W["ix"] = ev.CorpusIndex(ev.load_all())
    _W["cfg"] = replace(TrackerConfig(), **cfg_kw)
    _W["fcfg"] = replace(FollowerConfig(), **fw_kw)
    _W.update(asr_tag=asr_tag, ctc_tag=ctc_tag, delay=delay, f_delay=f_delay, display=display)
    if sc_kw is not None:
        from dua_recognition.stream_follower import StreamConfig

        _W["scfg"] = replace(StreamConfig(), **sc_kw)
    _W["same_text"] = same_text
    if same_text:
        _W["passages"] = _passages(_W["ix"], same_text)


def _passages(ix, k: int) -> dict:
    """k-word passages more than one du'a reads: words -> {du'a index: [its last word, ...]}."""
    text = [w.text for w in ix.words]
    seen: dict = {}
    for d, (lo, hi) in enumerate(ix.dua_word_span):
        for e in range(lo + k - 1, hi):
            seen.setdefault(tuple(text[e - k + 1 : e + 1]), {}).setdefault(d, []).append(e)
    return {key: v for key, v in seen.items() if len(v) > 1}


def _same_text(shown: list, truth_word: list, ix, k: int) -> list:
    """The display moved onto the reader's du'a wherever it shows the same k words the reader's
    du'a reads (Ayat al-Kursi in three texts, the salam passages of the ziyarat): no display
    could tell those apart, and the reader sees their own words."""
    passages, text, out = _W["passages"], [w.text for w in ix.words], []
    for s, tw in zip(shown, truth_word):
        if s is None or tw is None or s[0] == ix.dua_ids[ix.word_dua[tw]]:
            out.append(s)
            continue
        w, td = s[2], int(ix.word_dua[tw])
        whole = w - k + 1 >= ix.dua_word_span[ix.word_dua[w]][0]  # k words inside the shown du'a
        hits = passages.get(tuple(text[w - k + 1 : w + 1]), {}).get(td) if whole else None
        if not hits:
            out.append(s)
            continue
        p = min(hits, key=lambda e: abs(e - tw))
        out.append((ix.dua_ids[td], int(ix.word_segment[p]), p))
    return out


def replay(it: dict) -> list[tuple] | None:
    """The phone's display for one item: per 0.1 s tick, (dua id, line, global word) or None."""
    import evaluate as ev
    import follow_eval as fe
    import word_eval as we

    from dua_recognition.align import encode

    ix, cfg, fcfg = _W["ix"], _W["cfg"], _W["fcfg"]
    fa, fc, fq = asr_paths(it, _W["asr_tag"], _W["ctc_tag"])
    if not (fa.exists() and fc.exists() and fq.exists()):
        return None
    # The tracker's replay depends only on its config, the ASR rows and the corpus: cached, so
    # follower variants re-run in a fraction of the time.
    if "tkey" not in _W:
        ref = hashlib.sha1(" ".join(w.text for w in ix.words).encode("utf-8")).hexdigest()
        _W["tkey"] = hashlib.sha1(repr((cfg, _W["asr_tag"], _W["delay"], ref)).encode()).hexdigest()[:12]
    tc = BENCH / "tracker_cache" / _W["tkey"] / f"{it['id']}.pkl"
    if tc.exists() and tc.stat().st_mtime >= fa.stat().st_mtime:
        import pickle

        ups, anchors = pickle.loads(tc.read_bytes())
    else:
        import pickle

        a = json.loads(fa.read_text(encoding="utf-8"))
        rows = a["rows"]
        costs = ev.LazyCosts(ix, [x for _, x in rows])
        letters = [len(encode(x)) if x else 0 for _, x in rows]
        anchors = []
        ups = we.hmm_updates(ix, rows, costs, a["quiet"], cfg, _W["delay"], "fixed", None, still=True,
                             anchors=anchors, letters=letters)
        anchors = [(t, w, None if m is None else np.asarray(m, dtype=np.float32)) for t, w, m in anchors]
        tc.parent.mkdir(parents=True, exist_ok=True)
        tc.write_bytes(pickle.dumps((ups, anchors)))
    z = np.load(fq)
    good = sorted([[ix.dua_word_span[ix.dua_ids.index(d)][0] + w, a, b] for d, w, a, b in it["words"]
                   if d in ix.dua_ids], key=lambda x: x[1])
    fit = {"id": it["id"], "ctc": fc, "anchors": anchors, "ups": ups, "good": good, "pause_s": 0.0, "pauses": [],
           "repeats": [], "program": None}
    mode = _W.get("display", "follower")
    if mode == "tracker":  # Whisper + tracker alone: its evidence position, on screen after the delay
        fups = [(t + _W["delay"], None if w is None else ix.dua_ids[ix.word_dua[w]],
                 None if w is None else int(ix.word_segment[w]), w, None, 0.0) for t, w, _ in anchors]
    elif mode == "stream":  # the stream decoder (stream_follower.py), anchored on the tracker
        fit["quiet"] = (z["t"], z["q"])
        fups = stream_updates(ix, fit)
    else:  # the phone: the CTC word follower anchored on the tracker (or on the truth: "oracle")
        fe._QUIET[(it["id"], 0.0, 0, 0, 0)] = (z["t"], z["q"])
        fups = fe.follow_updates(ix, fit, fcfg, _W["delay"], _W["f_delay"], oracle_anchor=mode == "oracle")
        fe._QUIET.pop((it["id"], 0.0, 0, 0, 0), None)
    ticks = np.arange(0.0, it["duration"], 0.1)
    out, k, cur = [], 0, None
    for t in ticks:
        while k < len(fups) and fups[k][0] <= t:
            _, dua, seg, w, _, _ = fups[k]
            cur = None if w is None else (dua, int(seg), int(w))
            k += 1
        out.append(cur)
    return out


def stream_updates(ix, fit: dict) -> list[tuple]:
    """follow_eval.follow_updates for the stream decoder: one step per CTC window."""
    from dua_recognition.stream_follower import StreamConfig, StreamFollower

    z = np.load(fit["ctc"])
    lp, nf, ts = z["lp"], z["n_frames"], z["t"]
    sf = StreamFollower(ix, _W.get("scfg") or StreamConfig())
    qt, qv = fit["quiet"]

    def quiet_at(t: float) -> float | None:
        i = int(np.searchsorted(qt, t, side="right")) - 1
        return float(qv[i]) if i >= 0 else None
    anchors = fit["anchors"]
    at = [a[0] + _W["delay"] for a in anchors]
    ups = []
    for k, t in enumerate(ts):
        j = bisect.bisect_right(at, t) - 1
        w_anchor = anchors[j][1] if j >= 0 else None
        lm = None
        if j >= 0 and len(anchors[j]) > 2 and anchors[j][2] is not None and w_anchor is not None:
            masses = anchors[j][2]
            first = sf._dua(int(ix.word_dua[w_anchor]))
            line0 = int(_line_no(ix)[first.lo])
            lm = lambda w, m=masses, f=line0: float(m[int(_line_no(ix)[w]) - f])  # noqa: E731
        w = sf.step(lp[k, : nf[k]].astype(np.float32), float(t), w_anchor, quiet_now=quiet_at(t), line_mass=lm,
                    anchor_t=at[j] if j >= 0 else None, quiet_then=quiet_at(t - sf.cfg.lookahead))
        if w is None:
            ups.append((t + _W["f_delay"], None, None, None, None, 0.0))
        else:
            ups.append((t + _W["f_delay"], ix.dua_ids[ix.word_dua[w]], int(ix.word_segment[w]), w, None, 0.0))
    return ups


_LINE_NO = {}


def _line_no(ix) -> np.ndarray:
    """Running line number over the corpus (follower.LocalFollower._line_no)."""
    if "x" not in _LINE_NO:
        new_line = np.r_[True, (np.diff(ix.word_segment) != 0) | (np.diff(ix.word_dua) != 0)]
        _LINE_NO["x"] = np.cumsum(new_line) - 1
    return _LINE_NO["x"]


def _code(dua: str | None, seg: int | None) -> str | None:
    return None if dua is None else f"{dua}#{seg}"


def metrics(it: dict, shown: list, ix) -> dict:
    """Reader-facing numbers for one item (sums; aggregate() turns them into rates)."""
    T = len(shown)
    ticks = np.arange(T) * 0.1
    gw = []  # truth words: (global word, start, end)
    for d, w, a, b in it["words"]:
        if d in ix.dua_ids:
            gw.append((ix.dua_word_span[ix.dua_ids.index(d)][0] + w, a, b))
    gw.sort(key=lambda x: x[1])
    starts = [x[1] for x in gw]
    still = it["still"]
    if gw and _W.get("same_text"):
        tw = [None if (k := bisect.bisect_right(starts, t) - 1) < 0 else gw[k][0] for t in ticks]
        shown = _same_text(shown, tw, ix, _W["same_text"])
    D = [None if s is None else _code(s[0], s[1]) for s in shown]
    m = {"items": 1, "minutes": it["duration"] / 60}
    if not gw:  # out of corpus: is anything shown?
        late = [d for t, d in zip(ticks, D) if t >= 5.0]
        m.update(ooc_ticks=len(late), ooc_shown_ticks=sum(d is not None for d in late))
        return m
    in_still = np.zeros(T, bool)
    for a, b, kind in still:
        in_still[int(a * 10) : int(b * 10) + 1] = True
    TL = [None] * T  # the reader's line per tick
    TW = [None] * T
    reading = np.zeros(T, bool)
    for i, t in enumerate(ticks):
        k = bisect.bisect_right(starts, t) - 1
        if k < 0:
            continue
        w, a, b = gw[k]
        TL[i] = _code(ix.dua_ids[ix.word_dua[w]], int(ix.word_segment[w]))
        TW[i] = w
        nxt = gw[k + 1][1] if k + 1 < len(gw) else b + 1.0
        if not in_still[i] and t <= gw[-1][2] + 1.0 and (t <= b + 2.5 or nxt - t < 0.5):
            reading[i] = True
    first = starts[0]
    found = next((i for i in range(T) if D[i] is not None and TL[i] is not None
                  and D[i].split("#")[0] == TL[i].split("#")[0]), None)
    m["found_n"] = 1
    m["found_10s"] = int(found is not None and ticks[found] - first <= 10.0)
    if _W.get("same_text"):  # the clock from the first words no other text reads (basmala, salawat first)
        k, text = _W["same_text"], [w.text for w in ix.words]
        own = [(w, a) for w, a, _ in gw if w - k + 1 >= ix.dua_word_span[ix.word_dua[w]][0]
               and tuple(text[w - k + 1 : w + 1]) not in _W["passages"]]
        if own:
            m["found_d_n"] = 1
            m["found_d10s"] = int(found is not None and ticks[found] - own[0][1] <= 10.0)
    m["found_s"] = float(ticks[found] - first) if found is not None else 60.0
    # on the line (1 s lag grace, 0.3 s early grace), after the du'a was first found
    ok = np.zeros(T, bool)
    for i in range(T):
        if D[i] is None:
            continue
        lo = max(0, i - 10)
        ok[i] = D[i] in TL[lo : i + 1] or (i + 3 < T and D[i] == TL[i + 3])
    rd = reading.copy()
    m["read_ticks"] = int(rd.sum())
    m["on_line_ticks"] = int((ok & rd).sum())
    m["shown_wrong_dua_ticks"] = int(sum(1 for i in range(T) if rd[i] and D[i] is not None and TL[i] is not None
                                         and D[i].split("#")[0] != TL[i].split("#")[0]))
    m["word_ticks"] = int(rd.sum())
    m["word_exact_ticks"] = int(sum(1 for i in range(T) if rd[i] and shown[i] is not None and shown[i][2] == TW[i]))
    # Moves away from the reader. A jump: from where the reader is (or just was) to a line 2+ away,
    # or another du'a, that the reader isn't on within 3 s before / 1 s after. Early: to the
    # neighbouring line before the reader gets there. Moves while already lost (stepping along on
    # the wrong repetition of a refrain, catching up late) are not new jumps: lost time counts them.
    jumps = early = far_moves = 0
    for i in range(1, T):
        if D[i] is None or D[i] == D[i - 1]:
            continue
        lo, hi = max(0, i - 30), min(T, i + 11)
        if D[i] in TL[lo:hi]:
            continue
        ref = TL[i] or next((x for x in TL[lo:hi] if x), None)
        if ref is None:
            continue
        dd, ds = D[i].split("#")
        rd_, rs = ref.split("#")
        far_ = dd != rd_ or abs(int(ds) - int(rs)) >= 2
        far_moves += far_
        was_ok = D[i - 1] is not None and (ok[i - 1] or D[i - 1] in TL[lo:hi])
        if was_ok and far_:
            jumps += 1
        elif was_ok:
            early += 1
    m["jumps"], m["early"], m["far_moves"] = jumps, early, far_moves
    # wrong place: reading time (after the du'a is found) with the highlight 2+ lines away or on another du'a
    wp = 0
    for i in range(found or T, T):
        if not rd[i] or D[i] is None or TL[i] is None or ok[i]:
            continue
        dd, ds = D[i].split("#")
        rd_, rs = TL[i].split("#")
        wp += dd != rd_ or abs(int(ds) - int(rs)) >= 2
    m["wrong_place_ticks"] = wp
    # lost episodes: 4+ s of reading ticks off the line, after the du'a was found
    lost, run = 0, 0
    for i in range(found or T, T):
        if not rd[i]:
            continue
        if ok[i]:
            if run >= 40:
                lost += 1
            run = 0
        else:
            run += 1
    lost += run >= 40
    m["lost"] = lost
    m["lost_ticks"] = int(sum(1 for i in range(found or T, T) if rd[i] and not ok[i]))
    m["after_found_ticks"] = int(sum(1 for i in range(found or T, T) if rd[i]))
    # events: time to the target line
    ev_lags = []
    evs = sorted(it["events"], key=lambda e: e["t"])
    for j, e in enumerate(evs):
        d, w = e["word"]
        if d not in ix.dua_ids:
            continue
        gwid = ix.dua_word_span[ix.dua_ids.index(d)][0] + w
        target = _code(d, int(ix.word_segment[gwid]))
        end = min(evs[j + 1]["t"] if j + 1 < len(evs) else it["duration"], e["t"] + 15.0)
        span = range(max(0, int((e["t"] - 0.5) * 10)), min(T, int(end * 10)))
        if e["kind"] in ("repeat", "restart"):  # the same line again: back at its first words?
            lag = next((ticks[i] - e["t"] for i in span if shown[i] is not None and gwid <= shown[i][2] <= gwid + 1),
                       None)
        else:
            lag = next((ticks[i] - e["t"] for i in span if D[i] == target), None)
        ev_lags.append((e["kind"], None if lag is None else float(lag)))
    m["events"] = ev_lags
    # still: does the highlight stay where it was when the reader stopped reading?
    st_ticks = st_ok = st_moves = 0
    for a, b, kind in still:
        if kind == "breath" or b - a < 1.0:
            continue
        i0, i1 = int(a * 10), min(T, int(b * 10))
        if i0 >= T or D[i0] is None:
            continue
        for i in range(i0 + 3, i1):
            st_ticks += 1
            st_ok += D[i] == D[i0]
            st_moves += D[i] != D[i - 1]
    m["still_ticks"], m["still_ok_ticks"], m["still_moves"] = st_ticks, st_ok, st_moves
    # line entry lag in ordinary reading (a word that follows the previous one in the text)
    lags = []
    for k in range(1, len(gw)):
        w, a, _ = gw[k]
        if w != gw[k - 1][0] + 1 or ix.word_segment[w] == ix.word_segment[gw[k - 1][0]]:
            continue
        code = _code(ix.dua_ids[ix.word_dua[w]], int(ix.word_segment[w]))
        i0 = int(max(0, a - 1.0) * 10)
        lag = next((ticks[i] - a for i in range(i0, min(T, int((a + 6.0) * 10))) if D[i] == code
                    and (i == 0 or D[i - 1] != code)), None)
        if lag is not None:
            lags.append(float(lag))
    m["entry_lags"] = lags
    return m


def _score_one(it: dict) -> tuple[str, dict] | None:
    shown = replay(it)
    if shown is None:
        return None
    m = metrics(it, shown, _W["ix"])
    m.update(scenario=it["scenario"], lane=it["lane"], voice=it["voice"], rate=it["tags"].get("rate"),
             split=split_of(it["voice"]), dua=it["dua"])
    return it["id"], m


def aggregate(ms: list[dict]) -> dict:
    s = lambda k: sum(m.get(k, 0) for m in ms)  # noqa: E731
    minutes = s("minutes")
    evs = [lag for m in ms for _, lag in m.get("events", [])]
    lags = [x for m in ms for x in m.get("entry_lags", [])]
    out = {"items": len(ms), "minutes": round(minutes, 1), "voices": len({m["voice"] for m in ms})}
    if s("read_ticks"):
        out["on_line"] = s("on_line_ticks") / s("read_ticks")
        out["word_exact"] = s("word_exact_ticks") / max(1, s("word_ticks"))
        out["wrong_dua"] = s("shown_wrong_dua_ticks") / s("read_ticks")
        out["jumps_10"] = s("jumps") / minutes * 10
        out["early_10"] = s("early") / minutes * 10
        out["wrong_place"] = s("wrong_place_ticks") / max(1, s("after_found_ticks"))
        out["lost_10"] = s("lost") / minutes * 10
        out["lost_share"] = s("lost_ticks") / max(1, s("after_found_ticks"))
        out["found_10s"] = s("found_10s") / max(1, s("found_n"))
        if s("found_d_n"):
            out["found_d10s"] = s("found_d10s") / s("found_d_n")
        out["found_med"] = float(np.median([m["found_s"] for m in ms if "found_s" in m]))
        out["entry_lag"] = float(np.median(lags)) if lags else float("nan")
    if evs:
        out["event_3s"] = float(np.mean([x is not None and x <= 3.0 for x in evs]))
        out["event_found"] = float(np.mean([x is not None for x in evs]))
        f = [x for x in evs if x is not None]
        out["event_med"] = float(np.median(f)) if f else float("nan")
        out["n_events"] = len(evs)
    if s("still_ticks"):
        out["stay"] = s("still_ok_ticks") / s("still_ticks")
    if s("ooc_ticks"):
        out["ooc_shown"] = s("ooc_shown_ticks") / s("ooc_ticks")
    return out


def passes(agg: dict) -> dict:
    out = {}
    for k, (op, v) in BARS.items():
        if k in agg and agg[k] == agg[k]:
            out[k] = agg[k] >= v if op == ">=" else agg[k] <= v
    return out


COLS = [("on_line", "on line", "{:.0%}"), ("word_exact", "word", "{:.0%}"), ("jumps_10", "jumps/10m", "{:.2f}"),
        ("early_10", "early/10m", "{:.2f}"), ("lost_10", "lost/10m", "{:.2f}"), ("wrong_place", "wrong place", "{:.0%}"),
        ("event_3s", "follows≤3s", "{:.0%}"), ("event_med", "follow med", "{:.1f}s"), ("stay", "stays put", "{:.0%}"),
        ("found_10s", "found≤10s", "{:.0%}"), ("found_d10s", "found≤10s*", "{:.0%}"),
        ("found_med", "found med", "{:.1f}s"), ("wrong_dua", "wrong du'a", "{:.1%}"),
        ("entry_lag", "line lag", "{:+.2f}s"), ("ooc_shown", "ooc shown", "{:.0%}")]


def table(rows: list[tuple[str, dict]], ref: dict | None = None) -> str:
    head = "| cell | items | min | voices | " + " | ".join(c[1] for c in COLS) + " |"
    lines = [head, "|" + "---|" * (4 + len(COLS))]
    for name, agg in rows:
        ok = passes(agg)
        cells = []
        for k, _, f in COLS:
            if k not in agg or agg[k] != agg[k]:
                cells.append("")
                continue
            txt = f.format(agg[k])
            if ref and name in ref and k in ref[name] and ref[name][k] == ref[name][k]:
                dlt = agg[k] - ref[name][k]
                if "%" in f and abs(dlt) >= 0.005:
                    txt += f" ({dlt * 100:+.0f})"
                elif "%" not in f and abs(dlt) >= 0.005:
                    txt += f" ({dlt:+.2f})"
            if k in ok and not ok[k]:
                txt = f"**{txt}** ✗"
            cells.append(txt)
        lines.append(f"| {name} | {agg['items']} | {agg['minutes']:.0f} | {agg['voices']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_score(args) -> None:
    from multiprocessing import Pool

    items = load_items(args.scenarios, args.split)
    if args.lane:
        items = [it for it in items if it["lane"] == args.lane]
    import evaluate as ev

    have = set(ev.load_all())  # (DUA_CORPUS_DIR: an older corpus, without the texts added since)
    n = len(items)
    items = [it for it in items if all(w[0] in have for w in it["words"])]
    if len(items) < n:
        print(f"{n - len(items)} items left out: their du'a isn't in this corpus", flush=True)
    cfg_kw ={k: json.loads(v) for k, v in (a.split("=") for a in args.tracker.split(",") if a)}
    fw_kw = {}
    for a in (x for x in args.fw.split(",") if x):
        k, v = a.split("=")
        fw_kw[k] = json.loads(v.lower()) if v.lower() in ("true", "false") else (int(v) if v.isdigit() else float(v))
    asr_tag, ctc_tag = args.asr, args.ctc
    t0 = time.time()
    sc_kw = {}
    for a in (x for x in args.sc.split(",") if x):
        k, v = a.split("=")
        sc_kw[k] = tuple(float(x) for x in v.split(";")) if ";" in v else             (int(v) if k in ("line_steps", "next_steps") else float(v))
    init = (cfg_kw, fw_kw, asr_tag, ctc_tag, args.delay, args.follow_delay, args.display, sc_kw, args.same_text)
    results = {}
    with Pool(args.workers, initializer=_worker_init, initargs=init) as pool:
        for r in pool.imap_unordered(_score_one, items, chunksize=1):
            if r is not None:
                results[r[0]] = r[1]
    print(f"scored {len(results)} of {len(items)} items in {time.time() - t0:.0f} s "
          f"(tracker {cfg_kw or 'defaults'}, follower {fw_kw or 'defaults'})", flush=True)
    out = BENCH / "results" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "items": results}, ensure_ascii=False), encoding="utf-8")
    report(results, args.compare)


def report(results: dict, compare: str | None = None, split: str = "all") -> None:
    if split != "all":
        results = {k: v for k, v in results.items() if v.get("split") == split}
    ref_rows = None
    if compare:
        ref = json.loads((BENCH / "results" / f"{compare}.json").read_text(encoding="utf-8"))["items"]
        ref = {k: v for k, v in ref.items() if k in results}
        ref_rows = dict(grid_rows(ref))
    print(table(grid_rows(results), ref_rows))


def grid_rows(results: dict) -> list[tuple[str, dict]]:
    by_sc, by_lane = {}, {}
    for m in results.values():
        by_sc.setdefault(m["scenario"], []).append(m)
        by_lane.setdefault(m["lane"], []).append(m)
    rows = [(sc, aggregate(by_sc[sc])) for sc in SCENARIOS if sc in by_sc]
    rows += [(f"lane: {ln}", aggregate(ms)) for ln, ms in sorted(by_lane.items())]
    rows.append(("all", aggregate(list(results.values()))))
    return rows


def cmd_report(args) -> None:
    res = json.loads((BENCH / "results" / f"{args.name}.json").read_text(encoding="utf-8"))["items"]
    report(res, args.compare, args.split)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    src = sub.add_parser("sources")
    src.add_argument("--lane", choices=["mafatih", "ooc"], help="rebuild just these sources (ooc: out of corpus)")
    b = sub.add_parser("build")
    b.add_argument("scenarios", nargs="*")
    b.add_argument("--harvest-per-cell", type=int, default=30)
    b.add_argument("--rebuild", action="store_true")
    a = sub.add_parser("asr")
    a.add_argument("scenarios", nargs="*")
    a.add_argument("--model", default="models/whisper-base-syn-v5-ctx8ft-ct2")
    a.add_argument("--ctc-model", default="models/ctc-student-base-v6")
    a.add_argument("--device", default="cuda")
    a.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("scenarios", nargs="*")
    s.add_argument("--name", required=True)
    s.add_argument("--asr", default=ASR_TAG)
    s.add_argument("--ctc", default=CTC_TAG)
    s.add_argument("--tracker", default="", help="TrackerConfig overrides k=v,...")
    s.add_argument("--fw", default="", help="FollowerConfig overrides k=v,...")
    s.add_argument("--sc", default="", help="StreamConfig overrides k=v,... (tuples as a;b;c)")
    s.add_argument("--delay", type=float, default=1.2, help="Whisper result on screen this long after its window")
    s.add_argument("--follow-delay", type=float, default=0.15)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--compare", help="a previous --name to diff against")
    s.add_argument("--split", choices=["dev", "test", "all"], default="all", help="voices: tune on dev, report test")
    s.add_argument("--lane", help="only this lane (studio, majlis, harvest, user)")
    s.add_argument("--same-text", type=int, default=0, metavar="K",
                   help="count the display right where it shows the same K words as the reader's du'a there "
                        "(a passage several texts share); 0 = only the reader's own du'a counts")
    s.add_argument("--display", choices=["follower", "tracker", "oracle", "stream"], default="follower",
                   help="what drives the highlight: the phone's follower (default), the tracker alone, or the "
                        "follower anchored on the true word (a perfect tracker)")
    r = sub.add_parser("report")
    r.add_argument("--name", required=True)
    r.add_argument("--compare")
    r.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"sources": cmd_sources, "build": cmd_build, "asr": cmd_asr, "score": cmd_score,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
