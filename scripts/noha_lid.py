#!/usr/bin/env python
"""How well does Whisper's own language ID tell Urdu / Arabic / Farsi / English nohas apart?

A feasibility check for a noha mode that detects the language itself. Nohas
are sung over matam and chorus, so Whisper's language token, which was
trained on speech, may not transfer. This measures it the way the live app
would use it: a listener opens the app mid-noha (cold start), a 6 s window
every 1 s, silent windows skipped by the same VAD rule as the tracker, and the
language decided by summing the language log-probabilities over the windows
heard so far, restricted to the four candidates (Hindi counted as Urdu: they
sound the same and Whisper confuses them).

    python scripts/noha_lid.py fetch --per-lang 20     # YouTube -> data/noha/<lang>/
    python scripts/noha_lid.py eval --models base small large-v3-turbo large-v3

Labels are the language of the search that found each video (reciter + "noha"
/ "لطمية" / "نوحه" / "english noha"); `eval --show` prints a transcript
snippet per video so they can be checked by eye. Audio stays in the ignored
data/noha/ cache.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "noha"
sys.stdout.reconfigure(encoding="utf-8")
YTDLP = str(Path(sys.executable).with_name("yt-dlp"))
LANGS = ("ur", "ar", "fa", "en")
ALIAS = {"hi": "ur"}  # Whisper's Hindi token is spoken Urdu here

QUERIES = {
    "ur": ["Nadeem Sarwar noha", "Mir Hasan Mir noha", "Farhan Ali Waris noha", "Ali Shanawar noha",
           "Shadman Raza noha", "Hasan Sadiq noha", "Syed Raza Abbas Zaidi noha", "Ali Safdar noha",
           "Irfan Haider noha", "Mesum Abbas noha", "Anjuman noha matam"],
    "ar": ["باسم الكربلائي لطمية", "جليل الكربلائي لطمية", "محمد باقر الخاقاني لطمية", "علي حمادي لطمية",
           "عمار الكناني لطمية", "حسين فيصل لطمية", "سيد فاقد الموسوي لطمية", "حمزة الصغير قصيدة حسينية",
           "مسلم الوائلي لطمية", "محمد الجنامي لطمية", "لطمية حسينية ردات"],
    "fa": ["نوحه محمود کریمی", "نوحه عبدالرضا هلالی", "نوحه حسین طاهری", "نوحه میثم مطیعی",
           "نوحه حمید علیمی", "نوحه سید مجید بنی فاطمه", "نوحه محمدحسین پویانفر", "نوحه مهدی رسولی",
           "نوحه حنیف طاهری", "نوحه سینه زنی محرم"],
    "en": ["english noha", "english latmiya", "english nauha muharram", "noha in english imam hussain",
           "english latmiya karbala", "english noha abbas", "english noha sakina", "english majlis latmiya"],
}
# A title naming another language ("... | Noha | Farsi") is a mislabel waiting to happen.
LANG_WORDS = {"ur": ["urdu", "اردو"], "ar": ["arabic", "عربي", "عربى"], "fa": ["farsi", "persian", "فارسی"],
              "en": ["english"]}
NOT_A_NOHA = ["lecture", "majlis by", "speech", "bayan", "shorts", "#shorts", "tafseer", "reaction",
              "محاضرة", "مجلس", "سخنرانی", "tutorial", "karaoke", "instrumental", "live stream"]


# -- fetch ------------------------------------------------------------------
def search(query: str, n: int) -> list[dict]:
    out = subprocess.run([YTDLP, f"ytsearch{n}:{query}", "--flat-playlist", "--dump-json", "--no-warnings"],
                         capture_output=True, text=True, encoding="utf-8", timeout=300)
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def acceptable(v: dict, lang: str) -> bool:
    title = f"{v.get('title', '')} {v.get('channel', '')}".lower()
    if any(w in title for w in NOT_A_NOHA):
        return False
    if any(w in title for other, ws in LANG_WORDS.items() if other != lang for w in ws):
        return False
    if lang == "en" and "english" not in title:
        return False  # "english noha" searches also return Urdu nohas
    return 120 <= (v.get("duration") or 0) <= 1200


def download(v: dict, lang: str, query: str) -> bool:
    out_dir = OUT / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{v['id']}.json"
    if meta_path.exists():
        return True
    r = subprocess.run([YTDLP, "-f", "bestaudio[abr<=96]/bestaudio", "--no-playlist", "--no-warnings",
                        "-o", str(out_dir / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={v['id']}"],
                       capture_output=True, text=True, encoding="utf-8", timeout=900)
    audio = [p for p in out_dir.glob(f"{v['id']}.*") if p.suffix not in (".json", ".part")]
    if r.returncode != 0 or not audio:
        print(f"    download failed: {v['id']} {r.stderr.strip()[-200:]}")
        return False
    meta_path.write_text(json.dumps({
        "video_id": v["id"], "url": f"https://www.youtube.com/watch?v={v['id']}", "lang": lang,
        "query": query, "title": v.get("title"), "channel": v.get("channel") or v.get("uploader"),
        "duration_s": v.get("duration"), "audio": audio[0].name,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def fetch(per_lang: int, per_query: int, per_channel: int) -> None:
    for lang, queries in QUERIES.items():
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in (OUT / lang).glob("*.json")] \
            if (OUT / lang).exists() else []
        channels = defaultdict(int)
        for m in metas:
            channels[m["channel"]] += 1
        kept = len(metas)
        for q in queries:
            got = 0
            for v in search(q, 15):
                if kept >= per_lang or got >= per_query:
                    break
                ch = v.get("channel") or v.get("uploader")
                if channels[ch] >= per_channel or not acceptable(v, lang):
                    continue
                if (OUT / lang / f"{v['id']}.json").exists():
                    continue
                print(f"  {lang}: {v.get('title', '')[:70]} [{ch}] ({v.get('duration')} s)", flush=True)
                if download(v, lang, q):
                    kept, got = kept + 1, got + 1
                    channels[ch] += 1
        print(f"{lang}: {kept} recordings", flush=True)


# -- eval -------------------------------------------------------------------
def load_audio(path: Path) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=16000)


def lid_logprobs(model, windows: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """log P(lang | window) for each window over LANGS (Hindi merged into Urdu),
    renormalized over the four, and Whisper's unrestricted top language per
    window. One encoder pass per window, batched."""
    from faster_whisper.audio import pad_or_trim

    feats = np.stack([pad_or_trim(model.feature_extractor(w)) for w in windows])
    out = np.full((len(windows), len(LANGS)), -np.inf)
    raw = []
    for lo in range(0, len(windows), 16):
        enc = model.encode(feats[lo : lo + 16])
        for i, res in enumerate(model.model.detect_language(enc), start=lo):
            p = np.zeros(len(LANGS))
            raw.append(res[0][0][2:-2])
            for token, prob in res:
                code = ALIAS.get(token[2:-2], token[2:-2])
                if code in LANGS:
                    p[LANGS.index(code)] += prob
            out[i] = np.log(np.maximum(p / max(p.sum(), 1e-12), 1e-12))
    return out, raw


@lru_cache(maxsize=1)
def _ecapa():
    import torch
    import speechbrain

    # speechbrain 1.1 leaves lazy/deprecated module stubs in sys.modules;
    # inspect (via hyperpyyaml) touches them, which tries to import k2 and fails.
    for name in [n for n, m in sys.modules.items() if type(m).__module__.startswith("speechbrain.utils")]:
        del sys.modules[name]
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    clf = EncoderClassifier.from_hparams("speechbrain/lang-id-voxlingua107-ecapa",
                                         savedir=str(ROOT / "models" / "lang-id-voxlingua107-ecapa"),
                                         run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                         local_strategy=LocalStrategy.COPY)  # symlinks need admin on Windows
    labels = [clf.hparams.label_encoder.decode_ndim(i).split(":")[0] for i in range(107)]
    return clf, labels


def ecapa_logprobs(windows: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Same as lid_logprobs, from the VoxLingua107 ECAPA classifier (a
    dedicated LID model, ~20 M parameters) instead of Whisper."""
    import torch

    clf, labels = _ecapa()
    n = max(w.size for w in windows)
    wavs = torch.zeros(len(windows), n)
    for i, w in enumerate(windows):
        wavs[i, : w.size] = torch.from_numpy(w)
    lens = torch.tensor([w.size / n for w in windows])
    with torch.no_grad():
        logp = clf.classify_batch(wavs, lens)[0].cpu().numpy()
    probs = np.exp(logp - logp.max(axis=1, keepdims=True))
    out = np.zeros((len(windows), len(LANGS)))
    for j, code in enumerate(labels):
        code = ALIAS.get(code, code)
        if code in LANGS:
            out[:, LANGS.index(code)] += probs[:, j]
    out = np.log(np.maximum(out / out.sum(axis=1, keepdims=True), 1e-12))
    return out, [ALIAS.get(labels[j], labels[j]) for j in logp.argmax(axis=1)]


def evaluate(models: list[str], starts: tuple[float, ...], horizon: int, show: bool,
             out: Path = ROOT / "docs" / "results" / "noha_lid.json") -> None:
    from dua_recognition.asr import load_model, speech_in_tail

    items = []
    for lang in LANGS:
        for meta_path in sorted((OUT / lang).glob("*.json")):
            if meta_path.name.count(".") > 1:
                continue  # <id>.lyrics.json sidecars (scripts/noha_match.py)
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            if m.get("exclude"):
                continue
            items.append((m, meta_path.with_name(m["audio"])))
    print(f"{len(items)} recordings: " + ", ".join(f"{l} {sum(m['lang'] == l for m, _ in items)}" for l in LANGS))

    # Cold starts at several points of each noha; windows end 1..horizon s later.
    sr = 16000
    trials = []  # (item index, start s, [windows], [speech?])
    for k, (m, path) in enumerate(items):
        audio = load_audio(path)
        dur = audio.size / sr
        for frac in starts:
            s = frac * dur
            wins = [audio[int(max(s, s + t - 6) * sr) : int((s + t) * sr)] for t in range(1, horizon + 1)]
            trials.append((k, s, wins, [speech_in_tail(w) for w in wins]))

    results = {}
    for name in models:
        model = None if name == "ecapa" else load_model(name)
        per_trial = []
        raw_top = defaultdict(Counter)
        for k, s, wins, speech in trials:
            lp, top = ecapa_logprobs(wins) if model is None else lid_logprobs(model, wins)
            per_trial.append((k, lp, np.array(speech)))
            for code, sp in zip(top, speech):
                if sp:
                    raw_top[items[k][0]["lang"]][code] += 1
        results[name] = per_trial
        report(name, items, per_trial, horizon)
        print("unrestricted top-1 on speech windows, per true language:")
        for lang, counts in raw_top.items():
            total = sum(counts.values())
            print(f"  {lang}: " + ", ".join(f"{c} {n / total:.0%}" for c, n in counts.most_common(6)))
        if show and model is not None:
            show_snippets(model, items)
    out.write_text(json.dumps({
        "items": [{"lang": m["lang"], "url": m["url"], "title": m["title"], "channel": m["channel"]}
                  for m, _ in items],
        "starts": starts,
        "results": {name: [{"item": k, "logprobs": lp.round(3).tolist(), "speech": sp.tolist()}
                           for k, lp, sp in pt] for name, pt in results.items()},
    }, ensure_ascii=False), encoding="utf-8")


def decide(lp: np.ndarray, speech: np.ndarray, t: int, single: bool = False) -> int | None:
    """Language after t seconds: summed log-probs of the speech windows so far
    (or only the latest speech window if `single`). None = nothing heard yet."""
    idx = np.flatnonzero(speech[:t])
    if not idx.size:
        return None
    return int(lp[idx[-1:] if single else idx].sum(axis=0).argmax())


def report(name: str, items, per_trial, horizon: int) -> None:
    print(f"\n== {name} ==")
    ts = [t for t in (2, 3, 5, 10, horizon) if t <= horizon]
    ts = sorted(set(ts))
    print("accuracy after t s (cumulative | latest window only), per true language; undecided counts as wrong")
    print("lang   n   " + "  ".join(f"@{t:>2}s cum/one " for t in ts))
    for li, lang in enumerate((*LANGS, "all")):
        rows = [(k, lp, sp) for k, lp, sp in per_trial if lang == "all" or items[k][0]["lang"] == lang]
        if not rows:
            continue
        cells = []
        for t in ts:
            cum = np.mean([decide(lp, sp, t) == LANGS.index(items[k][0]["lang"]) for k, lp, sp in rows])
            one = np.mean([decide(lp, sp, t, single=True) == LANGS.index(items[k][0]["lang"]) for k, lp, sp in rows])
            cells.append(f"{cum:6.0%} / {one:4.0%}  ")
        print(f"{lang:4} {len(rows):4}  " + "  ".join(cells))
    # Confusions at the horizon, cumulative.
    conf = np.zeros((len(LANGS), len(LANGS) + 1), dtype=int)
    for k, lp, sp in per_trial:
        d = decide(lp, sp, horizon)
        conf[LANGS.index(items[k][0]["lang"]), len(LANGS) if d is None else d] += 1
    print(f"confusion @{horizon}s (rows true, cols predicted, last = no speech yet)")
    print("      " + " ".join(f"{c:>4}" for c in (*LANGS, "-")))
    for i, lang in enumerate(LANGS):
        print(f"  {lang}  " + " ".join(f"{x:>4}" for x in conf[i]))
    # Per-recording accuracy at the horizon, worst first, to spot label errors.
    per_item = defaultdict(list)
    for k, lp, sp in per_trial:
        per_item[k].append(decide(lp, sp, horizon) == LANGS.index(items[k][0]["lang"]))
    bad = sorted((np.mean(v), k) for k, v in per_item.items() if np.mean(v) < 1)
    for acc, k in bad:
        m = items[k][0]
        print(f"  {acc:4.0%} {m['lang']} {m['video_id']} {m['title'][:60]} [{m['channel']}]")


def show_snippets(model, items) -> None:
    """A transcript of 20 s from the middle of each recording, in its labelled
    language, to eyeball the labels."""
    for m, path in items:
        audio = load_audio(path)
        mid = audio.size // 2
        segs, _ = model.transcribe(audio[mid : mid + 20 * 16000], language=m["lang"], beam_size=1)
        text = " ".join(s.text.strip() for s in segs)
        print(f"  {m['lang']} {m['video_id']}: {re.sub(r'\s+', ' ', text)[:120]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--per-lang", type=int, default=20)
    f.add_argument("--per-query", type=int, default=2)
    f.add_argument("--per-channel", type=int, default=3)
    e = sub.add_parser("eval")
    e.add_argument("--models", nargs="+", default=["base", "small", "large-v3-turbo"])
    e.add_argument("--starts", nargs="+", type=float, default=[0.25, 0.45, 0.65, 0.85])
    e.add_argument("--horizon", type=int, default=10)
    e.add_argument("--show", action="store_true", help="print a transcript snippet per recording")
    e.add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "noha_lid.json")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.per_lang, args.per_query, args.per_channel)
    else:
        evaluate(args.models, tuple(args.starts), args.horizon, args.show, args.out)


if __name__ == "__main__":
    main()
