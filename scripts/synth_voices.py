#!/usr/bin/env python
"""Synthetic ordinary voices reading the du'as, for the phone model's training set.

The phone model learned from professional reciters and does worst on ordinary
voices: the people who actually use the app (RetaSy crowd CER 29.9%; Hasan's own
sessions 40-50% against the server model). OmniVoice (k2-fsa, Apache-2.0,
zero-shot, 600+ languages) clones a voice from a few seconds of audio. Here it
clones the crowd-sourced RetaSy volunteers already in the training set
(data/cache/finetune/crowd.jsonl: 339 of them, none with an annotated test clip,
so voice_eval.py's speakers stay unseen) reading spans of the vowelled du'a
texts at an unhurried range of speeds. Tawassul and Ashura stay out, as for the
real windows (build_finetune_set.HELD_OUT_DUAS).

    <tts venv>/python scripts/synth_voices.py generate --n 12000   # GPU, resumable, ~2 h
    python scripts/synth_voices.py check                           # round trip through turbo-ft (GPU)
    python scripts/synth_voices.py build                           # -> data/cache/finetune/synth.jsonl

The TTS runs in its own venv (omnivoice wants transformers >= 5.3 and pulls in
gradio; see docs/results/synthetic_voices.md). Rows look like crowd.jsonl's, so
finetune_whisper.py --synth mixes them in like --crowd.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OUT = ROOT / "data" / "cache" / "synth"
CLIPS = OUT / "clips"
FINETUNE = ROOT / "data" / "cache" / "finetune"
SR = 16000
# Quranic pause and annotation marks: not spoken, and they only confuse a TTS text front end.
_MARKS = re.compile("[ۖ-ۭ؀-؅۝۞]")


AGES = (("child", 0.15), ("teenager", 0.15), ("young adult", 0.3), ("middle-aged", 0.25), ("elderly", 0.15))
PITCHES = ("very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch")
ACCENTS = ((None, 0.5), ("indian accent", 0.2), ("british accent", 0.1), ("american accent", 0.1),
           ("russian accent", 0.05), ("chinese accent", 0.05))


def _pick(rng, weighted):
    x, acc = rng.random(), 0.0
    for item, w in weighted:
        acc += w
        if x < acc:
            return item
    return weighted[-1][0]


def design(rng) -> tuple[str, str]:
    """A voice from attributes (OmniVoice voice design): gender, age, pitch, sometimes an accent."""
    gender = rng.choice(("male", "female"))
    parts = [gender, _pick(rng, AGES), rng.choice(PITCHES)]
    accent = _pick(rng, ACCENTS)
    if accent:
        parts.append(accent)
    return ", ".join(parts), gender


def plan(n: int, seed: int, designed: float = 0.0) -> list[dict]:
    """n spans of 4-16 consecutive words, du'a chosen uniformly, voice and speed at random.
    `designed`: this share of the voices made from attributes instead of cloned."""
    from build_finetune_set import HELD_OUT_DUAS, whisper_style

    from dua_recognition.corpus import load_all

    rng = random.Random(seed)
    duas = load_all()
    texts = {}
    for d in duas.values():
        if d.id in HELD_OUT_DUAS:
            continue
        tokens = [_MARKS.sub("", t) for s in d.segments for t in s.arabic.split()]
        tokens = [t for t in tokens if whisper_style(t)]
        if len(tokens) >= 8:
            texts[d.id] = tokens
    voices = load_voices()
    ids = sorted(texts)
    out = []
    for k in range(n):
        dua = rng.choice(ids)
        tokens = texts[dua]
        span = rng.randint(4, 16)
        i = rng.randrange(0, max(1, len(tokens) - span))
        words = tokens[i : i + span]
        v = rng.choice(voices)
        item = {"id": f"s{seed}-{k:06d}", "dua": dua, "word": i, "tts": " ".join(words),
                "text": " ".join(whisper_style(t) for t in words), "voice": v["reciter"],
                "gender": v["gender"], "speed": round(rng.uniform(0.75, 1.1), 2)}
        if designed and rng.random() < designed:  # (no draw at all when designed == 0: seed 0's plan stays as it was)
            item["instruct"], item["gender"] = design(rng)
            item["voice"] = None
        out.append(item)
    return out


def load_voices() -> list[dict]:
    """One reference clip per crowd volunteer: their longest between 2 and 8 s."""
    rows = [json.loads(line) for line in (FINETUNE / "crowd.jsonl").read_text(encoding="utf-8").splitlines() if line]
    best: dict[str, dict] = {}
    for r in rows:
        n = np.load(ROOT / r["clip"], mmap_mode="r").shape[0] / SR
        if 2.0 <= n <= 8.0 and n > best.get(r["reciter"], {}).get("secs", 0):
            best[r["reciter"]] = {**r, "secs": n}
    return [best[k] for k in sorted(best)]


def generate(args) -> None:
    import torch
    from omnivoice import OmniVoice

    items = plan(args.n, args.seed, args.designed)
    CLIPS.mkdir(parents=True, exist_ok=True)
    (OUT / f"plan_s{args.seed}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in items),
                                                  encoding="utf-8")
    todo = [r for r in items if not (CLIPS / f"{r['id']}.npy").exists()]
    print(f"{len(items)} planned, {len(todo)} to generate", flush=True)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    sr_out = model.sampling_rate
    voices = {v["reciter"]: v for v in load_voices()}
    prompts = {}

    def prompt(name):
        if name not in prompts:
            v = voices[name]
            wav = torch.from_numpy(np.load(ROOT / v["clip"]).astype(np.float32) / 32767)
            prompts[name] = model.create_voice_clone_prompt((wav.unsqueeze(0), SR), ref_text=v["text"])
        return prompts[name]

    import torchaudio.functional as AF

    t0, done, secs = time.time(), 0, 0.0
    # Cloned and designed voices batch separately; similar lengths batch together.
    todo.sort(key=lambda r: (bool(r.get("instruct")), len(r["tts"])))
    batches = [todo[b : b + args.batch] for b in range(0, len(todo), args.batch)]
    batches = [[r for r in bt if bool(r.get("instruct")) == kind] for bt in batches for kind in (False, True)]
    for b, batch in enumerate(x for x in batches if x):
        voice = ({"instruct": [r["instruct"] for r in batch]} if batch[0].get("instruct") else
                 {"voice_clone_prompt": [prompt(r["voice"]) for r in batch]})
        try:
            audios = model.generate(text=[r["tts"] for r in batch], language="arb", **voice,
                                    speed=[r["speed"] for r in batch], num_step=args.steps)
        except Exception as e:  # one bad prompt shouldn't stop the night
            print(f"batch {b}: {type(e).__name__}: {e}", flush=True)
            continue
        for r, a in zip(batch, audios):
            y = AF.resample(torch.from_numpy(np.asarray(a, dtype=np.float32)), sr_out, SR).numpy()
            np.save(CLIPS / f"{r['id']}.npy", (np.clip(y, -1, 1) * 32767).astype(np.int16))
            secs += len(y) / SR
        done += len(batch)
        if b % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(todo)}  {secs / 3600:.2f} h audio in {el / 60:.1f} min (RTF {el / max(secs, 1):.3f})",
                  flush=True)
    print(f"done: {done} clips, {secs / 3600:.2f} h", flush=True)


def check(args) -> None:
    """Round trip: turbo-ft transcribes every clip; its CER against the label says
    whether the TTS said what it was asked to."""
    import jiwer

    from dua_recognition.asr import transcribe_batch
    from dua_recognition.text import normalize

    items = [json.loads(line) for p in sorted(OUT.glob("plan_s*.jsonl"))
             for line in p.read_text(encoding="utf-8").splitlines() if line]
    items = [r for r in items if (CLIPS / f"{r['id']}.npy").exists()]
    done_path = OUT / "check.jsonl"
    done = {json.loads(line)["id"] for line in done_path.read_text(encoding="utf-8").splitlines()} if done_path.exists() else set()
    todo = [r for r in items if r["id"] not in done]
    print(f"{len(items)} clips, {len(todo)} to check", flush=True)
    with done_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), 16):  # turbo at 64 filled the 16 GB card and spilled into RAM
            batch = todo[b : b + 16]
            wavs = [np.load(CLIPS / f"{r['id']}.npy").astype(np.float32) / 32767 for r in batch]
            hyps = transcribe_batch(wavs, model=args.model)
            for r, w, h in zip(batch, wavs, hyps):
                ref, hyp = normalize(r["text"]).replace(" ", ""), normalize(h).replace(" ", "")
                c = jiwer.cer(ref, hyp or "-") if ref else 1.0
                f.write(json.dumps({"id": r["id"], "cer": round(c, 4), "secs": round(len(w) / SR, 2), "hyp": h},
                                   ensure_ascii=False) + "\n")
            if b % 1600 == 0:
                print(f"  {b + len(batch)}/{len(todo)}", flush=True)


def build(args) -> None:
    """--seeds: which generation rounds to include (plan_s<seed>.jsonl); --name: the set's file name."""
    items = {json.loads(line)["id"]: json.loads(line) for s in args.seeds
             for line in (OUT / f"plan_s{s}.jsonl").read_text(encoding="utf-8").splitlines() if line}
    checked = [json.loads(line) for line in (OUT / "check.jsonl").read_text(encoding="utf-8").splitlines() if line]
    checked = [c for c in checked if c["id"] in items]
    kept = [c for c in checked if c["cer"] <= args.max_cer and 1.0 <= c["secs"] <= 12.0]

    def voice(it):
        return f"synth-design:{it['instruct']}" if it.get("instruct") else f"synth:{it['voice']}"

    rows = [{"clip": str((CLIPS / f"{c['id']}.npy").relative_to(ROOT)), "text": items[c["id"]]["text"],
             "dua": items[c["id"]]["dua"], "reciter": voice(items[c["id"]]),
             "gender": items[c["id"]]["gender"]} for c in kept]
    out = FINETUNE / f"{args.name}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    cers = np.array([c["cer"] for c in checked])
    designed = [c for c in kept if items[c["id"]].get("instruct")]
    print(f"{len(checked)} checked: CER median {np.median(cers):.1%}, <= {args.max_cer:.0%}: {len(kept)} "
          f"({sum(c['secs'] for c in kept) / 3600:.1f} h, {len({r['reciter'] for r in rows})} voices, "
          f"{len({r['dua'] for r in rows})} du'as; {len(designed)} designed voices) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=12000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=4, help="bigger batches were slower per second of audio")
    g.add_argument("--steps", type=int, default=32, help="OmniVoice decoding steps")
    g.add_argument("--designed", type=float, default=0.0,
                   help="share of voices made from attributes (gender, age, pitch, accent) instead of cloned")
    c = sub.add_parser("check")
    c.add_argument("--model", default=str(ROOT / "models" / "whisper-turbo-dua-ct2"))
    b = sub.add_parser("build")
    b.add_argument("--max-cer", type=float, default=0.3)
    b.add_argument("--seeds", type=int, nargs="+", default=[0], help="generation rounds to include")
    b.add_argument("--name", default="synth", help="-> data/cache/finetune/<name>.jsonl")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"generate": generate, "check": check, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
