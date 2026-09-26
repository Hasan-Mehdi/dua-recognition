#!/usr/bin/env python
"""Fine-tune a multilingual Whisper on nohas with lyrics-verified labels.

The du'a model went from 47% to 85% line accuracy by fine-tuning a small
Whisper on reference-snapped labels. This is the noha version, without human
line timings: the training recordings are YouTube uploads whose descriptions
carry the lyrics (the "distractors" of scripts/noha_match.py), from channels
that never appear in the test set.

  fetch   download their audio (data/noha/train/<lang>/)
  label   6 s windows (3 s hop), transcribed by a teacher (large-v3-turbo) in the
          recording's language; a window is kept only if its transcript fits
          the recording's own lyrics (per-letter skeleton cost <= --max-cost).
          The label is the matching stretch of the lyrics when they're written
          in the language's own script (so spelling comes from the lyrics, not
          the teacher), else the teacher's transcript (Roman-Urdu lyrics can
          verify an Urdu transcript but not spell it).
  train   whisper-small, all weights, per-row language token; exported to
          CTranslate2 at models/<name>-ct2 for noha_match.py transcribe.

    python scripts/noha_finetune.py fetch
    python scripts/noha_finetune.py label
    python scripts/noha_finetune.py train --base openai/whisper-small
    python scripts/noha_match.py transcribe --model models/whisper-small-noha-ct2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import noha_match as N  # noqa: E402

TRAIN = N.OUT / "train"
SR = 16000
WHISPER_LANG = {"ur": "urdu", "ar": "arabic", "fa": "persian", "en": "english"}
SCRIPT = {"ur": "arab", "ar": "arab", "fa": "arab", "en": "latn"}


def test_channels() -> set[str]:
    return {m["channel"] for m in N.load_items()}


def cmd_fetch(max_per_lang: int) -> None:
    ytdlp = str(Path(sys.executable).with_name("yt-dlp"))
    avoid = test_channels()
    for lang in N.LANGS:
        out = TRAIN / lang
        out.mkdir(parents=True, exist_ok=True)
        have = len(list(out.glob("*.json")))
        for p in sorted((N.OUT / "distractors" / lang).glob("*.lyrics.json")):
            if have >= max_per_lang:
                break
            vid = p.name.split(".")[0]
            if (out / f"{vid}.json").exists() or (out / f"{vid}.skip").exists():
                continue
            r = subprocess.run([ytdlp, "-f", "bestaudio[abr<=96]/bestaudio", "--no-playlist", "--no-warnings",
                                "--match-filter", "duration >= 90 & duration <= 1200",
                                "--print", "after_move:%(channel)s\t%(duration)s\t%(filepath)s", "--no-simulate",
                                "-o", str(out / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"],
                               capture_output=True, text=True, encoding="utf-8", timeout=900)
            parts = (r.stdout.strip().splitlines() or [""])[-1].split("\t")
            if r.returncode != 0 or len(parts) != 3 or not Path(parts[2]).exists():
                (out / f"{vid}.skip").write_text(r.stderr[-300:] or f"filtered: {parts}", encoding="utf-8")
                continue
            channel, duration, path = parts
            if channel in avoid:  # a test reciter: never train on them
                Path(path).unlink(missing_ok=True)
                (out / f"{vid}.skip").write_text("test channel", encoding="utf-8")
                continue
            lyr = json.loads(p.read_text(encoding="utf-8"))
            (out / f"{vid}.json").write_text(json.dumps({
                "video_id": vid, "lang": lang, "channel": channel, "duration_s": float(duration),
                "audio": Path(path).name, "script": lyr["script"], "lines": lyr["lines"]}, ensure_ascii=False,
                indent=1), encoding="utf-8")
            have += 1
            print(f"{lang} {have:3d} {vid} [{channel}] {duration}s", flush=True)
        print(f"{lang}: {have} training recordings", flush=True)


def _snap(text: str, lines: list[str]) -> tuple[float, str]:
    """(per-letter cost, the stretch of `lines` the transcript aligns to)."""
    from dua_recognition.align import semiglobal_end_costs

    tokens, codes, owner = [], [], []
    for line in lines:
        for tok in line.split():
            c = N.encode_key(tok)
            if c.size:
                owner += [len(tokens)] * c.size
                codes.append(c)
                tokens.append(tok)
    h = N.encode_key(text)
    if not h.size or not codes:
        return np.inf, ""
    r = np.concatenate(codes)
    end_costs = semiglobal_end_costs(h, r)
    e = int(end_costs.argmin())
    # Where does that alignment start? The same DP, reversed, over r[:e+1].
    rev = semiglobal_end_costs(h[::-1], r[: e + 1][::-1])
    s = e - int(rev.argmin())
    return float(end_costs[e]) / h.size, " ".join(tokens[owner[s]: owner[e] + 1])


def cmd_label(teacher: str, max_cost: float, hop: float) -> None:
    from faster_whisper.audio import decode_audio

    from dua_recognition.asr import load_model

    model = load_model(teacher)
    rows = []
    for meta_path in sorted(TRAIN.glob("*/*.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        audio_path = meta_path.with_name(m["audio"])
        y = decode_audio(str(audio_path), sampling_rate=SR)
        starts = np.arange(0, max(y.size / SR - 6, 0), hop)
        wins = [y[int(s * SR): int((s + 6) * SR)] for s in starts]
        texts = N.transcribe_all(teacher, wins, langs=(m["lang"],), signals=False)["texts"][m["lang"]]
        kept = 0
        for s, text in zip(starts, texts):
            if not N._usable(text) or N._looped(text):
                continue
            cost, stretch = _snap(text, m["lines"])
            # A one- or two-consonant transcript ("أشياء" = s) fits any lyric
            # perfectly; it verifies nothing.
            if cost > max_cost or N.encode_key(text).size < 6 or N.encode_key(stretch).size < 6:
                continue
            label = stretch if m["script"] == SCRIPT[m["lang"]] else text
            rows.append({"audio": str(audio_path), "start": float(s), "end": float(s) + 6, "lang": m["lang"],
                         "text": label, "teacher": text, "cost": round(cost, 3), "channel": m["channel"],
                         "video_id": m["video_id"]})
            kept += 1
        print(f"{m['lang']} {m['video_id']}: kept {kept}/{len(starts)} windows", flush=True)
    del model
    rng = random.Random(0)
    chans = sorted({r["channel"] for r in rows})
    rng.shuffle(chans)
    val_ch = set(chans[: max(2, len(chans) // 12)])
    for name, part in (("train", [r for r in rows if r["channel"] not in val_ch]),
                       ("val", [r for r in rows if r["channel"] in val_ch])):
        (TRAIN / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part),
                                             encoding="utf-8")
        print(f"{name}: {len(part)} windows", {l: sum(r['lang'] == l for r in part) for l in N.LANGS})


def cmd_train(base: str, name: str, epochs: int, batch: int, lr: float, room: float) -> None:
    import shutil

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

    from finetune_whisper import AudioBank, augment

    rows = {k: [json.loads(x) for x in (TRAIN / f"{k}.jsonl").read_text(encoding="utf-8").splitlines() if x]
            for k in ("train", "val")}
    print({k: len(v) for k, v in rows.items()}, flush=True)
    bank = AudioBank([r["audio"] for r in rows["train"] + rows["val"]])
    rng = random.Random(0)
    torch.manual_seed(0)
    out = ROOT / "models" / name
    proc = WhisperProcessor.from_pretrained(base)
    tok = proc.tokenizer
    model = WhisperForConditionalGeneration.from_pretrained(base).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.generation_config.forced_decoder_ids = None
    start_id = model.config.decoder_start_token_id

    def labels(rs):
        seqs = []
        for r in rs:
            tok.set_prefix_tokens(language=WHISPER_LANG[r["lang"]], task="transcribe", predict_timestamps=False)
            seqs.append(tok(r["text"]).input_ids)
        n = max(map(len, seqs))
        ids = torch.full((len(seqs), n), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        if (ids[:, 0] == start_id).all():
            ids = ids[:, 1:]
        return ids

    def batch_of(rs, train_mode):
        wav = [bank.window(r) for r in rs]
        if train_mode:
            wav = [augment(x, rng, room) for x in wav]
        feats = proc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features
        return feats.cuda(), labels(rs).cuda()

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = []
        for i in range(0, min(len(rows["val"]), 600), batch):
            feats, ids = batch_of(rows["val"][i: i + batch], False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls.append(model(input_features=feats, labels=ids).loss.item())
        model.train()
        return float(np.mean(ls))

    best = val_loss()
    print(f"before: val loss {best:.3f}", flush=True)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    train = rows["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / batch)
    sched = get_linear_schedule_with_warmup(opt, min(100, steps // 10), steps)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        rng.shuffle(train)
        for i in range(0, len(train), batch):
            feats, ids = batch_of(train[i: i + batch], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=feats, labels=ids).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f} "
                      f"{(time.time() - t0) / step:.2f} s/step", flush=True)
        vl = val_loss()
        print(f"epoch {epoch + 1}: val loss {vl:.3f}", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(out)
    ct2 = out.with_name(out.name + "-ct2")
    subprocess.run([sys.executable, "-m", "ctranslate2.converters.transformers", "--model", str(out),
                    "--output_dir", str(ct2), "--force", "--quantization", "float16"], check=True)
    shutil.copy(out / "tokenizer.json", ct2 / "tokenizer.json")
    proc.feature_extractor.to_json_file(ct2 / "preprocessor_config.json")
    print(f"done: {ct2.relative_to(ROOT)} (best val loss {best:.3f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--max-per-lang", type=int, default=60)
    lb = sub.add_parser("label")
    lb.add_argument("--teacher", default="large-v3-turbo")
    lb.add_argument("--max-cost", type=float, default=0.35)
    lb.add_argument("--hop", type=float, default=3.0)
    t = sub.add_parser("train")
    t.add_argument("--base", default="openai/whisper-small")
    t.add_argument("--name", default="whisper-small-noha")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--room", type=float, default=0.3)
    args = ap.parse_args()
    if args.cmd == "fetch":
        cmd_fetch(args.max_per_lang)
    elif args.cmd == "label":
        cmd_label(args.teacher, args.max_cost, args.hop)
    else:
        cmd_train(args.base, args.name, args.epochs, args.batch, args.lr, args.room)


if __name__ == "__main__":
    main()
