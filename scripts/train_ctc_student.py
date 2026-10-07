#!/usr/bin/env python
"""Train the phone CTC model (ctc_student.py) from the wav2vec2 teacher.

The student hears what the phone will give it: a fixed window of the latest few
seconds (3 s by default), ending anywhere, often mid-word, sometimes at the start of
a session (silence before). Its targets come from scripts/ctc_teacher.py, which ran
the teacher over the whole 6 s training window:

- the teacher's letter posteriors on the crop's frames (KL), with the context the
  student doesn't have: at the crop's right edge the student learns the best guess;
- CTC on the letters the teacher's forced alignment puts inside the crop (the
  reference text, not the teacher's reading of it), so it says what was said up
  to the edge, not later.

The student hears the audio after level, noise and (half the time) room reverb, plus
vocal tract length perturbation and SpecAugment; the teacher heard it clean.

    python scripts/train_ctc_student.py --name ctc-student-base
    python scripts/train_ctc_student.py --base openai/whisper-tiny --name ctc-student-tiny

Writes models/<name>/student.pt (+ meta); scripts/export_ctc_student.py takes it to the browser.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dua_recognition.align import _ALPHABET  # noqa: E402
from dua_recognition.ctc import N_COLS  # noqa: E402
from dua_recognition.ctc_student import SR, LogMel, WhisperCTC, n_frames, save  # noqa: E402

TEACH = ROOT / "data" / "cache" / "ctc_student"
FRAME = 320  # samples per 20 ms frame (teacher and student)
INV = {v: k for k, v in _ALPHABET.items()}


def unpack(name: str, teach: Path = TEACH) -> Path:
    """<set>.npz -> <set>/<array>.npy, so workers can memory-map the frames."""
    d = Path(teach) / name
    if not (d / "done").exists():
        d.mkdir(exist_ok=True)
        z = np.load(d.parent / f"{name}.npz")
        for k in z.files:
            np.save(d / f"{k}.npy", z[k])
        (d / "done").write_text("ok")
    return d


def phone_channel(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """What a phone's mic path does to a voice: automatic gain control that rides the level
    (and turns up the room hiss once the reciter stops), sometimes a narrower band and
    harder compression."""
    if not np.abs(x).max():
        return x
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    y = x + nrng.normal(0, 10 ** (rng.uniform(-75, -50) / 20), x.shape).astype(np.float32)  # a hiss floor
    # AGC: gain toward a target level from a smoothed envelope, fast attack, slow release.
    hop = 160
    env = np.sqrt(np.convolve(y**2, np.ones(hop * 5) / (hop * 5), mode="same")[::hop] + 1e-10)
    target = 10 ** (rng.uniform(-20, -12) / 20)
    want = np.clip(target / env, 10 ** (-6 / 20), 10 ** (rng.uniform(18, 32) / 20))
    g = np.empty_like(want)
    cur = want[0]
    up, down = 1 - np.exp(-1 / rng.uniform(20, 120)), 1 - np.exp(-1 / rng.uniform(1, 5))  # per 10 ms
    for i, w in enumerate(want):
        cur += (w - cur) * (down if w < cur else up)
        g[i] = cur
    y = y * np.interp(np.arange(y.size) / hop, np.arange(g.size), g).astype(np.float32)
    if rng.random() < 0.4:  # a narrower band
        f = np.fft.rfft(y)
        cut = int(rng.uniform(3400, 7000) / 8000 * (f.size - 1))
        lo = int(rng.uniform(50, 300) / 8000 * (f.size - 1))
        f[cut:] = 0
        f[:lo] = 0
        y = np.fft.irfft(f, y.size).astype(np.float32)
    if rng.random() < 0.3:  # hard compression
        y = np.tanh(y * rng.uniform(1.5, 4)) / np.tanh(rng.uniform(1.5, 4))
    return np.clip(y, -1, 1).astype(np.float32)


def noise_gate(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """A phone's noise suppression: between words the level drops to digital silence
    (Hasan's Android, x0fo: -78 dBFS gaps between -15 dBFS words), with short fades."""
    hop = 160
    n = x.size // hop
    if n < 2:
        return x
    env = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    open_ = env > env.max() - rng.uniform(20, 40)
    # Hold open a little after each sound (the gate's release), then fade.
    hold = rng.randint(2, 15)
    k = np.convolve(open_.astype(float), np.ones(hold), mode="full")[:n] > 0
    g = np.where(k, 1.0, 10 ** (-rng.uniform(40, 70) / 20))
    ramp = rng.randint(1, 4)
    g = np.convolve(g, np.ones(ramp) / ramp, mode="same")
    gain = np.interp(np.arange(x.size) / hop, np.arange(n), g).astype(np.float32)
    return (x * gain).astype(np.float32)


class Crops(torch.utils.data.Dataset):
    """(set, row) -> a crop of `window_s` seconds: audio, teacher frames, letters inside."""

    def __init__(self, items: list[tuple[str, int]], window_s: float, train: bool, room_p: float, seed: int = 0,
                 phone_p: float = 0.0, teach: Path = TEACH, gate_p: float = 0.0, hall_p: float = 0.0,
                 hall_voices: float = 0.0, shift: int = 0):
        self.items, self.window_s, self.train, self.room_p, self.seed = items, window_s, train, room_p, seed
        self.phone_p, self.teach, self.gate_p, self.hall_p = phone_p, teach, gate_p, hall_p
        self.hall_voices = hall_voices
        # targets `shift` frames later than the audio (the teacher's frame f + shift at the model's frame
        # f, the letters up to `shift` frames past the crop's end): a model that says each letter that
        # much sooner, so the display waits less for it (peak-first distillation)
        self.shift = shift
        self._open = None

    def _lazy(self):
        if self._open is None:
            from finetune_whisper import AudioBank, load_rows

            sets = sorted({s for s, _ in self.items})
            self._open = {}
            for s in sets:
                d = unpack(s, self.teach)
                rows = load_rows(s)
                arr = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in
                       ("frames", "offsets", "letters", "loff", "first", "last", "score", "samples")}
                self._open[s] = (rows, arr, AudioBank([r.get("audio") for r in rows]))
        return self._open

    def __len__(self):
        return len(self.items)

    # frames the model emits for a crop of this many samples (a wav2vec2 model: (n - 400) // 320 + 1)
    frames_fn = staticmethod(n_frames)

    def __getitem__(self, i):
        from finetune_whisper import augment

        # i, or (i, seconds) from a sampler that picks each batch's window length
        i, window_s = i if isinstance(i, tuple) else (i, self.window_s)
        s, k = self.items[i]
        rows, a, bank = self._lazy()[s]
        rng = random.Random((self.seed << 32) ^ (i * 2654435761) ^ (int(time.time() * 1e6) if self.train else 0))
        y = bank.window(rows[k])
        # a seat in a hall: the whole window is reverberated before the crop, so the crop starts in
        # the tail of what came before, as a live window does
        hall = bool(self.train and self.hall_p and rng.random() < self.hall_p)
        if hall:
            from halls import hall_aug

            y = hall_aug(y, rng, self.hall_voices)
        W = int(window_s * SR)
        tf = a["frames"][a["offsets"][k] : a["offsets"][k + 1]].astype(np.float32)  # teacher [Tt, C]
        T = tf.shape[0]
        lo, hi = a["loff"][k], a["loff"][k + 1]
        letters = a["letters"][lo:hi].astype(np.int64)
        first = a["first"][lo:hi]
        good = float(a["score"][k]) > -0.5
        # Crop end: a frame boundary; train: anywhere from 1 s in (edges of the 6 s
        # window are where labels are least sure, so not in its last 0.2 s); val: the end.
        nfr = len(y) // FRAME
        if self.train:
            end_f = rng.randint(min(nfr, 50), max(min(nfr, 50), nfr - (10 if nfr > 150 else 0)))
        else:
            end_f = nfr
        start_f = end_f - W // FRAME
        a0 = start_f * FRAME
        x = np.zeros(W, dtype=np.float32)
        src0 = max(0, a0)
        seg = y[src0 : end_f * FRAME]
        x[W - seg.size :] = seg  # silence before the recording's start
        if self.train:
            x = augment(x, rng, self.room_p) if np.abs(x).max() > 0 and not hall else x
            if self.phone_p and rng.random() < self.phone_p:
                x = phone_channel(x, rng)
            if self.gate_p and rng.random() < self.gate_p:
                x = noise_gate(x, rng)
        # Teacher frames for student frames 0..F-1 (blank where the crop is before the start).
        F = self.frames_fn(W)
        tgt = np.full((F, N_COLS), -30.0, dtype=np.float32)
        tgt[:, 0] = 0.0
        idx = np.arange(F) + start_f + self.shift
        ok = (idx >= 0) & (idx < T)
        tgt[ok] = tf[idx[ok]]
        # Letters emitted inside the crop. The window's label leaves out words cut at its
        # edges and sometimes adds one the audio never reaches (forced onto the last frames):
        # trust the labelled letters the teacher hears, and use the crop's letters only if
        # the teacher hears nothing in the crop outside that span.
        ok_lab = False
        lab = np.zeros(0, dtype=np.int64)
        if good and letters.size and (first >= 0).all():
            conf = np.exp(tf[np.clip(first, 0, T - 1), letters])
            heard = np.flatnonzero(conf > 0.2)
            if heard.size:
                j0, j1 = int(heard[0]), int(heard[-1])
                lo_f, hi_f = int(first[j0]), int(first[j1])
                letterish = np.exp(tf[:, 1:]).max(1) > 0.5
                s0, e0 = max(0, start_f + self.shift), min(T, end_f + self.shift)
                outside = letterish[s0 : max(s0, lo_f - 2)].any() or letterish[min(e0, hi_f + 6) : e0].any()
                if not outside:
                    keep = np.zeros(letters.size, dtype=bool)
                    keep[j0 : j1 + 1] = True
                    inside = keep & (first >= s0) & (first < end_f + self.shift)
                    lab = letters[inside]
                    ok_lab = True
        return torch.tensor(x), torch.tensor(tgt), torch.tensor(lab), bool(ok_lab)


def collate(b):
    x = torch.stack([e[0] for e in b])
    t = torch.stack([e[1] for e in b])
    labs = [e[2] for e in b]
    good = torch.tensor([e[3] for e in b])
    return x, t, labs, good


def greedy(lp: np.ndarray) -> str:
    ids = lp.argmax(-1)
    return "".join(INV[int(i)] for j, i in enumerate(ids) if i != 0 and (j == 0 or i != ids[j - 1]))


def cer(refs, hyps) -> float:
    import jiwer

    refs = [r or "-" for r in refs]
    hyps = [h or "-" for h in hyps]
    return jiwer.cer(refs, hyps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="models/whisper-base-syn-v5-ctx8ft")
    ap.add_argument("--name", default="ctc-student-base")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--kl", type=float, default=1.0, help="weight of the teacher-frame KL term")
    ap.add_argument("--room", type=float, default=0.5)
    ap.add_argument("--vtlp", type=float, default=0.5)
    ap.add_argument("--specaug", action="store_true", default=True)
    ap.add_argument("--phone", type=float, default=0.0, help="share of crops through phone_channel()")
    ap.add_argument("--synth", default="synth_v5", help="synthetic sets, comma-separated (empty: none)")
    ap.add_argument("--synth-share", type=float, default=0.15)
    ap.add_argument("--crowd", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="steps (smoke test)")
    ap.add_argument("--train-set", default="train_v4")
    ap.add_argument("--teacher", default="", help="teacher cache under data/cache/ (default ctc_student)")
    ap.add_argument("--init", help="start from this trained student (models/<name>) instead of --base's encoder")
    ap.add_argument("--gate", type=float, default=0.0, help="share of crops through noise_gate()")
    ap.add_argument("--hall", type=float, default=0.0,
                    help="share of windows heard in a hall through a PA, with a crowd (halls.hall_aug)")
    ap.add_argument("--hall-voices", type=float, default=0.0,
                    help="share of --hall crowds that are real people talking (Common Voice, training side)")
    ap.add_argument("--shift", type=int, default=0,
                    help="frames of anticipation: targets this many 20 ms frames after the audio (Crops.shift)")
    args = ap.parse_args()
    teach = ROOT / "data" / "cache" / args.teacher if args.teacher else TEACH

    from finetune_whisper import spec_augment, vtlp

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = ROOT / "models" / args.name
    n_train = len(np.load(unpack(args.train_set, teach) / "offsets.npy", mmap_mode="r")) - 1
    items = [(args.train_set, k) for k in range(n_train)]
    for syn in [x for x in args.synth.split(",") if x]:
        ns = len(np.load(unpack(syn, teach) / "offsets.npy", mmap_mode="r")) - 1
        take = min(ns, int(n_train * args.synth_share / (1 - args.synth_share)))
        pick = rng.sample(range(ns), take)
        items += [(syn, k) for k in pick]
        print(f"+{take} synthetic clips ({syn})", flush=True)
    if args.crowd:
        nc = len(np.load(unpack("crowd", teach) / "offsets.npy", mmap_mode="r")) - 1
        items += [("crowd", k) for k in range(nc)]
        print(f"+{nc} crowd clips", flush=True)
    nv = len(np.load(unpack("val_v4", teach) / "offsets.npy", mmap_mode="r")) - 1
    val_items = [("val_v4", k) for k in range(0, nv, max(1, nv // 600))]
    print(f"{len(items)} training items, {len(val_items)} val windows", flush=True)

    model = WhisperCTC.from_whisper(args.base)
    if args.init:
        ck = torch.load(Path(args.init) / "student.pt", map_location="cpu", weights_only=False)
        if ck["meta"].get("base") != args.base:
            raise SystemExit(f"--init {args.init} was trained from {ck['meta'].get('base')}, not --base {args.base}")
        model.load_state_dict(ck["state"])
        print(f"from {args.init} (step {ck['meta'].get('step')})", flush=True)
    model = model.cuda()
    mel = LogMel().cuda()
    head = list(model.head.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("head.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.limit or int(args.epochs * len(items) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)

    ds = Crops(items, args.window, True, args.room, args.seed, phone_p=args.phone, teach=teach, gate_p=args.gate,
               hall_p=args.hall, hall_voices=args.hall_voices, shift=args.shift)
    vds = Crops(val_items, args.window, False, 0.0, teach=teach, shift=args.shift)  # judged on its own targets
    # Val: whole 6 s windows too (the window's own text), for a CER comparable with the teacher's.
    from finetune_whisper import AudioBank, load_rows

    vrows = load_rows("val_v4")
    vbank = AudioBank([vrows[k].get("audio") for _, k in val_items])

    @torch.no_grad()
    def evaluate():
        model.eval()
        refs, hyps, kls = [], [], []
        for b in range(0, len(val_items), 64):
            batch = [vds[i] for i in range(b, min(len(val_items), b + 64))]
            x, t, labs, good = collate(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = model(mel(x.cuda())).float()
            tp = t.cuda()
            kls.append(((tp.exp() * (tp - lp)).sum(-1)).mean().item())
            for k in range(lp.shape[0]):
                refs.append("".join(INV[int(c)] for c in labs[k]))
                hyps.append(greedy(lp[k].cpu().numpy()))
        crop_cer = cer(refs, hyps)
        # whole windows
        refs, hyps = [], []
        for _, k in val_items[:300]:
            y = vbank.window(vrows[k])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = model(mel(torch.tensor(y)[None].cuda())).float()
            refs.append("".join(INV[int(c)] for c in __import__("dua_recognition.align", fromlist=["encode"]).encode(vrows[k]["text"])))
            hyps.append(greedy(lp[0].cpu().numpy()))
        model.train()
        return crop_cer, cer(refs, hyps), float(np.mean(kls))

    c0 = evaluate()
    print(f"before: val crop CER {c0[0]:.1%}  6 s CER {c0[1]:.1%}  KL {c0[2]:.3f}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0,
                                     prefetch_factor=4 if args.workers else None)
    model.train()
    step, t0, best = 0, time.time(), None
    log_every = 100
    run = {"ctc": 0.0, "kl": 0.0}
    meta = {"teacher": str(teach.name), "base": args.base, "max_positions": None, "window_s": args.window,
            "args": vars(args)}
    eval_every = max(500, steps // 12)
    done = False
    while not done:
        for x, t, labs, good in dl:
            x, t = x.cuda(non_blocking=True), t.cuda(non_blocking=True)
            with torch.no_grad():
                feats = mel(x)
                if args.vtlp:
                    feats = vtlp(feats.cpu(), rng, args.vtlp).cuda()
                if args.specaug:
                    feats = spec_augment(feats, rng, n_freq=2, f_max=10, n_time=2, t_max=20)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = model(feats)
            lp = lp.float()
            F = lp.shape[1]
            kl = (t.exp() * (t - lp)).sum(-1).sum(-1)  # per crop
            lens = torch.tensor([len(l) for l in labs])
            keep = good & (lens > 0) & (lens <= F // 2)
            ctc = torch.zeros((), device=lp.device)
            if keep.any():
                ki = keep.nonzero().flatten().tolist()
                tg = torch.cat([labs[i] for i in ki]).cuda()
                ctc = ctc_loss(lp[ki].transpose(0, 1), tg, torch.full((len(ki),), F, dtype=torch.long),
                               lens[ki].cuda())
            loss = (ctc + args.kl * kl.sum()) / (lp.shape[0] * F) * 50  # per second of audio
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run["ctc"] += ctc.item() / max(1, int(keep.sum())) / 1.0
            run["kl"] += kl.mean().item() / F
            if step % log_every == 0:
                print(f"step {step}/{steps}  ctc/crop {run['ctc'] / log_every:.2f}  kl/frame {run['kl'] / log_every:.4f}  "
                      f"{(time.time() - t0) / step:.3f} s/step", flush=True)
                run = {"ctc": 0.0, "kl": 0.0}
            if step % eval_every == 0 or step >= steps:
                c = evaluate()
                print(f"step {step}: val crop CER {c[0]:.1%}  6 s CER {c[1]:.1%}  KL {c[2]:.4f}", flush=True)
                score = c[0] + c[2]
                if best is None or score < best:
                    best = score
                    save(model, out, {**meta, "step": step, "val_crop_cer": c[0], "val_cer_6s": c[1], "val_kl": c[2]})
                    print(f"  saved -> {out.relative_to(ROOT)}", flush=True)
            if step >= steps:
                done = True
                break
    (out / "train_done").write_text(json.dumps({"steps": step, "minutes": (time.time() - t0) / 60}))


if __name__ == "__main__":
    main()
