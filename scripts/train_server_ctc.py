#!/usr/bin/env python
"""Train a CTC model for the server engine (dua_recognition/server_ctc.py).

The same crops as the phone's student (train_ctc_student.py: a window of the latest seconds
ending anywhere, the wav2vec2 teacher's frames over the whole 6 s window as KL targets, CTC on
the letters its forced alignment puts inside the crop), with a bigger model:

    python scripts/train_server_ctc.py --arch w2v --name ctc-server-w2v          # the teacher, made streaming
    python scripts/train_server_ctc.py --arch w2vbert --base facebook/w2v-bert-2.0 --name ctc-server-w2vbert
    python scripts/train_server_ctc.py --arch whisper --base models/whisper-turbo-dua --name ctc-server-turbo

Each batch takes one window length from --windows (s), so one model serves 2 s or longer
windows. Writes models/<name>/server_ctc.pt (+ meta.json); bench.py asr --ctc-model and the
server (DUA_CTC_MODEL) load it through ctc_student.load_ctc.
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

from dua_recognition.server_ctc import build, frames_for, save  # noqa: E402
from train_ctc_student import TEACH, Crops, cer, collate, greedy, unpack  # noqa: E402

INIT = {"w2v": "models/wav2vec2-quran-dua-voices", "w2vbert": "facebook/w2v-bert-2.0",
        "whisper": "models/whisper-turbo-dua"}


class LengthBatches(torch.utils.data.Sampler):
    """Shuffled batches, each of one window length: yields [(i, seconds), ...]."""

    def __init__(self, n: int, batch: int, lengths: list[float], seed: int, frames_per_batch: bool = True):
        self.n, self.batch, self.lengths, self.seed, self.epoch = n, batch, lengths, seed, 0
        self.scale = frames_per_batch

    def __iter__(self):
        rng = random.Random(self.seed * 1000 + self.epoch)
        self.epoch += 1
        order = list(range(self.n))
        rng.shuffle(order)
        k = 0
        while k < self.n:
            L = rng.choice(self.lengths)
            # about the same audio per batch whatever the length
            b = max(4, int(round(self.batch * min(self.lengths) / L))) if self.scale else self.batch
            chunk = order[k : k + b]
            k += b
            if len(chunk) == b:
                yield [(i, L) for i in chunk]

    def __len__(self):
        return self.n // self.batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["w2v", "w2vbert", "whisper"], required=True)
    ap.add_argument("--base", help="weights to start from (default per arch)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--windows", default="2,3,4", help="window lengths (s), one per batch")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32, help="crops per batch at the shortest window")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--room", type=float, default=0.5)
    ap.add_argument("--phone", type=float, default=0.5)
    ap.add_argument("--hall", type=float, default=0.3)
    ap.add_argument("--hall-voices", type=float, default=0.0)
    ap.add_argument("--synth", default="synth_v5,synth_v6,cv_ar,cv_ar_test")
    ap.add_argument("--synth-share", type=float, default=0.15)
    ap.add_argument("--crowd", action="store_true", default=True)
    ap.add_argument("--train-set", default="train_v4")
    ap.add_argument("--extra", default="", help="more sets at their full size, comma-separated (e.g. harvest)")
    ap.add_argument("--teacher", default="ctc_student_voices")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--freeze-layers", type=int, default=0, help="keep the encoder's first N layers fixed")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true", help="continue from models/<name>/last.pt")
    ap.add_argument("--shift", type=int, default=0, help="frames of anticipation (Crops.shift; val stays unshifted)")
    ap.add_argument("--init", help="start from this trained server model (models/<name>) instead of --base")
    args = ap.parse_args()
    base = args.base or INIT[args.arch]
    teach = ROOT / "data" / "cache" / args.teacher
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = ROOT / "models" / args.name

    n_train = len(np.load(unpack(args.train_set, teach) / "offsets.npy", mmap_mode="r")) - 1
    items = [(args.train_set, k) for k in range(n_train)]
    for syn in [x for x in args.synth.split(",") if x]:
        ns = len(np.load(unpack(syn, teach) / "offsets.npy", mmap_mode="r")) - 1
        take = min(ns, int(n_train * args.synth_share / (1 - args.synth_share)))
        items += [(syn, k) for k in rng.sample(range(ns), take)]
        print(f"+{take} synthetic clips ({syn})", flush=True)
    for ex in [x for x in args.extra.split(",") if x]:
        ne = len(np.load(unpack(ex, teach) / "offsets.npy", mmap_mode="r")) - 1
        items += [(ex, k) for k in range(ne)]
        print(f"+{ne} windows ({ex})", flush=True)
    if args.crowd:
        nc = len(np.load(unpack("crowd", teach) / "offsets.npy", mmap_mode="r")) - 1
        items += [("crowd", k) for k in range(nc)]
        print(f"+{nc} crowd clips", flush=True)
    nv = len(np.load(unpack("val_v4", teach) / "offsets.npy", mmap_mode="r")) - 1
    val_items = [("val_v4", k) for k in range(0, nv, max(1, nv // 600))]
    lengths = [float(x) for x in args.windows.split(",")]
    print(f"{len(items)} training items, {len(val_items)} val windows, windows {lengths}", flush=True)

    model = build(args.arch, base)
    if args.init:
        ck = torch.load(Path(args.init) / "server_ctc.pt", map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state"].items()}, strict=False)
        print(f"from {args.init} (step {ck['meta'].get('step')})", flush=True)
    frames = frames_for(args.arch)
    if args.grad_ckpt:
        inner = getattr(model, "enc", None) or getattr(model, "model", None)
        if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable()
    if args.arch == "w2v":
        model.model.freeze_feature_encoder()
    head_names = ("head.", "model.lm_head.", "net.head.")
    if args.freeze_layers:
        # the first N layers and what feeds them (Whisper's convolutions, w2v-BERT's projection) stay fixed:
        # no gradients through them, so a big encoder trains its top layers at a fraction of the cost
        front = ("net.conv1.", "net.conv2.", "enc.feature_projection.", "model.wav2vec2.feature_projection.")
        for n, p in model.named_parameters():
            if n.startswith(front):
                p.requires_grad = False
            for pre in ("enc.encoder.layers.", "model.wav2vec2.encoder.layers.", "net.layers."):
                if n.startswith(pre) and int(n[len(pre):].split(".")[0]) < args.freeze_layers:
                    p.requires_grad = False
    model = model.cuda()
    head = [p for n, p in model.named_parameters() if n.startswith(head_names) and p.requires_grad]
    body = [p for n, p in model.named_parameters() if not n.startswith(head_names) and p.requires_grad]
    print(f"{sum(p.numel() for p in body) / 1e6:.0f} M body + {sum(p.numel() for p in head) / 1e6:.2f} M head "
          f"parameters trained", flush=True)
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=0.01, fused=True)
    steps = args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    step = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)

    Crops.frames_fn = staticmethod(frames)
    ds = Crops(items, max(lengths), True, args.room, args.seed + step, phone_p=args.phone, teach=teach,
               hall_p=args.hall, hall_voices=args.hall_voices, shift=args.shift)
    vds = {L: Crops(val_items, L, False, 0.0, teach=teach, shift=args.shift) for L in sorted({2.0, max(lengths)})}

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for L, d in vds.items():
            refs, hyps, kls, kls_in = [], [], [], []
            for b in range(0, len(val_items), 32):
                batch = [d[i] for i in range(b, min(len(val_items), b + 32))]
                x, t, labs, good = collate(batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = model(x.cuda()).float()
                tp = t.cuda()[:, : lp.shape[1]]
                kl = (tp.exp() * (tp - lp)).sum(-1)
                kls.append(kl.mean().item())
                kls_in.append(kl[:, :-10].mean().item())  # all but the newest 0.2 s: what the decoder commits
                for k in range(lp.shape[0]):
                    refs.append("".join(__import__("train_ctc_student").INV[int(c)] for c in labs[k]))
                    hyps.append(greedy(lp[k].cpu().numpy()))
            res[L] = (cer(refs, hyps), float(np.mean(kls)), float(np.mean(kls_in)))
        model.train()
        return res

    def show(tag, res):
        print(tag + "  " + "  ".join(f"{L:g}s: crop CER {c:.1%} KL {k:.4f} KL-0.2s {ki:.4f}" for L, (c, k, ki) in res.items()),
              flush=True)

    r0 = evaluate()
    show(f"step {step}", r0)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=LengthBatches(len(items), args.batch, lengths, args.seed + step),
                                     num_workers=args.workers, collate_fn=collate, persistent_workers=True,
                                     prefetch_factor=2)
    meta = {"arch": args.arch, "base": base, "teacher": args.teacher, "window_s": 2.0, "args": vars(args)}
    best = None
    model.train()
    t0, run, n_run = time.time(), {"ctc": 0.0, "kl": 0.0}, 0
    done = step >= steps
    while not done:
        for x, t, labs, good in dl:
            x, t = x.cuda(non_blocking=True), t.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = model(x)
            lp = lp.float()
            F = lp.shape[1]
            t = t[:, :F]
            kl = (t.exp() * (t - lp)).sum(-1).sum(-1)
            lens = torch.tensor([len(l) for l in labs])
            keep = good & (lens > 0) & (lens <= F // 2)
            ctc = torch.zeros((), device=lp.device)
            if keep.any():
                ki = keep.nonzero().flatten().tolist()
                tg = torch.cat([labs[i] for i in ki]).cuda()
                ctc = ctc_loss(lp[ki].transpose(0, 1), tg, torch.full((len(ki),), F, dtype=torch.long), lens[ki].cuda())
            loss = (ctc + args.kl * kl.sum()) / (lp.shape[0] * F) * 50
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            run["ctc"] += ctc.item() / max(1, int(keep.sum()))
            run["kl"] += kl.mean().item() / F
            n_run += 1
            if step % 100 == 0:
                print(f"step {step}/{steps}  ctc/crop {run['ctc'] / n_run:.2f}  kl/frame {run['kl'] / n_run:.4f}  "
                      f"{(time.time() - t0) / n_run:.3f} s/step  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                      flush=True)
                run, n_run, t0 = {"ctc": 0.0, "kl": 0.0}, 0, time.time()
            if step % args.eval_every == 0 or step >= steps:
                r = evaluate()
                show(f"step {step}", r)
                c2, _, ki2 = r[2.0]
                score = c2 + ki2
                if best is None or score < best:
                    best = score
                    save(model, out, {**meta, "step": step, "val": {str(k): v for k, v in r.items()}})
                    print(f"  saved -> {out.relative_to(ROOT)}", flush=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step}, out / "last.pt")
                t0 = time.time()
            if step >= steps:
                done = True
                break
    (out / "train_done").write_text(json.dumps({"steps": step}))


if __name__ == "__main__":
    main()
