#!/usr/bin/env python
"""Vocals-only audio for the noha test windows (Demucs htdemucs), to measure
whether removing music, daf and matam before ASR helps.

Only the regions scripts/noha_match.py tests are separated: HORIZON seconds
from each cold start. Saved to data/noha/vocals/<id>.npz (one array per start);
`noha_match.py transcribe --vocals` reads them.

    pip install demucs
    python scripts/noha_separate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import noha_match as N  # noqa: E402


def main() -> None:
    import torch
    import torchaudio.functional as F
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from noha_lid import load_audio

    model = get_model("htdemucs").cuda().eval()
    sr = model.samplerate  # 44.1 kHz
    out_dir = N.OUT / "vocals"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in N.test_items():
        path = out_dir / f"{m['video_id']}.npz"
        if path.exists():
            continue
        audio = load_audio(Path(m["audio_path"]))
        segs = {}
        for k, frac in enumerate(N.STARTS):
            s = frac * audio.size / N.SR
            seg = torch.from_numpy(audio[int(s * N.SR): int((s + N.HORIZON) * N.SR)].copy())
            x = F.resample(seg, N.SR, sr)[None].repeat(2, 1)[None].cuda()  # mono -> stereo, batch
            with torch.no_grad():
                stems = apply_model(model, x, device="cuda", progress=False)[0]
            vocals = stems[model.sources.index("vocals")].mean(0)
            segs[f"s{k}"] = F.resample(vocals.cpu(), sr, N.SR).numpy().astype(np.float32)
        np.savez_compressed(path, **segs)
        print(m["lang"], m["video_id"], flush=True)


if __name__ == "__main__":
    main()
