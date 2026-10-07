"""Bigger CTC models for the server engine, behind the phone student's window interface.

The phone's CTC model (ctc_student.py) is a whisper-base encoder with a letter head, small
enough for a browser. The server can run a model fifteen to thirty times its size ten times a
second. scripts/train_server_ctc.py trains one of these on the same crops as the student (crops
ending anywhere, often mid-word, the wav2vec2 teacher's frames and the letters inside as
targets):

- ``w2v``: a wav2vec2 CTC model (default: the teacher itself, models/wav2vec2-quran-dua-voices),
  its vocabulary folded onto the corpus alphabet as ctc.CtcModel folds it;
- ``w2vbert``: a w2v-BERT 2.0 encoder (facebook/w2v-bert-2.0, or a fine-tune of it) with a new
  linear head onto the alphabet. Its input is Kaldi-style 80-bin filterbanks, two 10 ms frames
  stacked per 20 ms step (as transformers' SeamlessM4TFeatureExtractor makes them; ``Fbank``
  computes the same on the GPU);
- ``whisper``: a Whisper encoder of any size with a linear head (ctc_student.WhisperCTC).

All three emit one frame per 20 ms, column 0 = blank, as the stream decoder expects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ctc import LOG_FLOOR, N_COLS, fold_matrix

SR = 16000


def w2v_frames(n_samples: int) -> int:
    """Frames a wav2vec2 or w2v-BERT model emits for this many samples."""
    return 0 if n_samples < 400 else ((n_samples - 400) // 160 + 1) // 2


# SpecAugment in training for 2-4 s crops: the usual two 10-frame masks at least would hide a fifth
# of a 2 s crop (often its newest frames, the ones the decoder reads).
SHORT_MASKS = dict(mask_time_prob=0.05, mask_time_length=5, mask_time_min_masks=1)


class Fbank(nn.Module):
    """SeamlessM4TFeatureExtractor on the GPU: [B, N] float audio in [-1, 1] -> [B, F, 160]
    (80 log-mel bins, Kaldi framing: 25 ms povey window every 10 ms, DC removed, preemphasis
    0.97; each bin normalized over the window; pairs of frames stacked)."""

    def __init__(self):
        super().__init__()
        from transformers import SeamlessM4TFeatureExtractor

        fe = SeamlessM4TFeatureExtractor()
        self.register_buffer("mel", torch.tensor(np.asarray(fe.mel_filters, dtype=np.float32)), persistent=False)
        self.register_buffer("window", torch.tensor(np.asarray(fe.window, dtype=np.float32)), persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=y.device.type, enabled=False):  # powers of 16-bit samples overflow float16
            return self._fbank(y)

    def _fbank(self, y: torch.Tensor) -> torch.Tensor:
        x = y.float() * 32768.0
        fr = x.unfold(1, 400, 160)  # [B, T, 400]
        fr = fr - fr.mean(-1, keepdim=True)
        fr = torch.cat([fr[..., :1] * (1 - 0.97), fr[..., 1:] - 0.97 * fr[..., :-1]], -1)
        fr = fr * self.window
        spec = torch.fft.rfft(fr, n=512).abs() ** 2  # [B, T, 257]
        feats = torch.log(torch.clamp(spec @ self.mel, min=1.192092955078125e-07))  # [B, T, 80]
        mu = feats.mean(1, keepdim=True)
        var = feats.var(1, keepdim=True, unbiased=True)
        feats = (feats - mu) / torch.sqrt(var + 1e-7)
        T = feats.shape[1] // 2 * 2
        return feats[:, :T].reshape(feats.shape[0], T // 2, 160)


class W2VBertCTC(nn.Module):
    def __init__(self, base: str, dropout: float = 0.1, config_only: bool = False):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        kw = dict(add_adapter=False, layerdrop=0.0, **SHORT_MASKS)
        if config_only:
            self.enc = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_pretrained(base, **kw))
        else:
            self.enc = Wav2Vec2BertModel.from_pretrained(base, dtype=torch.float32, **kw)
        self.fbank = Fbank()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.enc.config.hidden_size, N_COLS)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.fbank(wav)).last_hidden_state
        return self.head(self.drop(h)).float().log_softmax(-1)


class W2VFoldCTC(nn.Module):
    def __init__(self, base: str, config_only: bool = False):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCTC, AutoProcessor

        if config_only:
            self.model = AutoModelForCTC.from_config(AutoConfig.from_pretrained(base, layerdrop=0.0, **SHORT_MASKS))
        else:
            self.model = AutoModelForCTC.from_pretrained(base, dtype=torch.float32, layerdrop=0.0, **SHORT_MASKS)
        proc = AutoProcessor.from_pretrained(base)
        self.normalize = bool(getattr(proc.feature_extractor, "do_normalize", True))
        self.register_buffer("fold", torch.tensor(fold_matrix(proc.tokenizer.get_vocab(), proc.tokenizer.pad_token_id)),
                             persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.float()
        if self.normalize:
            x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-7)
        logits = self.model(x).logits.float()
        return torch.log((logits.softmax(-1) @ self.fold).clamp_min(math.exp(LOG_FLOOR)))


class WhisperEncCTC(nn.Module):
    def __init__(self, base: str, config_only: bool = False):
        super().__init__()
        from transformers import WhisperConfig, WhisperForConditionalGeneration

        from .ctc_student import LogMel, WhisperCTC

        if config_only:
            w = WhisperForConditionalGeneration(WhisperConfig.from_pretrained(base))
        else:
            w = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32, attn_implementation="sdpa")
        self.net = WhisperCTC(w.model.encoder)
        self.mel = LogMel(w.config.num_mel_bins)
        del w

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(self.mel(wav)).float()


ARCHS = {"w2v": W2VFoldCTC, "w2vbert": W2VBertCTC, "whisper": WhisperEncCTC}


def frames_for(arch: str):
    from .ctc_student import n_frames

    return n_frames if arch == "whisper" else w2v_frames


def build(arch: str, base: str, config_only: bool = False) -> nn.Module:
    return ARCHS[arch](base, config_only=config_only)


def save(model: nn.Module, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
    tmp = out / "server_ctc.pt.part"
    torch.save({"state": state, "meta": meta}, tmp)
    tmp.replace(out / "server_ctc.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


class ServerCtc:
    """A trained server CTC model with ctc.CtcModel's window interface (bench.py, the server)."""

    def __init__(self, path: str | Path, device: str | None = None):
        import os

        path = Path(path)
        ck = torch.load(path / "server_ctc.pt", map_location="cpu", weights_only=False)
        self.meta = meta = ck["meta"]
        self.device = device or os.environ.get("DUA_CTC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build(meta["arch"], meta["base"], config_only=True)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state"].items()},
                                   strict=False)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
            if hasattr(self.model, "fbank"):
                self.model.fbank.float()
            if hasattr(self.model, "mel"):
                self.model.mel.float()
            if hasattr(self.model, "fold"):
                self.model.fold = self.model.fold.float()
        self.frames = frames_for(meta["arch"])
        self.window_s = float(meta.get("window_s", 3.0))
        self.torch = torch

    def _run(self, wins: list[np.ndarray]) -> list[np.ndarray]:
        n = int(round(self.window_s * SR))
        # shorter than the window (a session's first seconds): silence before, as the page's zeroed buffer
        wins = [np.r_[np.zeros(n - len(w), np.float32), w] if 0 < len(w) < n else w for w in wins]
        res: dict[int, np.ndarray] = {}
        groups: dict[int, list[int]] = {}
        for i, w in enumerate(wins):
            groups.setdefault(len(w), []).append(i)
        with torch.no_grad():
            for m, idx in groups.items():
                if m < 400:
                    for i in idx:
                        res[i] = np.zeros((0, N_COLS), dtype=np.float32)
                    continue
                for b in range(0, len(idx), 32):
                    sub = idx[b : b + 32]
                    y = torch.tensor(np.stack([wins[i] for i in sub]), device=self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                        lp = self.model(y).float()
                    for k, i in enumerate(sub):
                        res[i] = lp[k].cpu().numpy()
        return [res[i] for i in range(len(wins))]

    def windows(self, y: np.ndarray, times: list[float], window: float, batch: int = 16):
        from .align import _ALPHABET

        inv = {v: k for k, v in _ALPHABET.items()}
        wins = [np.ascontiguousarray(y[int(max(0.0, t - window) * SR) : int(t * SR)], dtype=np.float32) for t in times]
        outs = self._run(wins)
        lens = [o.shape[0] for o in outs]
        fmax = max(lens) if lens else 0
        arr = np.full((len(wins), fmax, N_COLS), LOG_FLOOR, dtype=np.float16)
        texts = []
        for k, o in enumerate(outs):
            arr[k, : o.shape[0]] = np.maximum(o, LOG_FLOOR)
            ids = o.argmax(-1) if o.size else np.zeros(0, int)
            seq = [i for j, i in enumerate(ids) if i != 0 and (j == 0 or i != ids[j - 1])]
            texts.append("".join(inv[int(i)] for i in seq))
        return arr, np.array(lens, dtype=np.int32), texts, wins

    def window(self, y: np.ndarray) -> np.ndarray:
        if not len(y):
            return np.zeros((0, N_COLS), dtype=np.float32)
        out = self._run([np.ascontiguousarray(y, dtype=np.float32)])[0]
        return np.maximum(out, LOG_FLOOR).astype(np.float16).astype(np.float32)


def check_fbank(seconds: float = 3.0) -> float:
    """Max abs difference between Fbank and SeamlessM4TFeatureExtractor on noise + tone."""
    from transformers import SeamlessM4TFeatureExtractor

    n = int(seconds * SR)
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(n) + 0.3 * np.sin(2 * math.pi * 440 * np.arange(n) / SR)).astype(np.float32)
    ref = SeamlessM4TFeatureExtractor()(y, sampling_rate=SR, return_tensors="np").input_features[0]
    ours = Fbank()(torch.tensor(y)[None])[0].numpy()
    return float(np.abs(ref - ours[: ref.shape[0]]).max()), ref.shape, ours.shape
