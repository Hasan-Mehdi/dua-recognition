"""A wav2vec2-style CTC model, folded onto the corpus alphabet.

Per-frame log posteriors with column 0 = blank (which also absorbs diacritics,
word delimiters and specials) and one column per align._ALPHABET letter: what
ctc_align.end_scores and the word follower (follower.py) score against.
scripts/dump_ctc.py runs it over whole recordings; the server runs one window
at a time (CtcModel.window).

    DUA_CTC_DEVICE=cpu   run on the CPU (default: cuda if available)
"""
from __future__ import annotations

import os

import numpy as np

from .align import _ALPHABET
from .text import normalize

SR = 16000
N_COLS = len(_ALPHABET) + 1
LOG_FLOOR = -30.0


def fold_matrix(vocab: dict[str, int], blank_id: int) -> np.ndarray:
    """[V, N_COLS] 0/1: which corpus column each model token's mass goes to."""
    m = np.zeros((max(vocab.values()) + 1, N_COLS), dtype=np.float32)
    for tok, i in vocab.items():
        n = normalize(tok) if len(tok) == 1 else ""
        m[i, _ALPHABET.get(n, 0) if i != blank_id else 0] = 1.0
    return m


class CtcModel:
    """A CTC model plus the fold onto the corpus alphabet, ready to run over windows."""

    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.device = device or os.environ.get("DUA_CTC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCTC.from_pretrained(name).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        self.fold = torch.tensor(fold_matrix(vocab, self.blank), device=self.device)
        self.inv = {i: t for t, i in vocab.items()}

    def windows(self, y: np.ndarray, times: list[float], window: float, batch: int = 16):
        """Folded log posteriors [n, frames, N_COLS] (float16, padded with LOG_FLOOR),
        frame counts, and greedy transcripts for the windows ending at `times`."""
        torch = self.torch
        wins = [y[int(max(0.0, t - window) * SR) : int(t * SR)] for t in times]
        lps, lens, texts = [], [], []
        with torch.no_grad():
            for b in range(0, len(wins), batch):
                chunk = wins[b : b + batch]
                inp = self.proc(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
                vals = inp.input_values.to(self.device)
                mask = inp.get("attention_mask")
                logits = self.model(vals.half() if self.device == "cuda" else vals,
                                    attention_mask=mask.to(self.device) if mask is not None else None).logits.float()
                probs = logits.softmax(-1)
                folded = torch.log((probs @ self.fold).clamp_min(np.exp(LOG_FLOOR)))
                ids = logits.argmax(-1).cpu().numpy()
                for k, x in enumerate(chunk):
                    f = int(self.model._get_feat_extract_output_lengths(torch.tensor(len(x))).item()) if len(x) else 0
                    lps.append(folded[k, :f].cpu().numpy().astype(np.float16))
                    lens.append(f)
                    # greedy CTC decode: collapse repeats, drop blanks
                    seq = [i for j, i in enumerate(ids[k, :f]) if i != self.blank and (j == 0 or i != ids[k, j - 1])]
                    texts.append("".join(self.inv[i] for i in seq).replace("|", " ").strip())
        fmax = max(lens) if lens else 0
        arr = np.full((len(wins), fmax, N_COLS), LOG_FLOOR, dtype=np.float16)
        for k, a in enumerate(lps):
            arr[k, : len(a)] = a
        return arr, np.array(lens, dtype=np.int32), texts, wins

    def window(self, y: np.ndarray) -> np.ndarray:
        """One window (all of `y`): its folded log posteriors [frames, N_COLS], float32,
        rounded through float16 like the dumps the follower was tuned on."""
        if not len(y):
            return np.zeros((0, N_COLS), dtype=np.float32)
        arr, lens, _, _ = self.windows(y, [(len(y) + 0.5) / SR], len(y) / SR + 1.0, 1)  # +0.5: no float cut
        return arr[0, : lens[0]].astype(np.float32)
