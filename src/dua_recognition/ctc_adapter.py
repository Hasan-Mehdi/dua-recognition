"""A sub-word (BPE) text-CTC model for the word follower, without folding pieces away.

ctc.fold_matrix maps every *single-character* token of a wav2vec2 vocabulary onto the
corpus alphabet and sends everything else to blank. For a model whose tokens are
mostly multi-letter pieces (Tilawa's FastConformer: 947 of 1,025) that would discard
nearly all of its evidence. This adapter keeps the model's own tokens instead:

    columns   0 = blank, 1 = the best non-blank token (for has_speech / onset),
              k + 2 = native token k (k != blank)
    frames    the model's own hop (80 ms for FastConformer; never assumed 20 ms)
    reference each corpus word tokenized into the model's pieces, word-initial
              pieces with "▁" (greedy longest match: validated against the model's
              own greedy output in docs/results/small_ctc.md, not assumed)

TokenFollower is LocalFollower with that reference and frame hop; the scoring
(ctc_align.end_scores), move rules and config are unchanged.

The model is run on buffered windows (like wav2vec2 here): FastConformer's encoder
attends to the whole window, so this is *not* cached streaming; each window is
recomputed and its right edge sees no future audio.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from .align import CorpusIndex
from .follower import FollowerConfig, LocalFollower
from .text import strip_diacritics

SR = 16000
LOG_FLOOR = -30.0
SPACE = "▁"
_KEEP = re.compile(r"[^ء-غف-ي ]")


def surface(text: str) -> str:
    """Undiacritized Arabic in ordinary spelling (hamza forms and ة kept, unlike text.normalize)."""
    text = unicodedata.normalize("NFKC", text).replace("ٱ", "ا").replace("ـ", "")
    return re.sub(r"\s+", " ", _KEEP.sub(" ", strip_diacritics(text))).strip()


class TokenCtcModel:
    """An ONNX CTC model that takes raw 16 kHz audio (FastConformer export: audio_signal, length)."""

    def __init__(self, model_dir: str | Path, threads: int = 4):
        import onnxruntime as ort

        d = Path(model_dir)
        onnx = next(d.glob("*.onnx"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.pieces = [vocab[str(i)] for i in range(len(vocab))]
        self.blank = self.pieces.index("<blank>")
        self.name = onnx.name
        n = self.session.run(None, {"audio_signal": np.zeros((1, 3 * SR), np.float32),
                                    "length": np.array([3 * SR], np.int64)})[0].shape[1]
        self.hop = int(round(3 * SR / n / 160)) * 160  # samples per frame (measured, not assumed)
        self.frame_s = self.hop / SR
        self._by_text = {}
        for i, p in enumerate(self.pieces):
            if i != self.blank and not p.startswith("<"):
                self._by_text.setdefault(p, i)
        self._max_len = max(len(p) for p in self._by_text)

    def native(self, y: np.ndarray) -> np.ndarray:
        """Log posteriors [frames, vocab] of one window."""
        x = np.ascontiguousarray(y, dtype=np.float32)[None]
        return self.session.run(None, {"audio_signal": x, "length": np.array([x.shape[1]], np.int64)})[0][0]

    def columns(self, lp: np.ndarray) -> np.ndarray:
        """Native [T, V] -> adapter columns [T, V + 1]: blank, best non-blank, then each token."""
        other = np.delete(lp, self.blank, axis=1)
        return np.concatenate([lp[:, [self.blank]], other.max(axis=1, keepdims=True), other], axis=1)

    def col(self, token: int) -> int:
        return token + 2 if token < self.blank else token + 1

    def greedy(self, lp: np.ndarray) -> list[int]:
        ids = lp.argmax(1)
        return [int(i) for j, i in enumerate(ids) if i != self.blank and (j == 0 or i != ids[j - 1])]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.pieces[t] for t in tokens).replace(SPACE, " ").strip()

    def tokenize(self, word: str) -> list[int]:
        """Greedy longest-match pieces for one word; the first carries the word marker."""
        out, i, s = [], 0, SPACE + word
        while i < len(s):
            for n in range(min(self._max_len, len(s) - i), 0, -1):
                t = self._by_text.get(s[i: i + n])
                if t is not None:
                    out.append(t)
                    i += n
                    break
            else:
                i += 1  # a character the vocabulary lacks: skipped (counted in validation)
        return out


def corpus_surface_words(ix: CorpusIndex) -> list[str]:
    """Each corpus word (ix.words order) in surface spelling; falls back to the folded text
    where a line's surface split doesn't line up word for word with the index."""
    from .text import normalize

    out = [w.text for w in ix.words]
    for di, dua in enumerate(ix.duas):
        lo, hi = ix.dua_word_span[di]
        for seg in dua.segments:
            idx = [w for w in range(lo, hi) if ix.word_segment[w] == seg.id]
            surf = []
            for tok in seg.arabic.split():
                n = normalize(tok).split()
                surf += [surface(tok)] if len(n) == 1 else n
            surf = [s for s in surf if s]
            if len(surf) == len(idx) and all(normalize(a) == ix.words[w].text for a, w in zip(surf, idx)):
                for a, w in zip(surf, idx):
                    out[w] = a
    return out


class TokenFollower(LocalFollower):
    """LocalFollower over a token-CTC model's columns (TokenCtcModel.columns)."""

    def __init__(self, index: CorpusIndex, model_meta: dict, config: FollowerConfig | None = None,
                 word_tokens: list[list[int]] | None = None):
        super().__init__(index, config)
        self.frame_s = model_meta["frame_s"]
        blank = model_meta["blank"]
        toks = word_tokens if word_tokens is not None else model_meta["word_tokens"]
        cols = [np.array([t + 2 if t < blank else t + 1 for t in ts], dtype=np.int64) for ts in toks]
        lens = np.array([len(c) for c in cols])
        self._tok = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        self._tok_owner = np.repeat(np.arange(len(cols)), lens)
        self._tok_start = np.r_[0, np.cumsum(lens)]

    def _reference(self, anchor: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.ix.dua_word_span[self.ix.word_dua[anchor]]
        a, b = max(lo, anchor - self.cfg.back_words), min(hi, anchor + self.cfg.ahead_words)
        sa, sb = self._tok_start[a], self._tok_start[b]
        return self._tok[sa:sb], self._tok_owner[sa:sb]

    def _jump_search(self, lp: np.ndarray, hmm_word: int, line_mass=None) -> None:
        # The jump search reads the letter model's columns (ix.letters): a token model has none.
        return None


def expand(stored: np.ndarray, cols: np.ndarray, width: int) -> np.ndarray:
    """A dump restricted to `cols` (blank, best non-blank, one du'a's tokens) back to full width."""
    out = np.full(stored.shape[:-1] + (width,), LOG_FLOOR, dtype=np.float32)
    out[..., cols] = stored
    return out
