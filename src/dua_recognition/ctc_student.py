"""A small CTC model for the phone: a Whisper encoder with a letter head.

The word follower (follower.py) needs timed letter evidence several times a second.
wav2vec2-quran-dua gives it but has ~300 M parameters; a phone can't run it. This is
a Whisper encoder (whisper-base: 20 M parameters, the encoder of the phone's own
fine-tuned model to start from) reading a few seconds of audio, with one linear
layer onto the corpus alphabet (ctc.py: column 0 = blank, one per align._ALPHABET
letter). Whisper's encoder emits one vector per 20 ms, the same frame rate as
wav2vec2, so the follower takes either. scripts/train_ctc_student.py trains it
from the wav2vec2 teacher (scripts/ctc_teacher.py); scripts/export_ctc_student.py
puts it in the browser.

The input is any length (Whisper itself insists on 30 s, or 8 s for the cut
models): the log-mel features are Whisper's, normalized per window as Whisper's
feature extractor does (log10, floor at the window's max - 8, (x + 4) / 4).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ctc import LOG_FLOOR, N_COLS

SR = 16000
HOP = 160
N_FFT = 400


def mel_filters(n_mels: int = 80) -> torch.Tensor:
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=n_mels)
    return torch.tensor(np.asarray(fe.mel_filters, dtype=np.float32))  # [n_fft//2+1, n_mels]


class LogMel(nn.Module):
    """Whisper's log-mel spectrogram, batched, for any length: [B, N] -> [B, 80, N // 160]."""

    def __init__(self, n_mels: int = 80):
        super().__init__()
        self.register_buffer("filters", mel_filters(n_mels), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        st = torch.stft(y.float(), N_FFT, HOP, window=self.window, return_complex=True)  # center, reflect
        power = st[..., :-1].abs() ** 2  # Whisper drops the last frame
        mel = (power.transpose(1, 2) @ self.filters).transpose(1, 2)  # [B, n_mels, F]
        lg = torch.clamp(mel, min=1e-10).log10()
        lg = torch.maximum(lg, lg.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (lg + 4.0) / 4.0


class WhisperCTC(nn.Module):
    """Whisper encoder (any input length up to its position table) + a linear CTC head."""

    def __init__(self, encoder: nn.Module, n_cols: int = N_COLS):
        super().__init__()
        self.conv1, self.conv2 = encoder.conv1, encoder.conv2
        self.pos = nn.Parameter(encoder.embed_positions.weight.detach().clone(), requires_grad=False)
        self.layers = encoder.layers
        self.layer_norm = encoder.layer_norm
        self.head = nn.Linear(self.layer_norm.normalized_shape[0], n_cols)

    @classmethod
    def from_whisper(cls, name: str, max_positions: int | None = None) -> "WhisperCTC":
        from transformers import WhisperForConditionalGeneration

        w = WhisperForConditionalGeneration.from_pretrained(name, attn_implementation="eager")
        m = cls(w.model.encoder)
        if max_positions:
            m.pos = nn.Parameter(m.pos[:max_positions].clone(), requires_grad=False)
        return m

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, 80, F] log-mel -> [B, F // 2, n_cols] log posteriors."""
        x = nn.functional.gelu(self.conv1(mel))
        x = nn.functional.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.pos[: x.shape[1]]
        for layer in self.layers:
            x = layer(x, None)
            if isinstance(x, tuple):
                x = x[0]
        return self.head(self.layer_norm(x)).log_softmax(-1)


def n_frames(n_samples: int) -> int:
    """Frames the student emits for this many samples."""
    return (n_samples // HOP + 1) // 2


def save(model: WhisperCTC, out: Path, meta: dict) -> None:
    import json

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": meta}, out / "student.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


class StudentCtc:
    """The trained student, with the same window interface as ctc.CtcModel (dump_ctc.py,
    follow_eval.py and the server take either)."""

    def __init__(self, path: str | Path, device: str | None = None):
        import json

        path = Path(path)
        ck = torch.load(path / "student.pt", map_location="cpu", weights_only=False)
        meta = ck["meta"]
        self.meta = meta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WhisperCTC.from_whisper(meta["base"], meta.get("max_positions"))
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.mel = LogMel().to(self.device)
        self.window_s = float(meta.get("window_s", 3.0))
        self.torch = torch
        (path / "meta.json").write_text(json.dumps(meta, indent=1))

    def _run(self, wins: list[np.ndarray]):
        torch = self.torch
        # Shorter than the model's window (a session's first seconds): silence before, as the page's
        # zeroed buffer gives it.
        n = int(round(self.window_s * SR))
        wins = [np.r_[np.zeros(n - len(w), np.float32), w] if 0 < len(w) < n else w for w in wins]
        lens = [n_frames(len(w)) for w in wins]
        outs = []
        with torch.no_grad():
            # Same-length windows run as one batch; others one at a time.
            groups: dict[int, list[int]] = {}
            for i, w in enumerate(wins):
                groups.setdefault(len(w), []).append(i)
            res: dict[int, np.ndarray] = {}
            for n, idx in groups.items():
                if n < N_FFT:
                    for i in idx:
                        res[i] = np.zeros((0, N_COLS), dtype=np.float32)
                    continue
                for b in range(0, len(idx), 64):
                    sub = idx[b : b + 64]
                    y = torch.tensor(np.stack([wins[i] for i in sub]), device=self.device)
                    with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                        lp = self.model(self.mel(y)).float()
                    for k, i in enumerate(sub):
                        res[i] = lp[k].cpu().numpy()
            outs = [res[i] for i in range(len(wins))]
        return outs, lens

    def windows(self, y: np.ndarray, times: list[float], window: float, batch: int = 16):
        """As CtcModel.windows: folded log posteriors [n, frames, N_COLS] (float16, padded
        with LOG_FLOOR), frame counts, greedy transcripts, and the windows."""
        from .align import _ALPHABET

        inv = {v: k for k, v in _ALPHABET.items()}
        wins = [np.ascontiguousarray(y[int(max(0.0, t - window) * SR) : int(t * SR)], dtype=np.float32) for t in times]
        outs, _ = self._run(wins)
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
        outs, _ = self._run([np.ascontiguousarray(y, dtype=np.float32)])
        return np.maximum(outs[0], LOG_FLOOR).astype(np.float16).astype(np.float32)


def check_mel(seconds: float = 3.0) -> float:
    """Max abs difference between LogMel and Whisper's feature extractor on noise + tone."""
    from transformers import WhisperFeatureExtractor

    n = int(seconds * SR)
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(n) + 0.3 * np.sin(2 * math.pi * 440 * np.arange(n) / SR)).astype(np.float32)
    fe = WhisperFeatureExtractor(feature_size=80, chunk_length=int(math.ceil(seconds)))
    fe.n_samples = n
    fe.nb_max_frames = n // HOP
    ref = fe(y, sampling_rate=SR, return_tensors="np").input_features[0]
    ours = LogMel()(torch.tensor(y)[None])[0].numpy()
    return float(np.abs(ref[:, : ours.shape[1]] - ours).max())


def load_ctc(name: str):
    """A student folder (student.pt), a server model (server_ctc.pt) or a Hugging Face CTC model, behind the
    same window interface."""
    if (Path(name) / "student.pt").exists():
        import json

        meta = Path(name) / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("arch") == "stream":  # stream_ctc.py
            from .stream_ctc import StreamStudent

            return StreamStudent(name)
        return StudentCtc(name)
    if (Path(name) / "server_ctc.pt").exists():  # a bigger model for the server engine (server_ctc.py)
        from .server_ctc import ServerCtc

        return ServerCtc(name)
    from .ctc import CtcModel

    return CtcModel(name)
