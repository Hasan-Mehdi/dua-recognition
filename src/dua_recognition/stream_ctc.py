"""The phone's CTC model, made streaming: each step encodes only the new audio.

The window student (ctc_student.py) re-encodes the latest 2 s every 0.1 s: twenty times the
work for each frame, 100-300 ms a step on a phone (phone_power.md), so steps come every
0.25 s and the display waits for them. This model has the same Whisper-base encoder and
letter head, with attention that a stream can cache:

- layer 0 sees `left0` frames back and `lookahead` frames ahead (0.2 s: the right context the
  stream decoder already waits for before it commits a frame, stream_follower.py);
- layers 1-5 see only the past, `left` frames back (3 s), so their keys and values, once
  computed, never change;
- no absolute positions (a stream has no start to count from): a per-head distance penalty
  on the attention scores instead (ALiBi, Press et al. 2022), learnt from its initial slopes;
- log-mel normalized by the running maximum of the last 3 s, not the window's.

A frame is final once `lookahead` frames after it have arrived; the newest `lookahead` frames
are tentative (computed with the future there is, never cached), as the window model's
newest frames are. `forward` on any stretch of audio gives exactly what the stream would
have given at its end: final frames for all but the last `lookahead`, tentative ones there.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .ctc import LOG_FLOOR, N_COLS
from .ctc_student import HOP, N_FFT, SR, WhisperCTC, mel_filters

FRAME = 320  # samples per encoder frame (20 ms)
NORM_FRAMES = 300  # mel frames (3 s) of the running maximum


class CausalLogMel(nn.Module):
    """Whisper's log-mel ([B, N] -> [B, 80, N // 160]), its floor at the running maximum of
    the last 3 s less 8 instead of the whole window's maximum less 8."""

    def __init__(self, n_mels: int = 80):
        super().__init__()
        self.register_buffer("filters", mel_filters(n_mels), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        st = torch.stft(y.float(), N_FFT, HOP, window=self.window, return_complex=True)
        power = st[..., :-1].abs() ** 2
        mel = (power.transpose(1, 2) @ self.filters).transpose(1, 2)
        lg = torch.clamp(mel, min=1e-10).log10()
        top = lg.amax(dim=1)  # [B, M]
        run = nn.functional.max_pool1d(nn.functional.pad(top, (NORM_FRAMES - 1, 0), value=-30.0)[:, None],
                                       NORM_FRAMES, stride=1)[:, 0]
        lg = torch.maximum(lg, run[:, None, :] - 8.0)
        return (lg + 4.0) / 4.0


def frames_ready(n_samples: int) -> int:
    """Encoder frames whose inputs have all arrived after n_samples: frame i reads mel frames
    up to 2i + 2, each of which reads 200 samples past its centre."""
    return max(0, (n_samples - 2 * HOP - N_FFT // 2) // FRAME + 1)


def alibi_slopes(n_heads: int) -> list[float]:
    return [2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)]


class StreamCTC(nn.Module):
    def __init__(self, src: WhisperCTC, lookahead: int = 10, left: int = 150, left0: int = 150):
        super().__init__()
        self.conv1, self.conv2 = src.conv1, src.conv2
        self.layers = src.layers
        self.layer_norm = src.layer_norm
        self.head = src.head
        self.lookahead, self.left, self.left0 = lookahead, left, left0
        att = self.layers[0].self_attn
        self.n_heads, self.head_dim = att.num_heads, att.head_dim
        self.slopes = nn.Parameter(torch.tensor([alibi_slopes(self.n_heads)] * len(self.layers)))

    @classmethod
    def from_whisper(cls, name: str, **kw) -> "StreamCTC":
        return cls(WhisperCTC.from_whisper(name), **kw)

    def frontend(self, mel: torch.Tensor) -> torch.Tensor:
        """As the stream starts (StreamStep): two mel frames of silence before, no padding after,
        so M mel frames give exactly frames_ready(M * 160) encoder frames."""
        mel = torch.cat([torch.full_like(mel[:, :, :2], -1.5), mel], 2)
        x = nn.functional.gelu(nn.functional.conv1d(mel, self.conv1.weight, self.conv1.bias))
        x = nn.functional.gelu(nn.functional.conv1d(x, self.conv2.weight, self.conv2.bias, stride=2))
        return x.transpose(1, 2)

    def _bias(self, li: int, qpos: torch.Tensor, kpos: torch.Tensor, kvalid: torch.Tensor | None = None):
        """Additive scores [.., H, Tq, Tk] for query/key frame positions [.., Tq] / [.., Tk]."""
        rel = qpos[..., :, None] - kpos[..., None, :]  # i - j
        if li == 0:
            ok = (rel >= -self.lookahead) & (rel <= self.left0)
        else:
            ok = (rel >= 0) & (rel <= self.left)
        if kvalid is not None:
            ok = ok & kvalid[..., None, :]
        dist = rel.abs().to(self.slopes.dtype)
        b = -self.slopes[li].abs()[:, None, None] * dist[..., None, :, :]  # [.., H, Tq, Tk]
        return b.masked_fill(~ok[..., None, :, :], float("-inf"))

    def _layer(self, li: int, xq: torch.Tensor, xk: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """One pre-LN encoder layer: queries from xq [B, Tq, D], keys and values from xk [B, Tk, D]."""
        L = self.layers[li]
        a = L.self_attn
        B, Tq, D = xq.shape
        hq = L.self_attn_layer_norm(xq)
        hk = hq if xk is xq else L.self_attn_layer_norm(xk)
        q = (a.q_proj(hq) * a.scaling).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = a.k_proj(hk).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = a.v_proj(hk).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), scale=1.0)
        x = xq + a.out_proj(o.transpose(1, 2).reshape(B, Tq, D))
        return x + L.fc2(L.activation_fn(L.fc1(L.final_layer_norm(x))))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """[B, 80, M] log-mel -> [B, M // 2, n_cols]: what the stream gives at the end of this audio
        (its last frame read padding the stream never has: frames_ready leaves it out)."""
        return self.forward_frames(self.frontend(mel))

    def forward_frames(self, h: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(h.shape[1], device=h.device)
        for li in range(len(self.layers)):
            h = self._layer(li, h, h, self._bias(li, pos, pos))
        return self.head(self.layer_norm(h)).log_softmax(-1)

    # ---- the stream over a whole recording (the bench) ------------------------------------
    @torch.no_grad()
    def final_states(self, x0: torch.Tensor, seg: int = 400) -> list[torch.Tensor]:
        """Every layer's input for all frames of one recording ([T, D] each, the frames final),
        layer by layer in segments: [x0, h1, ..., h6]."""
        hs = [x0]
        T = x0.shape[0]
        for li in range(len(self.layers)):
            prev, out = hs[-1], []
            back = self.left0 if li == 0 else self.left
            ahead = self.lookahead if li == 0 else 0
            for s in range(0, T, seg):
                e = min(T, s + seg)
                k0, k1 = max(0, s - back), min(T, e + ahead)
                qpos = torch.arange(s, e, device=x0.device)
                kpos = torch.arange(k0, k1, device=x0.device)
                out.append(self._layer(li, prev[None, s:e], prev[None, k0:k1], self._bias(li, qpos, kpos))[0])
            hs.append(torch.cat(out))
        return hs

    @torch.no_grad()
    def tentative(self, hs: list[torch.Tensor], ends: list[int], batch: int = 256) -> torch.Tensor:
        """Log posteriors of the newest `lookahead` frames before each end (frames ready), as the
        stream computes them then: [len(ends), R, C]."""
        R = self.lookahead
        dev = hs[0].device
        outs = []
        for b0 in range(0, len(ends), batch):
            E = torch.tensor(ends[b0 : b0 + batch], device=dev)
            qpos = E[:, None] - R + torch.arange(R, device=dev)  # [K, R]
            qvalid = qpos >= 0
            # layer 0: keys are the frontend's frames, up to the end
            back = self.left0
            kpos = E[:, None] - R - back + torch.arange(back + R, device=dev)
            kvalid = kpos >= 0
            x0 = hs[0]
            xq = x0[qpos.clamp(min=0)]
            xk = x0[kpos.clamp(min=0)]
            t = self._layer(0, xq, xk, self._bias(0, qpos, kpos, kvalid))
            for li in range(1, len(self.layers)):
                back = self.left
                kpos = E[:, None] - R - back + torch.arange(back, device=dev)  # final frames before the tentative ones
                kvalid = kpos >= 0
                fk = hs[li][kpos.clamp(min=0)]
                xk = torch.cat([fk, t], 1)
                kp = torch.cat([kpos, qpos], 1)
                kv = torch.cat([kvalid, qvalid], 1)
                t = self._layer(li, t, xk, self._bias(li, qpos, kp, kv))
            outs.append(self.head(self.layer_norm(t)).log_softmax(-1))
        return torch.cat(outs)


INVALID = -1.0e6  # the position of an empty cache slot: further back than any frame can see


class StreamStep(nn.Module):
    """One step of the stream, for ONNX (the page's ctc-worker.js): the audio the new frames need
    in, their final frames and the newest tentative ones out, every cache handed back.

    Inputs (n = new encoder frames, f0 = the first one's index since the session's start):
      audio [320 n + 240]   samples for mel frames 2 f0 + 1 .. 2 (f0 + n) (200 before the first's
                            centre to 200 after the last's), zeros before the session's start
      f0 [1]                float
      mel_c [3, 80]         normalized mel frames 2 f0 - 2 .. 2 f0 (silence: -1.5)
      top_c [299]           the loudest bin of each of the 299 mel frames before 2 f0 + 1 (start: -30)
      x0q [R, D]            layer 0's inputs for frames f0 - R .. f0 - 1 (start: zeros)
      k0c, v0c [H, R+L0, d] layer 0's keys and values for frames f0 - R - L0 .. f0 - 1
      x0p [R + L0]          their positions (INVALID: none yet)
      kc, vc [5, H, L, d]   layers 1-5: keys and values of the last L final frames
      hp [L]                their positions
    Outputs: lp_final [n, C] (frames f0 - R .. f0 + n - R - 1; those before 0 are meaningless),
      lp_tent [R, C] (frames f0 + n - R .. f0 + n - 1), then the caches in the same order.
    """

    def __init__(self, m: StreamCTC):
        super().__init__()
        self.m = m
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
        ang = 2 * math.pi * k[:, None] * n[None, :] / N_FFT
        win = torch.hann_window(N_FFT, dtype=torch.float64)
        dft = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win])  # [402, 400]
        self.register_buffer("dft", dft.float()[:, None, :], persistent=False)  # a strided conv = the STFT
        self.register_buffer("fb", mel_filters().T.contiguous()[:, :, None], persistent=False)  # [80, 201, 1]

    # (linear layers get a batch dimension: on 2-D inputs the export writes Gemm, which the int8
    # quantizer mishandles)
    def _kv(self, li: int, x: torch.Tensor):
        L = self.m.layers[li]
        h = L.self_attn_layer_norm(x[None])
        a = L.self_attn
        H, d = self.m.n_heads, self.m.head_dim
        return (a.k_proj(h)[0].view(-1, H, d).transpose(0, 1), a.v_proj(h)[0].view(-1, H, d).transpose(0, 1))

    def _attend(self, li: int, xq: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos, kpos):
        m = self.m
        L = m.layers[li]
        a = L.self_attn
        H, d = m.n_heads, m.head_dim
        q = (a.q_proj(L.self_attn_layer_norm(xq[None]))[0] * a.scaling).view(-1, H, d).transpose(0, 1)  # [H, Tq, d]
        rel = qpos[:, None] - kpos[None, :]
        ok = ((rel >= -m.lookahead) & (rel <= m.left0)) if li == 0 else ((rel >= 0) & (rel <= m.left))
        s = q @ k.transpose(1, 2) - m.slopes[li].abs()[:, None, None] * rel.abs()[None]
        s = s.masked_fill(~ok[None], -1e9)
        o = (s.softmax(-1) @ v).transpose(0, 1).reshape(-1, H * d)
        x = xq[None] + a.out_proj(o[None])
        return (x + L.fc2(L.activation_fn(L.fc1(L.final_layer_norm(x)))))[0]

    def forward(self, audio, f0, mel_c, top_c, x0q, k0c, v0c, x0p, kc, vc, hp):
        m = self.m
        R, L, Kx = m.lookahead, m.left, m.lookahead + m.left0
        # log-mel of the new frames (Whisper's, the floor at the last 3 s's maximum less 8)
        spec = nn.functional.conv1d(audio[None, None], self.dft, stride=HOP)[0]  # [402, 2n]
        nb = N_FFT // 2 + 1
        power = spec[:nb] ** 2 + spec[nb:] ** 2
        mel = nn.functional.conv1d(power[None], self.fb)[0]  # [80, 2n]
        lg = torch.clamp(mel, min=1e-10).log10()
        tops = torch.cat([top_c, lg.amax(0)])  # [299 + 2n]
        run = nn.functional.max_pool1d(tops[None, None], NORM_FRAMES, stride=1)[0, 0]  # [2n]
        x = (torch.maximum(lg, run[None] - 8.0) + 4.0) / 4.0
        melx = torch.cat([mel_c.T, x], 1)  # [80, 2n + 3]: mel frames 2 f0 - 2 .. 2 (f0 + n)
        c1 = nn.functional.gelu(nn.functional.conv1d(melx[None], m.conv1.weight, m.conv1.bias))
        xn = nn.functional.gelu(nn.functional.conv1d(c1, m.conv2.weight, m.conv2.bias, stride=2))[0].T  # [n, D]
        n = xn.shape[0]
        ar = torch.arange(n, dtype=torch.float32)
        npos = f0 + ar
        X = torch.cat([x0q, xn])  # frames f0 - R .. f0 + n - 1
        k0n, v0n = self._kv(0, xn)
        K0, V0, P0 = torch.cat([k0c, k0n], 1), torch.cat([v0c, v0n], 1), torch.cat([x0p, npos])
        # final frames: f0 - R .. f0 + n - R - 1
        qp = f0 - R + ar
        h = self._attend(0, X[:n], K0, V0, qp, P0)
        hp_new = torch.where(qp >= 0, qp, torch.full_like(qp, INVALID))  # frames before the start: none
        hp_all = torch.cat([hp, hp_new])
        kcs, vcs = [], []
        for li in range(1, len(m.layers)):
            kn, vn = self._kv(li, h)
            Kl, Vl = torch.cat([kc[li - 1], kn], 1), torch.cat([vc[li - 1], vn], 1)
            h = self._attend(li, h, Kl, Vl, qp, hp_all)
            kcs.append(Kl[:, -L:])
            vcs.append(Vl[:, -L:])
        lp_final = m.head(m.layer_norm(h[None]))[0].log_softmax(-1)
        hp2 = hp_all[-L:]
        # tentative: the newest R frames, with the future there is
        tp = f0 + n - R + torch.arange(R, dtype=torch.float32)
        t = self._attend(0, X[-R:], K0, V0, tp, P0)
        for li in range(1, len(m.layers)):
            kt, vt = self._kv(li, t)
            Kl = torch.cat([kcs[li - 1], kt], 1)
            Vl = torch.cat([vcs[li - 1], vt], 1)
            t = self._attend(li, t, Kl, Vl, tp, torch.cat([hp2, tp]))
        lp_tent = m.head(m.layer_norm(t[None]))[0].log_softmax(-1)
        return (lp_final, lp_tent, melx.T[-3:], tops[-(NORM_FRAMES - 1):], X[-R:], K0[:, -Kx:], V0[:, -Kx:],
                P0[-Kx:], torch.stack(kcs), torch.stack(vcs), hp2)

    def init_state(self):
        m = self.m
        H, d, D = m.n_heads, m.head_dim, m.layers[0].self_attn.embed_dim
        R, L, Kx, nl = m.lookahead, m.left, m.lookahead + m.left0, len(m.layers) - 1
        return {"mel_c": torch.full((3, 80), -1.5), "top_c": torch.full((NORM_FRAMES - 1,), -30.0),
                "x0q": torch.zeros(R, D), "k0c": torch.zeros(H, Kx, d), "v0c": torch.zeros(H, Kx, d),
                "x0p": torch.full((Kx,), INVALID), "kc": torch.zeros(nl, H, L, d), "vc": torch.zeros(nl, H, L, d),
                "hp": torch.full((L,), INVALID)}


STATE_NAMES = ["mel_c", "top_c", "x0q", "k0c", "v0c", "x0p", "kc", "vc", "hp"]


def run_steps(step: StreamStep, y: np.ndarray, ends_samples: list[int], run=None) -> list[tuple]:
    """Play a recording through StreamStep as the page does: at each sample count, the frames
    ready then. `run(feeds) -> outputs` (an onnxruntime session's) or the torch module.
    Returns per call (frames ready, lp_final, lp_tent)."""
    st = {k: v.numpy() for k, v in step.init_state().items()}
    f0 = 0
    out = []
    for e in ends_samples:
        F = frames_ready(e)
        n = F - f0
        if n <= 0:
            continue
        a, b = HOP * (2 * f0 + 1) - N_FFT // 2, HOP * 2 * F + N_FFT // 2
        chunk = np.zeros(b - a, np.float32)
        lo, hi = max(0, a), min(len(y), b)
        chunk[lo - a : hi - a] = y[lo:hi]
        feeds = {"audio": chunk, "f0": np.array([f0], np.float32), **st}
        if run is None:
            with torch.no_grad():
                res = step(*[torch.tensor(feeds[k]) for k in ["audio", "f0"] + STATE_NAMES])
            res = [r.numpy() for r in res]
        else:
            res = run(feeds)
        out.append((F, res[0], res[1]))
        st = dict(zip(STATE_NAMES, res[2:]))
        f0 = F
    return out


def export_step(model: StreamCTC, onnx_path, quantized_path=None) -> dict:
    """StreamStep to ONNX (and int8 MatMuls); returns what the page's meta.json needs about it."""
    import onnx

    step = StreamStep(model.eval()).eval()
    init = step.init_state()
    feeds = {"audio": torch.zeros(320 * 5 + 240), "f0": torch.zeros(1), **init}
    names_in = ["audio", "f0"] + STATE_NAMES
    names_out = ["lp_final", "lp_tent"] + [k + "_out" for k in STATE_NAMES]
    torch.onnx.export(step, tuple(feeds[k] for k in names_in), str(onnx_path), input_names=names_in,
                      output_names=names_out, opset_version=17, dynamo=False,
                      dynamic_axes={"audio": {0: "samples"}, "lp_final": {0: "frames"}})
    # The export reaches weights used twice (final and tentative frames) through Identity nodes; the
    # quantizer only quantizes a MatMul whose weight is an initializer, so point them at it directly.
    g = onnx.load(str(onnx_path))
    inits = {i.name for i in g.graph.initializer}
    alias = {n.output[0]: n.input[0] for n in g.graph.node if n.op_type == "Identity" and n.input[0] in inits}
    keep = [n for n in g.graph.node if not (n.op_type == "Identity" and n.output[0] in alias)]
    for n in keep:
        for k, x in enumerate(n.input):
            if x in alias:
                n.input[k] = alias[x]
    del g.graph.node[:]
    g.graph.node.extend(keep)
    onnx.save(g, str(onnx_path))
    if quantized_path:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QUInt8,
                         op_types_to_quantize=["MatMul"], extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    return {"arch": "stream", "frame_s": 0.02, "states": {k: list(v.shape) for k, v in init.items()},
            "n_heads": model.n_heads, "head_dim": model.head_dim, "n_cols": int(model.head.out_features),
            "stream": {"lookahead": model.lookahead, "left": model.left, "left0": model.left0}}


class PageStream:
    """web/ctc-stream.js in Python (tests/test_ctc_stream.py holds them to the same frames): the page
    sends the latest window and its end (samples since the session's start), gets the latest 2 s."""

    OUT = 100

    def __init__(self, step: StreamStep, run=None):
        self.step_mod, self.run, self.s = step, run, None

    def _call(self, feeds):
        if self.run is not None:
            return self.run(feeds)
        with torch.no_grad():
            res = self.step_mod(*[torch.tensor(feeds[k]) for k in ["audio", "f0"] + STATE_NAMES])
        return [r.numpy() for r in res]

    def step(self, audio: np.ndarray, end: int):
        s = self.s
        if s is None or end <= s["end"] or end - s["end"] > len(audio):
            start = end - len(audio)
            f0 = max(0, -(-(start + 40) // 320))
            s = self.s = {"st": {k: v.numpy() for k, v in self.step_mod.init_state().items()}, "f0": f0, "end": end,
                          "hs": start, "hist": audio.astype(np.float32).copy(), "finals": [], "first": f0, "tent": None}
        else:
            s["hist"] = np.r_[s["hist"], audio[len(audio) - (end - s["end"]):]].astype(np.float32)
            s["end"] = end
        R = self.step_mod.m.lookahead
        F = frames_ready(end)
        n = F - s["f0"]
        if n > 0:
            a, b = HOP * (2 * s["f0"] + 1) - N_FFT // 2, HOP * 2 * F + N_FFT // 2
            chunk = np.zeros(b - a, np.float32)
            lo = max(a, s["hs"])
            chunk[lo - a:] = s["hist"][lo - s["hs"]: b - s["hs"]]
            # positions from the stream's first frame (the frames before it stay out of the caches)
            res = self._call({"audio": chunk, "f0": np.array([s["f0"] - s["first"]], np.float32), **s["st"]})
            s["st"] = dict(zip(STATE_NAMES, res[2:]))
            for i in range(n):
                if s["f0"] - R + i >= s["first"]:
                    s["finals"].append(res[0][i])
            s["finals"] = s["finals"][-self.OUT:]
            s["tent"] = res[1]
            s["f0"] = F
            keep = HOP * (2 * F + 1) - N_FFT // 2
            if keep > s["hs"]:
                s["hist"] = s["hist"][keep - s["hs"]:]
                s["hs"] = keep
        fin = s["finals"][-(self.OUT - R):]
        C = int(self.step_mod.m.head.out_features)
        tent = s["tent"]
        if tent is None:
            tent = np.full((R, C), LOG_FLOOR, np.float32)
            tent[:, 0] = 0.0
        return np.concatenate([np.array(fin, np.float32).reshape(-1, C), tent])


class StreamStudent:
    """A trained StreamCTC with the bench's interface: `stream(y, times, window)` gives, for each
    step time, the latest `window` seconds of frames as the stream has them then (final, then
    the tentative newest), shaped like StudentCtc.windows."""

    def __init__(self, path, device: str | None = None):
        import json
        from pathlib import Path

        path = Path(path)
        ck = torch.load(path / "student.pt", map_location="cpu", weights_only=False)
        meta = ck["meta"]
        self.meta = meta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        a = meta["stream"]
        self.model = StreamCTC.from_whisper(meta["base"], lookahead=a["lookahead"], left=a["left"], left0=a["left0"])
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.mel = CausalLogMel().to(self.device)
        self.window_s = float(meta.get("window_s", 2.0))
        if not (path / "meta.json").exists():
            (path / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    @torch.no_grad()
    def stream(self, y: np.ndarray, times: list[float], window: float):
        m = self.model
        yt = torch.tensor(np.ascontiguousarray(y, dtype=np.float32), device=self.device)[None]
        mel = self.mel(yt)
        x0 = m.frontend(mel)[0]
        hs = m.final_states(x0)
        final = m.head(m.layer_norm(hs[-1])).log_softmax(-1).float()  # [T, C]
        ends = [min(frames_ready(int(round(t * SR))), x0.shape[0]) for t in times]
        tent = m.tentative(hs, ends).float()  # [K, R, C]
        W = int(round(window / 0.02))
        R = m.lookahead
        arr = np.full((len(times), W, N_COLS), LOG_FLOOR, dtype=np.float16)
        lens = np.zeros(len(times), np.int32)
        fin = final.cpu().numpy()
        ten = tent.cpu().numpy()
        for k, e in enumerate(ends):
            if e <= 0:
                continue
            a = max(0, e - W)
            n = e - a
            row = np.empty((n, N_COLS), np.float32)
            cut = max(a, e - R)
            row[: cut - a] = fin[a:cut]
            row[cut - a :] = ten[k, R - (e - cut) :]
            # frames before the recording's start: silence, as the window model's zeroed buffer
            arr[k, W - n :] = np.maximum(row, LOG_FLOOR)
            arr[k, : W - n, 0] = 0.0
            lens[k] = W
        return arr, lens

    def windows(self, y: np.ndarray, times: list[float], window: float, batch: int = 16):
        arr, lens = self.stream(y, times, window)
        return arr, lens, [""] * len(times), None

    @torch.no_grad()
    def compact(self, y: np.ndarray, times: list[float]) -> dict:
        """The same stream, stored once: every final frame, and per step its tentative newest frames and
        how many frames were ready (`window_rows` rebuilds a step's window): ~8x smaller than windows."""
        m = self.model
        yt = torch.tensor(np.ascontiguousarray(y, dtype=np.float32), device=self.device)[None]
        x0 = m.frontend(self.mel(yt))[0]
        hs = m.final_states(x0)
        final = m.head(m.layer_norm(hs[-1])).log_softmax(-1).float().cpu().numpy()
        ends = [min(frames_ready(int(round(t * SR))), x0.shape[0]) for t in times]
        tent = m.tentative(hs, ends).float().cpu().numpy()
        return {"final": np.maximum(final, LOG_FLOOR).astype(np.float16),
                "tent": np.maximum(tent, LOG_FLOOR).astype(np.float16), "ends": np.array(ends, np.int32)}


def load_frames(path, rows: int = 100):
    """A bench CTC cache file as (lp [steps, frames, C], n_frames, t), whichever way it was stored: windows
    (lp, n_frames, t) or a streaming model's compact form (final, tent, ends, t), expanded to windows of
    `rows` frames."""
    z = np.load(path)
    if "final" not in z.files:
        return z["lp"], z["n_frames"], z["t"]
    fin, ten, ends = z["final"], z["tent"], z["ends"]
    lp = np.stack([window_rows(fin, ten[k], int(ends[k]), rows) for k in range(len(ends))]) if len(ends) else \
        np.zeros((0, rows, fin.shape[1]), np.float32)
    return lp, np.full(len(ends), rows, np.int32), z["t"]


def window_rows(final: np.ndarray, tent: np.ndarray, end: int, rows: int = 100) -> np.ndarray:
    """One step's window from a compact stream cache: the `rows` frames before `end` (final, then the
    step's tentative ones), blank before the recording's start."""
    R = tent.shape[0]
    out = np.full((rows, final.shape[1]), LOG_FLOOR, np.float32)
    out[:, 0] = 0.0
    if end <= 0:
        return out
    a = max(0, end - rows)
    cut = max(a, end - R)
    n = end - a
    seg = np.concatenate([final[a:cut], tent[R - (end - cut):]]).astype(np.float32)
    out[rows - n:] = seg
    return out


def _check_stream(model: StreamCTC, seconds: float = 7.0, seed: int = 0) -> float:
    """Max abs difference between the stream's frames at a few ends and `forward` on the audio
    up to each end (they must agree: the cache changes nothing)."""
    g = torch.Generator().manual_seed(seed)
    y = 0.1 * torch.randn(int(seconds * SR), generator=g)
    mel_fn = CausalLogMel()
    worst = 0.0
    x0 = model.frontend(mel_fn(y[None]))[0]
    hs = model.final_states(x0, seg=97)
    final = model.head(model.layer_norm(hs[-1])).log_softmax(-1)
    for n in (int(0.44 * len(y)), int(0.79 * len(y)), len(y)):
        e = frames_ready(n)
        # forward on the audio up to n: frames up to e are those whose inputs are all there
        ref = model.forward_frames(model.frontend(mel_fn(y[None, :n]))[:, :e])[0]
        tent = model.tentative(hs, [e])[0]
        R = model.lookahead
        worst = max(worst, float((final[: e - R] - ref[: e - R]).abs().max()),
                    float((tent - ref[e - R : e]).abs().max()))
    return worst
