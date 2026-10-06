#!/usr/bin/env python
"""Dump the real data the explainer animation draws.

One moment of Dua Tawassul by a held-out reciter (Hussein Ghareeb), at t = 296 s, as
the reciter says a refrain that recurs 14 times in the du'a:

- the sound: a loudness envelope, a log-mel spectrogram of the 6 s window and a
  few milliseconds of samples (pictures of the audio, not the audio itself);
- the evidence: what Whisper heard, and its edit-distance cost ending at every word;
- the tracker: belief before the step, prediction, evidence and posterior, plus a
  cold-start identification run;
- the word follower's view of the next 6 s: the phone's letter model, frame by frame,
  and the forced-aligned word timings.

    python scripts/transcribe_windows.py --model models/whisper-base-quran-dua-ct2 --tag whisper-base-quran-dua --test-only
    python docs/anim/dump_explainer_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402

STEP = 295  # window ending at t = 296 s: "يا وجيه عند الله"
CTC_MODEL = ROOT / "models" / "ctc-student-base-v6"
# The letter model's frames drawn for the word follower: "ishfa' lana 'inda", words 4-6 of line 38.
# (The last word, allah, is held for three seconds and the model hears no letters in it.)
WORD_SPAN = (297.0, 301.6)
WORDS = (4, 7)


def letter_frames(y: np.ndarray, t0: float, t1: float, window: float = 2.0, hop: float = 0.1) -> np.ndarray:
    """The phone's letter model as it runs live: a 2 s window every 0.1 s, keeping the
    newest 0.1 s of frames. Returns letter probabilities per 20 ms frame [frames, N_COLS]
    (column 0 = blank)."""
    from dua_recognition.ctc_student import SR, StudentCtc

    model = StudentCtc(CTC_MODEL)
    keep = int(round(hop / 0.02))
    frames = [model.window(y[int((t - window) * SR) : int(t * SR)])[-keep:]
              for t in np.arange(t0 + hop, t1 + 1e-6, hop)]
    return np.exp(np.concatenate(frames))


def ctc_path(p: np.ndarray, letters: np.ndarray) -> np.ndarray:
    """Best CTC path of `letters` through the frames (Viterbi, blanks between letters,
    free end): for each frame, the letter index it sits on (-1 before the first)."""
    lp = np.log(np.maximum(p, 1e-9))
    states = np.r_[[0], np.ravel(np.c_[letters, np.zeros_like(letters)])]  # blank, l1, blank, l2, ...
    n, s = lp.shape[0], len(states)
    score = np.full((n, s), -np.inf)
    back = np.zeros((n, s), dtype=np.int32)
    score[0, :2] = lp[0, states[:2]]
    for t in range(1, n):
        for j in range(s):
            cand = [(score[t - 1, j], j)]
            if j >= 1:
                cand.append((score[t - 1, j - 1], j - 1))
            if j >= 2 and states[j] != 0 and states[j] != states[j - 2]:
                cand.append((score[t - 1, j - 2], j - 2))
            best, k = max(cand)
            score[t, j], back[t, j] = best + lp[t, states[j]], k
    j = int(score[-1].argmax())
    path = np.zeros(n, dtype=np.int32)
    for t in range(n - 1, -1, -1):
        path[t] = j
        j = back[t, j]
    return (path - 1) // 2  # state -> letter index (blank after letter k counts as k)


def main() -> None:
    duas = ev.load_all()
    ix = ev.CorpusIndex(duas)
    dua = duas["dua-tawassul"]
    rec = next(r for r in ev.load_recordings(dua) if r.reciter == "Hussein Ghareeb")
    rows = ev.load_rows("whisper-base-quran-dua", rec, 6.0, 1.0)
    lo, hi = ix.dua_word_span[ix.dua_ids.index("dua-tawassul")]
    costs = [ix.word_costs(t) if t else None for _, t in rows[: STEP + 8]]

    tr = ev.Tracker(ix)
    seq = []
    for k in range(STEP):
        tr.update_costs(costs[k], 1.0)
        if k >= STEP - 15:
            seq.append(tr.post[lo:hi].copy())
    before = tr.post.copy()
    tr._advance(1.0, locked=tr._locked())
    pred = tr.post.copy()
    kappa = tr.cfg.kappa if tr._locked() else tr.cfg.kappa_search
    c = costs[STEP].astype(float)
    lik = np.exp(-kappa * (c - c.min()))
    post = pred * lik
    post /= post.sum()
    seg = ix.word_segment[lo:hi]

    cold = ev.Tracker(ix)
    m0 = np.bincount(ix.word_dua, weights=cold.post, minlength=len(ix.duas))  # before anything is heard
    ident = []
    for k in range(STEP, STEP + 8):
        cold.update_costs(costs[k], 1.0)
        m = np.bincount(ix.word_dua, weights=cold.post, minlength=len(ix.duas))
        ident.append([[duas[ix.dua_ids[i]].name_en, float(m[i])] for i in np.argsort(-m)[:4]])

    # Per-line views (bar charts): belief mass per line, best evidence per line.
    segs = np.unique(seg)
    per_line = {name: np.array([v[lo:hi][seg == s].sum() for s in segs]) for name, v in
                (("pred_line", pred), ("post_line", post), ("before_line", before))}
    per_line["lik_line"] = np.array([lik[lo:hi][seg == s].max() for s in segs])

    # The audio itself isn't redistributed: a loudness envelope (50 ms bins), the 6 s
    # window's log-mel spectrogram (8-bit, 20 ms columns) and 30 ms of samples.
    import torch
    from faster_whisper.audio import decode_audio

    from dua_recognition.ctc_student import LogMel

    t_end = rows[STEP][0]
    audio = decode_audio(str(rec.path), sampling_rate=16000)
    y = audio[int((t_end - 16) * 16000) : int(t_end * 16000)]
    env = np.abs(y[: len(y) // 800 * 800]).reshape(-1, 800).max(axis=1)
    win = audio[int((t_end - 6) * 16000) : int(t_end * 16000)]
    mel = LogMel()(torch.tensor(win)[None])[0].numpy()  # [80, 600], ~[-1, 1.5]
    mel = mel[:, : mel.shape[1] // 2 * 2].reshape(80, -1, 2).mean(-1)
    mel = np.clip((mel - mel.min()) / (mel.max() - mel.min()) * 255, 0, 255).astype(np.uint8)
    loud = int(np.abs(win).argmax())
    samples = win[loud - 240 : loud + 240]

    from dua_recognition.align import encode

    ctc = letter_frames(audio, *WORD_SPAN)
    truth = json.loads((ROOT / "data" / "cache" / "word_truth" / f"{rec.audio_id}.json").read_text(encoding="utf-8"))
    words = [(ix.words[lo + k].token, ix.words[lo + k].text, s, e) for k, s, e, _ in truth["words"]
             if ix.words[lo + k].segment == 38]
    said = [w for _, w, _, _ in words[slice(*WORDS)]]
    letters = np.concatenate([encode(w) for w in said])
    path = ctc_path(ctc, letters)

    np.savez(HERE / "explainer_data.npz", before=before[lo:hi], pred=pred[lo:hi], lik=lik[lo:hi],
             post=post[lo:hi], seg=seg, seq=np.array(seq), segs=segs, env=env / env.max(), cost=c[lo:hi],
             mel=mel, samples=samples / np.abs(samples).max(), ctc=ctc.astype(np.float16), ctc_letters=letters,
             ctc_path=path, **per_line)
    line38 = next(x for x in dua.segments if x.id == 38)
    meta = {
        "text": rows[STEP][1],
        "truth_seg": int(rec.segment_at(rows[STEP][0])),
        "argmax_seg": int(seg[post[lo:hi].argmax()]),
        "kappa": kappa,
        "refrain_segs": sorted(int(s) for s in ev.refrain_ids(dua)),
        "ident": ident,
        # The shown du'as' share before anything is heard: the popularity prior, not 1 / n_texts.
        "prior": {duas[d].name_en: float(m0[i]) for i, d in enumerate(ix.dua_ids)
                  if duas[d].name_en in {n for step in ident[:2] for n, _ in step}},
        "prior_all": [round(float(v), 7) for v in m0],  # every text's, in dua_index order
        "lines": {int(x.id): x.arabic for x in dua.segments if 31 <= x.id <= 38},
        "line38": {"arabic": line38.arabic, "translit": line38.transliteration, "english": line38.translation},
        "words38": words,
        "word_span": WORD_SPAN,
        "said_words": WORDS,
        "line_starts": [(s, i) for s, i in rec.starts if 36 <= i <= 39],
        "t_end": t_end,
        "samples_t": t_end - 6 + loud / 16000,
        "dua_index": ix.dua_ids.index("dua-tawassul"),
        "n_texts": len(duas),
        "n_words": ix.n_words,
        "n_lines": len(dua.segments),
        "corpus_min_cost_hits": int((costs[STEP] == costs[STEP].min()).sum()),
    }
    (HERE / "explainer_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"line {meta['argmax_seg']} (human label {meta['truth_seg']})")

    # Real audio for a local, with-sound render (DUA_EXPLAINER_AUDIO=1). It goes to the
    # ignored cache: the recordings belong to DuaPlayer and its reciters.
    import subprocess
    media = ROOT / "data" / "cache" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for name, start, secs in (("explainer_hook.wav", t_end - 11.5, 11.5), ("explainer_window.wav", t_end - 6, 6.0),
                              ("explainer_words.wav", WORD_SPAN[0], WORD_SPAN[1] - WORD_SPAN[0])):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{secs:.2f}",
                        "-i", str(rec.path), "-af", f"afade=t=in:d=0.3,afade=t=out:st={secs - 0.8:.2f}:d=0.8",
                        "-ac", "1", "-ar", "44100", str(media / name)], check=True)


if __name__ == "__main__":
    main()
