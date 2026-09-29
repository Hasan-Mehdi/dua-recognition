# The phone's speech gate (2026-09-26): what it throws away, and three policies

Run `rel-20260926` (manifest: `data/cache/reliability/rel-20260926/manifest.json`, written
before any result below). Development data only: DuaPlayer train reciters. Nothing here
touches the test split.

## Why this was measured

The phone app runs Whisper in a web worker (`web/asr-worker.js`). Before this change,
the worker only transcribed a window if Silero heard a speech run in its last 1.5 s
(`speechInTail`). That check was separate from the energy fallback that `quietAtEnd`
(pause handling) had gained, so the page could say "the reciter is making sound" while
still skipping Whisper and sending an empty transcript. The server path doesn't gate
at all (`Recognizer(vad=False)`). So none of the offline or server results said anything
about how much speech the *phone* discards.

## What changed in the code

- **`web/gate.js`** (new): the gate as plain functions over 16 kHz audio. Silero is passed
  in, so the page and a Node replay run the same code. It has three policies (`?gate=`):

  | policy | runs Whisper when |
  |---|---|
  | `legacy` (**default, unchanged**) | Silero finds a speech run (≥ 7 frames ≥ 0.35, peak ≥ 0.5) in the last 1.5 s |
  | `energy_assisted` | legacy, **or** ≥ 7 consecutive 32 ms frames in the last 1.5 s at ≥ 6 dB over the window's floor (10th-percentile frame energy, never below −70 dBFS) |
  | `ungated` | always (Silero not consulted) |

  All three keep the −45 dBFS tail floor. That is a level threshold, not "digital
  silence": quiet input can fall under it.
- **`asr-worker.js`** uses `gate.js` and reports, per hop, the policy, the decision and why:
  - `below_floor` or `vad_reject`: skipped before Whisper;
  - `empty`: Whisper ran and wrote nothing;
  - `hallucination`: filtered (only with `?filter=1`);
  - `error`: inference failed.

  It also reports input level, Silero run/peak, energy run, inference time, and receive/done
  times as epoch milliseconds (worker and page clocks have different origins, so only these
  are comparable).
- **`capture-worklet.js`** stamps each chunk with its AudioContext time. `app.js` maps that
  to performance time with `getOutputTimestamp()`, which gives each hop's capture time on the
  audio clock (microphone hardware latency before the AudioContext is not included).
  - Each hop logs `captured_ms`, `sent_ms`, `recv_ms`, `done_ms`, `queue_ms` and a `cold` flag.
  - A `visible` event, taken at the next animation frame, carries `age_ms`: capture of the
    window's last sample to paint.
  - Old session files still load, and `scripts/session_report.py` prints a gate/latency
    section only for sessions that have these fields.
- The browser still has **no hallucination filter**; the Python one
  (`asr._looks_hallucinated`) is ported to `gate.js` for logging only, and applied only
  with `?filter=1`: a separate ablation, not changed together with the gate.

## How it was measured

- **Gate decisions:** `scripts/gate_replay.mjs` runs `web/gate.js` itself in Node
  (onnxruntime-node, the same `silero_vad_v6.onnx`), one 6 s window per 1 s of audio, as
  the device engine sends them.
  - **Parity:** on a development recording, 1,828 of 1,828 legacy decisions match Python's
    `speech_in_tail`.
  - `tests/test_gate.py` drives the page code through Node: quiet input reports quiet; a
    Silero rejection with an energy rise takes the energy branch; stationary noise stays
    gated; the hallucination check matches Python.
- **Transcripts:** the phone checkpoint (`whisper-base-aug-v4`) through CTranslate2, from
  the ungated window cache. A hop's text is used only if the policy lets it run; otherwise
  the tracker gets `""`, as on the page.
  - That is a different runtime from the page's ONNX q8. The browser runtime is timed
    separately (below).
- **Tracker:** Python, which is parity-tested with `tracker.js`, in page mode with a 0.5 s delay.
- **Data:** `scripts/gate_eval.py`:
  - 23 train recordings (6.0 h; line scores on 22: Ashura's line labels are unreliable);
  - the same audio at −10 and −20 dB (gate decisions only);
  - the train pause set: 4 s of each recording's own room tone after every third line end,
    2,316 pause hops;
  - the 13 saved debug sessions (unlabelled; gate statistics only).
- **Labels:**
  - Line accuracy, onset lag and missed transitions use DuaPlayer's **human** line starts.
  - "Annotated speech" for the discarded-speech measure uses the **automatic** forced-aligned
    words, so that measure is an automatic-label diagnostic.

## Results

### Discarded speech (share of annotated-speech seconds whose hop was skipped)

| input level | legacy | energy_assisted | ungated |
|---|---:|---:|---:|
| as recorded | 3.8% | 0.75% | 0.10% |
| −10 dB | 8.0% | 1.4% | 0.45% |
| −20 dB | 23.7% | 11.2% | 10.1% |

- **Why hops are skipped:** at the recorded level, legacy skips 1,155 of 21,778 hops as
  `vad_reject` (Silero rejects audio that is above the floor) and 127 as `below_floor`.
  Energy-assisted leaves 257 `vad_reject`.
- **Paired per recording, grouped by reciter (bootstrap 95%):** energy-assisted −3.4 pts of
  speech discarded [−6.0, −1.9], ungated −4.9 pts [−9.4, −2.4].
- **Quieter input:** at −20 dB most of what remains is the −45 dBFS floor, which no policy
  changes. That is the case for a separate floor comparison, not for opening the gate further.

### Line following (22 recordings, 3,173 human line transitions)

| policy | line exact | ±1 | wrong du'a | onset lag median / p90 | early > 0.3 s | missed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 82.0% | 96.4% | 0.20% | +0.07 / +1.61 s | 31.9% | 2.3% |
| energy_assisted | 82.0% | 96.3% | 0.23% | +0.10 / +1.60 s | 30.7% | 2.2% |
| ungated | 82.0% | 96.3% | 0.23% | +0.10 / +1.60 s | 30.5% | 2.3% |

Paired against legacy:
- **Energy-assisted:** line exact −0.1 pt [−0.6, +0.3]; wrong du'a +0.28 pt [0.0, +0.64].
- **Ungated:** line exact −0.1 pt [−0.7, +0.5]; wrong du'a the same as energy-assisted.

The hallucination filter changed nothing on this data: every `+filter` row is identical to
its unfiltered row, because the fine-tuned checkpoint produced no filtered phrases here.

### Nonspeech (inserted room-tone pauses, 2,316 hops)

| policy | hops transcribed during room tone | next line shown in pause | wrong du'a in pause |
|---|---:|---:|---:|
| legacy | 16.1% | 38.2% | 0% |
| energy_assisted | 18.4% | 36.6% | 0% |
| ungated | 44.7% | 36.1% | 0% |

### Saved sessions (server-mode recordings replayed through the phone gate)

The 77 s session with no transcripts (`20260926-170753`) sat at a median of −77.8 dBFS:
99% of its hops fall under the −45 dBFS floor under **every** policy. No speech-gate change
would have rescued it. The audio reaching the recognizer was near-silent, which points at
capture or input, not at Silero.

Across the other sessions, the policies differ by 0–16 pts in hops run. The largest
difference is `20260926-184241`: 56% of hops run with legacy against 72% with energy-assisted.

### Latency

- **Smoke run, desktop headless Chrome** (`scripts/gate_bench.mjs`, the page's real worker,
  ONNX q8/WASM, CPU shared with another job): about 2.0–2.5 s per inference, evidence about
  2.3 s old when painted.
- **Quiet-CPU run:** PENDING (the per-policy p95 evidence age goes here once the CPU is free).
- **Phone measurement:** not available. It is required before any default change.

## Verdict against the preregistered rule

The rule (manifest): reduce discarded speech; line exact no more than 1 pt below legacy;
wrong-du'a/nonspeech exposure no more than 0.5 pt above; p95 evidence age no more than
0.2 s worse.

| criterion | energy_assisted | ungated |
|---|---|---|
| discarded speech reduced | **pass** (3.8% → 0.75%) | **pass** (→ 0.10%) |
| line exact within 1 pt | **pass** (−0.1 pt) | **pass** (−0.1 pt) |
| wrong du'a within 0.5 pt | pass on the point estimate (+0.28 pt; interval reaches +0.64) | same |
| nonspeech exposure within 0.5 pt | **fail**: +2.4 pts of room-tone hops transcribed (the display did not get worse: next line in pause 38.2 → 36.6%) | **fail**: +28.6 pts |
| p95 evidence age within 0.2 s | not yet measured on desktop; no phone measurement | same |

**Decision: `legacy` stays the default.**
- `ungated` is ruled out: nearly half of the room-tone hops reach Whisper.
- `energy_assisted` is the candidate. It recovers about 80% of the speech the gate discards,
  at no measurable cost to line following, but by the literal preregistered margin it
  transcribes slightly more room tone.
- It stays **opt-in** (`?gate=energy_assisted`) until:
  1. the desktop latency comparison, then a phone one, shows no p95 regression;
  2. a real phone session in a normal room confirms the nonspeech cost stays harmless.

The −45 dBFS floor is the larger lever for quiet input (−20 dB copies), but these data
don't justify a floor change: the only floor-bound real session was near-silent input.

Where this is weak:
- The room tone is each studio recording's own quietest second, cleaner than a real room.
- Transcripts come from CTranslate2, not the page's ONNX runtime.
- The discarded-speech share rests on automatic word labels.

## Rerun on the full corpus (`rel-20260927`)

`rel-20260926` ran while the shared tree was on the 91-text review branch (see the data-loss
note, 2026-09-27), so its line numbers came from a smaller corpus than the app uses. The
rerun repeats it on main, with all 505 texts and the same preregistration, which was written
before either run. Same data, same code, CPU only.

- **Discarded speech: unchanged** to the rounding (3.8 / 0.75 / 0.10% as recorded;
  7.9 / 1.4 / 0.45% at −10 dB; 23.7 / 11.2 / 10.1% at −20 dB). The gate never sees the corpus.
- **Line following** (22 recordings, 3,173 transitions):

  | policy | line exact | ±1 | wrong du'a | onset lag median / p90 | early > 0.3 s | missed |
  |---|---:|---:|---:|---:|---:|---:|
  | legacy | 81.8% | 96.3% | 0.38% | +0.07 / +1.61 s | 31.9% | 2.6% |
  | energy_assisted | 81.8% | 96.2% | 0.40% | +0.09 / +1.61 s | 30.7% | 2.6% |
  | ungated | 81.9% | 96.2% | 0.42% | +0.10 / +1.61 s | 30.3% | 2.4% |

  With five times as many texts to confuse, the wrong-du'a baseline roughly doubles
  (0.20 → 0.38%) for every policy.
- **Paired against legacy** (bootstrap 95%, by reciter):
  - energy-assisted: line exact +0.05 pt [−0.4, +0.5]; wrong du'a **+0.03 pt [−0.02, +0.10]**,
    down from +0.28 [0.0, +0.64] in the first run;
  - ungated: line exact +0.1 pt [−0.5, +0.8]; wrong du'a +0.27 pt [−0.04, +0.80].
- **Nonspeech** (2,319 pause hops): room-tone hops transcribed 16.2 / 18.5 / 44.7%;
  next line shown in the pause 38.0 / 36.9 / 36.2%; wrong du'a in the pause 0.13% for all three.
- **Hallucination filter:** still identical to the unfiltered rows.

**Verdict: unchanged.** For energy-assisted, the wrong-du'a criterion now passes clearly
(the whole interval sits under +0.5 pt), but nonspeech exposure still fails by the literal
margin (+2.3 pts of room-tone hops), and latency is still unmeasured on desktop and phone.
`legacy` stays the default; `energy_assisted` stays opt-in with the same two conditions.

Artefacts: `data/cache/reliability/rel-20260927/` (`manifest.json`, `gate/results.json`,
`gate_score.txt`).

## Reproduce

```
python scripts/reliability_manifest.py rel-20260926 --check     # inputs unchanged?
python scripts/gate_eval.py prepare
node scripts/gate_replay.mjs data/cache/reliability/rel-20260926/gate/jobs.json data/cache/reliability/rel-20260926/gate/decisions
python scripts/gate_eval.py score        # -> .../gate/results.json
node scripts/gate_bench.mjs <clip.f32> <start s> <seconds> <out.json> legacy energy_assisted ungated
```

The Node dependencies (onnxruntime-node, puppeteer-core) are installed in
`data/cache/reliability/node` (gitignored). The benchmark uses the installed Chrome.
