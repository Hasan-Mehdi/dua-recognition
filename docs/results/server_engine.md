# The server engine runs the page's code (2026-10-04)

Hasan, after trying both engines: "the phone model i guess bc we worked on it so much is much much
better than the "cloud" model. Cloud model just stops following at certain points and overall
worse."

## Why it was worse

The two engines were separate programs. The device engine is the page: Whisper and the CTC model
in Web Workers, the tracker, the stream decoder, the highlight and the stop detector in `app.js`.
The server engine (`DUA_ENGINE=server`) streamed the audio to `app/server.py`, which ran its own
pipeline (`pipeline.py`) and sent back positions. What was tuned on the page from 2026-10-03 on
had not reached it:

| | device engine (the page) | server engine until 2026-10-04 |
|---|---|---|
| Whisper | `whisper-base-syn-v5-ctx8ft` (synthetic ordinary voices) | stock `large-v3-turbo` |
| word follower | stream decoder (`stream-follower.js`, since 2026-10-03) | rule-based `follower.py` |
| CTC windows | 2 s | 3 s |
| next-line evidence gate, stop handling, anchoring on the tracker's last result | yes | no |

His server session hce4 (2026-10-04 17:10, Dua Hujjah on a headset) shows the result: stock turbo
heard his voice poorly ("وحنا وحنا وحنا في عدية ساعة"), the tracker dropped the du'a at 12 s, and
the word follower then had no anchor.

## What it is now

One engine, with its two models in one of two places. The page always runs `DeviceEngine`. In the
server engine its two Web Workers are replaced by `RemoteEar`, which takes the same messages and
sends them to `/ws/ear` on the server:

- the page streams its audio as 16 kHz int16 (32 KB/s);
- for a transcript it sends `{"type": "transcribe", "id", "voiceDb", "dt"}`, naming the window by
  the sample it ends at, and the server cuts the window from the audio it already has: the speech
  gate (`legacy` policy, as `gate.js`), the stop measures (`asr.QuietMeter`, which `gate.js`
  mirrors) and Whisper;
- for CTC frames it sends `{"type": "frames", "id"}` and gets a JSON header followed by T x C float16
  log probabilities (100 x 43 for 2 s, 8.6 KB; about 86 KB/s while words are followed);
- `gen` numbers each session, so a reply that arrives after the reader started again is dropped.

The server's models: Whisper `whisper-base-syn-v5-ctx8ft` as CTranslate2 (`DUA_ASR_MODEL`) and the
student `ctc-student-base-v6` on the page's 2 s windows (`DUA_CTC_MODEL`; `window_s` set to 2.0,
as the page's `-w2` export). Whisper runs every second while words are followed (`anchorHop` 1 s;
2 s on the phone, which needs the time for the CTC model). Practice mode works in both engines.
The old `/ws` endpoint, the server's use of `pipeline.py`, and `ServerEngine` in `app.js` are gone;
a change to following is now made once, in the page (and its Python mirror).

## Checks

- **Frames.** For the same 2 s of audio, the frames from `/ws/ear` equal a local run of the student
  at 2 s (largest difference 0.0). Whisper's transcript equals a local `transcribe_batch`.
- **Time per step**, this PC (RTX 5080 / CPU), session n8wi: CTC 13 ms on the server, 25 ms round
  trip; Whisper 35 ms on the server, 48 ms round trip. With both models on the CPU (as in the bench
  run below, for memory): Whisper base int8 54 ms per 6 s window, the student 21 ms per 2 s window
  (29 ms on the GPU, where one small window doesn't fill it).
- `tests/test_server_ear.py`: windows cut by sample number (and after a reconnect), and one
  transcribe + frames round trip through the endpoint.

## Through the real page

The same 20 test items (38 minutes, one per scenario) that the device page played on 2026-10-04 at
phone speed (`data/testbed/page/pg_gate_on2`: Whisper held to 1.2 s and the CTC model to 150 ms a
step, today's defaults), played again through the page served by `app/server.py` with
`DUA_ENGINE=server` (`pg_server`; `scripts/page_replay.mjs --server`, headless Chrome, real time).
In the server run the CTC steps came every 0.1 s (median round trip 28 ms, p90 104 ms) and Whisper
answered in 115 ms (median).

| real page, 20 test items | device, phone speed | server engine |
|---|---:|---:|
| on the reader's line | 86% | **89%** |
| right word | 65% | **78%** |
| jumps /10 min | 2.13 | **1.86** |
| early moves /10 min | 1.33 | 1.33 |
| lost /10 min | 1.86 | 1.86 |
| go-back / skip / jump followed within 3 s | 66% | **69%** |
| du'a found within 10 s | 84% | **89%** |
| du'a found, median | 6.2 s | **5.2 s** |
| wrong du'a shown | 2.5% | **0.5%** |
| line-change lag | +0.51 s | **+0.33 s** |
| a known du'a shown for one it doesn't have (1 item) | **31%** | 40% |

The gains are what lower latency should give: the word and the line change 0.18 s sooner, and the
du'a is named a second sooner. Jumps, early moves and lost moves are a handful of events in 38
minutes. Of the two items where the server run jumped and the device run didn't (fan, skip), each
is one move; on the jumps item the server run made fewer (15.1 against 26.5 /10 min). The one
out-of-corpus item showed a known du'a longer (40% against 31%), one item.

Limits: one item per scenario. The device run is from the morning of 2026-10-04, with the phone's
delay simulated on this PC's CPU; real phones vary, and slow down after ~20 s of sustained load.
The test voices were looked at in the jump work (jumps.md). The old server engine was not played
through the bench (that needs stock turbo on the GPU beside the scorer another session was running,
with the machine's commit charge at 40-43 GB of 42.5); his own sessions and the table above show
how it differed. On a real network each CTC step also waits a round trip: at 100 ms the steps
come every ~0.12 s, still sooner than the phone's 0.15 s.

## Found on the way

The bench's cached CTC frames (`data/testbed/ctc/ctc-student-base-v6_w2_h0.1`) are 150 rows per
window, not 100: `StudentCtc._run` pads windows shorter than the model's `window_s` (3.0 for the
student) with silence before, so the bench's encoder sees 1 s of silence and then the 2 s, while
the page's `-w2` export sees the 2 s alone. The committed frames cover the same audio, but the
encoder context differs. Not changed here (it would invalidate every cached run).

## Next

The server can afford models the phone can't. Both are trained and load through the same
endpoint: `DUA_ASR_MODEL=models/whisper-turbo-dua-ct2` (large-v3-turbo fine-tuned; 84.0% correct
line on the studio set against 85.4% for the phone's base model, and without the synthetic
ordinary voices) and `DUA_CTC_MODEL=models/wav2vec2-quran-dua-voices` (the teacher the student
learned from). Each needs a `bench.py asr` pass and a scored run before it could be the server's
default; the stream decoder and the tracker were tuned on the phone models' outputs.

Code: `app/server.py` (`EarAudio`, `_gate`, `_hear`, `/ws/ear`), `web/app.js` (`RemoteEar`,
`DeviceEngine`), `scripts/page_replay.mjs --server`, `scripts/bench_page.py --server`.
