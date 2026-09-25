# Speed prior, tempo adaptation, and scoring text with CTC (2026-09-25)

## Where the errors were

Breaking the tracker's wrong steps down (whisper-base-quran-dua, test reciters,
old flat speed prior):

| outcome | share of steps |
|---|---|
| right line | 81.0% |
| **one line ahead** | **11.9%** |
| one line behind | 3.5% |
| nothing shown yet | 2.4% |
| wrong du'a | 1.0% |

Being *behind* is just display lag: 99% of those steps fall in the first 3 s of a
new line. Being *ahead* is the real problem, and it grows with how long the reciter
holds a line: the error rate is 5% two seconds into a line and 35% after eight.

The cause was the speed prior. It spread each hop's belief flat over 0-2.5 words
ahead, an average of ~1.25-1.5 words/s, while the human timings say reciters
average 0.87 words/s per line (0.14-1.19 per recording). During a long,
melodic line the belief drifted into the next line faster than one window's
evidence could pull it back: being one word further on costs only
exp(-kappa x letters) = ~0.55 in likelihood.

## The fix

- **Speed prior from data.** P(moved d words in dt) is a Poisson mixture over the
  time-weighted deciles of per-line speed on the train reciters, scaled by a
  factor chosen by grid search on the train split (x1.8; the transcript of a
  6 s window lags the voice, so the prior has to run a little ahead).
- **Tempo adaptation.** Each of the 10 speeds keeps a weight, multiplied after
  every update by the evidence its own prediction received (an interacting
  multiple-model filter, forgetting factor 0.98). A slow reciter's belief stops
  running ahead; a fast one's keeps up.

Chosen on the train split (stock large-v3-turbo transcripts, the only model not
trained on those reciters), then reported on test:

| | train line | train refrain | test line (turbo) | test line (base-ft) | test refrain (base-ft) | jumps/min (base-ft) | lag (base-ft) |
|---|---|---|---|---|---|---|---|
| flat prior (before) | 79.9% | 70.5% | 80.4% | 81.0% | 86.8% | 0.22 | 0.7 s |
| data speeds, no scaling | 75.4% | 64.4% | 79.8% | 85.4% | 89.5% | 0.05 | 1.2 s |
| x1.8 speeds | 80.2% | 70.0% | 82.0% | 85.6% | 88.1% | 0.11 | 1.0 s |
| **x1.8 speeds + tempo (new default)** | **80.2%** | **75.1%** | **81.6%** | **85.4%** | **89.1%** | **0.05** | 1.2 s |

The cost is 0.5 s more lag switching to a new line. In exchange there are 4x
fewer jumps, and the phone model (base-ft) gains 4.4 points. `web/tracker.js`
mirrors the change; on 300 cached windows its outputs match `tracker.py` step
for step.

## Negative result: scoring the text directly with CTC

The idea: instead of transcribing each window and aligning the 1-best string,
score how well the text ending at every word explains the window's CTC
posteriors (`ctc_align.py`: a free-start CTC Viterbi over the whole corpus,
~28 ms per window after collapsing blank frames). This keeps what the model was
unsure about.

With the fine-tuned wav2vec2 (models/wav2vec2-quran-dua, val CER 23.9%), test
reciters, new tracker:

| observation | line | refrain | wrong du'a | found @3 s |
|---|---|---|---|---|
| greedy CTC transcript -> edit distance | 82.5% | 83.7% | 0.7% | 84% |
| CTC posteriors scored directly (kappa 0.2, best of 4 *picked on test*, so optimistic) | 82.0% | 84.2% | 1.0% | 92% |

The two are the same for following. Direct scoring finds the du'a faster from a
cold start (+8 points at 3 s), but it isn't the step change it would be for a
weaker or less peaky model. Both are below whisper-base-ft (85.4%), which stays
the phone model. On the other hand, the wav2vec2 model needs no autoregressive
decoder and runs a window in ~30 ms on a desktop CPU.
