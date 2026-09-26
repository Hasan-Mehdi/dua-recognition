**Current-state review: dua-recognition — 26 September 2026**

Recommendation: continue the acoustic word follower, but first close the browser/evaluation gap and establish independent timing labels. Then benchmark a small streaming CTC model. These offer a clearer path to reliable phone use than another broad model sweep or more professional-reciter training audio.

This is a research and code review, not a new performance evaluation. Sources inspected: recent local Claude conversations from September 25–26, STATUS.md, result reports, the working tree, seven saved session logs, primary papers, and upstream projects. Snapshot: 13:47 America/New_York; HEAD was `28de645` with concurrent uncommitted changes. Two Claude sessions were actively implementing follower round 2 and the amateur/majlis test sets. Their unfinished changes are observations, not validated results. No models were loaded, jobs interrupted, or application code changed for this review.

**What the existing evidence establishes**

The [round-1 follower report](../results/follower.md) gives this comparison on professional test recordings:

| Metric | Page mode | CTC follower |
|---|---:|---:|
| Exact word, against automatic word labels | 38.0% | 65.2% |
| Next line shown during inserted pauses | 32.9% | 1.6% |
| Jerks/minute, flowing recordings | 0.75 | 0.31 |
| Median line entry after human line start | +0.30 s | +0.94 s |
| Line entries more than 0.3 s early | 27.4% | 1.1% |

The follower is promising, but these are not ordinary-user accuracy numbers. Its word labels come from the same fine-tuned wav2vec2 model used by the follower. The stock-model guard helps, but does not create independent word truth. Increasing simulated follower compute delay from 0.1 to 0.3 s reduced exact-word accuracy to 53.6%, so deployment latency matters materially.

The latest seven saved sessions were all server sessions, totaling about six minutes, with only one correction tap. That tap indicated the displayed segment was one behind. This is useful diagnostic evidence, not enough to score live accuracy. One 77.5 s microphone session produced no transcripts; its aggregate recorded level was about -59.8 dBFS. It deserves inspection for input/gating problems, but its level alone cannot establish whether intelligible speech was present.

**1. First resolve the browser's inconsistent speech gate**

In [web/asr-worker.js](../../web/asr-worker.js), `quietAtEnd()` combines Silero with energy evidence, but the transcription branch independently requires `speechInTail()`. Therefore `quiet` can indicate recent sound while the worker still skips Whisper and returns empty text. The energy fallback added to improve pause handling does not rescue that transcription.

Meanwhile [pipeline.py](../../src/dua_recognition/pipeline.py) defaults to `vad=False`, explicitly because gating hurt noisy-room results. [transcribe_windows.py](../../scripts/transcribe_windows.py) also generates transcripts without that extra gate and records speech presence separately. Python transcript/tracker results therefore do not fully measure the deployed browser path. ONNX and CTranslate2 may also differ; sharing a checkpoint does not establish end-to-end equivalence.

Both paths have an absolute -45 dBFS silence threshold. That is an amplitude policy, not a test for literal digital silence; quiet input can fall below it. Simply opening the gate on every sound would also admit fans and PA noise, so this needs a controlled comparison.

Next experiment: replay identical train/development audio through the actual browser and server, comparing current gating, an energy-assisted rule, and no extra VAD gate while retaining hallucination handling. Include low input levels, elongated vowels, real pauses, and room noise. Log `skip_reason`, audio level, VAD result, energy result, inference time, and audio-end-to-display age. Score false silence and hallucinations alongside line/word quality. Use real phone timing before promoting a change.

**2. Strengthen the new test-set plan before treating its results as ground truth**

The active plan already addresses the right domains: ordinary voices and majlis audio, with separate silver and gold tiers. Add the following distinctions:

- **Human-reviewed line labels and human-reviewed word labels are separate.** Importing a corrected SRT can establish line truth, but rerunning wav2vec2 afterward still gives automatic word boundaries. Store label provenance at each granularity. Review a stratified sample of actual word onsets, especially first words, long vowels, repeats, and stops; mark uncertain boundaries explicitly.
- **Preserve difficult recordings.** `follow_eval.py` filters word truth by alignment score. The new labelling plan also sends teacher disagreements to review. Publish retained coverage and failure counts, and sample rejected material for human review. Otherwise the recognizer helps select the easiest parts of its own test.
- **Two Whisper teachers are correlated.** Agreement between stock large-v3 and fine-tuned turbo is a useful review signal, not proof of correctness. Canonical-text snapping can erase a user's actual substitutions and omissions. Keep “where in the du'a?” labels distinct from “what was spoken?” transcripts.
- **Projected slide changes are not automatically spoken-line starts.** They may be anticipatory or late. Use OCR/slide changes to propose boundaries and reduce review effort, then audit their relation to the audio.
- **Make development and final test partitions now.** Keep speakers, recordings, reuploads, and majlis venues disjoint. The old test has informed repeated architectural decisions; one new run on those same recordings is not a fresh independent test. Use it for regression, and reserve new untouched people/venues for the final comparison. Adaptive holdout reuse can itself cause overfitting: [Dwork et al., 2015](https://arxiv.org/abs/1506.02629).
- Report per-person/per-venue results and uncertainty over those groups. Thousands of correlated 0.1 s ticks are not thousands of independent observations. Include missed transitions, late-entry tails, false advances, and recovery after repeats/skips, not just median lag.

The [Quran-Lab benchmark](https://huggingface.co/datasets/Quran-Lab/quranic-asr-benchmark) has 200 phone recordings among its 600 clips and restricts use to research/evaluation. It is useful supplementary ASR evidence; it is not a substitute for du'a tracking tests. [Iqra'Eval](https://aclanthology.org/2025.arabicnlp-sharedtasks.61/) targets pronunciation assessment, so its deliberate-error material also needs appropriate reference/scoring choices.

**3. Keep global identification and local acoustic following, but separate preview from commitment**

The current architecture already has the right division: Whisper/HMM identifies and re-anchors; CTC follows nearby words. The unresolved question is when a new line has enough acoustic evidence to become active.

The in-progress `_onset()` in [follower.py](../../src/dua_recognition/follower.py) can advance from a line's final word after a blank run followed by any nonblank letter. This deserves explicit adversarial cases: continuing the last word, repeating it after a breath, coughing, another person responding, and starting a different line. In [ctc.py](../../src/dua_recognition/ctc.py), the folded blank column also absorbs diacritics, delimiters, and special tokens. It is even less suitable as a literal silence probability.

CTC blanks can occur during speech. The [less-peaky CTC paper](https://arxiv.org/abs/2406.02560) explains why good transcription can coexist with poor boundaries and studies label-prior training to improve alignment. This supports investigating the emission representation; it does not establish that inference-time blank scaling will fix this app.

My proposed next design, if round 2 still misses its gate:

1. Keep the next line visible as a preview while the current word remains active.
2. Commit when fresh acoustic evidence favors the next line's allowed opening over continued/repeated current-line speech. Let acoustic activity support that decision; do not treat activity alone as word identity.
3. Retain competing local positions when the evidence is ambiguous. Permit re-anchoring for genuine repeats/skips, rather than imposing global monotonic movement.
4. Measure preview availability separately from active-highlight correctness. This avoids making the reader wait for text while reducing pressure to highlight unheard words.

This is a hypothesis to compare with round 2, not a claimed fix. Real-time music/lyrics work supports combining complementary evidence streams, but its results do not transfer directly to du'a: [Brazier and Widmer, 2021](https://aclanthology.org/2021.nlp4musa-1.1/).

First attribute the lag to human-label offset, acoustic emission, 0.2 s scheduling, inference, and display delivery. Tune only the component responsible. [Delay-penalized CTC](https://www.isca-archive.org/interspeech_2023/yao23b_interspeech.html) targets emission delay through training; [Delayed-KD](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.pdf) addresses streaming teacher/student timing mismatch. They are later training experiments if the measured bottleneck is emission latency. Neither guarantees lower du'a latency without an accuracy cost.

**4. The most relevant projects to borrow from**

| Project | Concrete use here | Limitation |
|---|---|---|
| [Tilawa](https://github.com/yazinsai/tilawa) | Browser TypeScript/ONNX implementation with streaming recognition and word-progress events; inspect its search/tracking separation and token handling. README lists a 66 MB Zipformer asset and an 88 MB text-CTC FastConformer alternative. | Quran-specific corpus and symbols; not a replacement for the du'a index. Code and model licenses differ. |
| [k2 / icefall](https://github.com/k2-fsa/icefall) | Recipes and infrastructure for CTC/Zipformer training and later timing-loss experiments. | A training toolchain, not a ready-made du'a follower; avoid adding it before a small model comparison justifies the cost. |
| [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | Offline alternative aligner with Arabic examples and word/character output, useful for finding disagreements in the review queue. | Automatic labels remain silver until reviewed; this does not itself provide a live follower. |
| [ctc-segmentation](https://github.com/lumaku/ctc-segmentation) | A comparison for placing known text in long recordings and obtaining alignment confidence. | Requires a suitable CTC acoustic model; extraction confidence must not silently define test coverage. |

Tilawa's [experiment notes](https://github.com/yazinsai/tilawa/blob/main/lab/EXPERIMENTS.md) make an important distinction: some excellent short-clip identification results include a fallback after the live engine failed to lock. They also describe deferred verse emission until fresh next-verse evidence. Borrow the latter idea; do not compare their headline identification percentages with our continuous word-tracking scores.

The underlying [Quran-Lab Zipformer](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3) is a 65.5M-parameter streaming phoneme model with ONNX exports. Its card documents weaknesses on children and pronunciation deviations, and its model terms are NPL-1.2. Its output uses 251 symbols and blank ID 250, so the existing Arabic-character `fold_matrix` is not a compatible adapter. The reference phoneme inventory must be extended/validated for du'as. For a cheaper first integration, compare a text-CTC candidate before committing to that phoneme work.

Do not assume every FastConformer checkpoint is stateful streaming. The [cache-aware Conformer paper](https://arxiv.org/abs/2312.17279) describes explicit context limits and activation caching. The current wav2vec2 path instead reruns a 3 s window every 0.2 s: substantial repeated computation. A real streaming implementation must carry model state and decode state across chunks, use the model's actual frame stride, and account for right context and tail flushing. Measure full audio-to-visible-output latency, not only ONNX call duration.

**5. Concrete experiment order**

| Order | Experiment | Decision it should settle |
|---|---|---|
| 1 | Browser gating comparison on existing development recordings, with explicit skip reasons and real-phone latency | Is speech being discarded before recognition? |
| 2 | Human audit of first-word onsets, pauses, repeats, and failed alignments; freeze new development/final-test groups | Are the measured gains and late-entry targets faithful to users? |
| 3 | Finish the active round-2 experiment, then compare evidence-based commitment with separate preview if needed | Can we retain pause stability while reducing late line activation? |
| 4 | Small CTC comparison on the same development audio: existing wav2vec2, one text-CTC candidate, then the phoneme candidate if warranted | Is there a phone-feasible follower with acceptable timing and coverage? |
| 5 | A balanced next training set, chosen from measured failures | Do ordinary voices improve without sacrificing professional/majlis performance? |

For round 2, retain its preregistered criteria; do not loosen them after looking at test. Add actual device latency and human-audited boundaries before making it the default. For a later architecture comparison, set gates in advance for false advancement during real pauses, early/late transition distributions, wrong-du'a exposure, and recovery from genuine repetitions, as well as exact words.

For the next training run, cap contributions by speaker and du'a and mix ordinary voices with professional and room recordings. More overlapping windows from the same long performance do not create more independent speakers. Use model disagreement, poor alignment coverage, and user correction taps to choose material for review. Keep a randomly sampled evaluation portion as well, so a deliberately hard diagnostic set is not mistaken for a population estimate.

Defer further large ASR sweeps, more HMM parameter grids, and expanding noha mode until these decisions are resolved. The local results already show limited gains from several model substitutions and identification tuning. Shared openings also impose real ambiguity: retain multiple candidates until discriminating words arrive, and report acquisition from both cold start and the first distinctive phrase. Faster commitment to an indistinguishable opening is not automatically an improvement.

**Scope and verification**

This review verified source branches, existing result tables, current chat plans, and saved-session metadata. It did not independently reproduce the reported model metrics or benchmark the external projects. New proposals are explicitly experiments. The only file created by this review is this document; no application edits, model downloads, training, commits, or pushes were performed.
