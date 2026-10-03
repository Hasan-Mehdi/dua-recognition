**Execution plan for Claude Code: reliable browser recognition and acoustic word following**

Work in `C:\Users\hasan\code\speech-ml\dua-recognition`.

Execute this plan through implementation, development-set experiments, and a reproducible report. The objective is to determine and improve what actually limits live du'a following: discarded speech, acoustic emission latency, inaccurate reference timings, or display behavior. Preserve successful existing behavior. A documented negative result is an acceptable experiment outcome; unsupported accuracy claims are not.

Read [the research review](current-state-review-2026-09-26.md), `STATUS.md`, and the latest `docs/results/follower.md` first. This plan supplements the two sessions already working on follower round 2 and amateur/majlis test data. Reuse their completed work; do not reimplement those plans.

**Current evidence and an update to the review**

- Round 1: follower exact words 65.2% versus page mode 38.0%; next line during inserted pauses 1.6% versus 32.9%; median human-line-entry lag +0.94 s versus +0.30 s. Word references are automatic and share the follower's acoustic model.
- Browser `web/asr-worker.js` requires `speechInTail()` before transcription, independently of `quietAtEnd()`'s energy fallback. Server `Recognizer` defaults to `vad=False`. Therefore current offline/server metrics do not establish browser behavior.
- The latest round-2 **train diagnostics**, added after the initial review, show that oracle position anchoring does not reduce the lag, while additional right context does. The follower reportedly moves when the first relevant letter appears. Treat this as evidence against more broad tracker tuning; inspect the finished report before designing another follower change.
- Right-context diagnostic improvements are not live improvements unless the time spent waiting for that context is included.
- The reported ~0.30 s difference between human line marks and forced-aligned first words does not prove that humans mark breaths: the automatic word alignment could itself be late. Human audit must distinguish those explanations.

**Execution constraints**

- Work sequentially. Do not launch additional agents, broad hyperparameter searches, new bulk harvests, or training sweeps.
- Inventory the dirty tree and active jobs before editing. Do not reset, clean, stash, or overwrite another session's changes. If a needed file is still actively edited, build a separate helper/report first and integrate after that work settles. Recheck its diff immediately before integration.
- Do not stop or restart Hasan's phone server or existing harvest/evaluation jobs. A running GPU server counts as GPU work. Check actual processes, GPU use, and Windows commit headroom before compute; run heavy CPU evaluations sequentially and at most one GPU workload at a time. Continue lightweight implementation when compute is unavailable.
- Use `.venv/Scripts/python.exe`. Inspect the current environment before installing dependencies. Keep optional research dependencies isolated. Large assets/caches belong under the existing D: data/cache arrangement; use `D:/hf-cache` where appropriate. Do not change paging files or reboot.
- Use unique run names under `data/cache/reliability/`; do not overwrite the other sessions' logs or caches. Long jobs must survive the interactive session and write resumable progress. Launch Windows helpers hidden.
- Keep private audio, session details, source manifests, and reviewed labels in gitignored storage. Reports contain aggregate results. Do not upload private recordings to external services.
- Do not commit or push as part of this plan. Deliver reviewable diffs and results. Do not change default model selection simply because an experimental implementation loads.
- Human review and physical-phone measurements cannot be fabricated. Prepare review bundles and exact instructions if those inputs are unavailable; continue every independent phase and finish with an explicitly provisional verdict where needed.

**Step 0 — Establish a reproducible starting point**

1. Read `git status`, recent commits, running-job logs, the current follower report, and the actual test-set implementation. `STATUS.md` may lag behind ongoing work.
2. Record the current checkpoint/export identifiers, corpus hash, relevant source-file hashes, parameters, dependency versions, and platform. A commit ID alone is insufficient with this dirty tree.
3. Save a small inventory in `data/cache/reliability/<run>/manifest.json`: recordings already available, split/label provenance, current defaults, and which files are actively owned by the other work.
4. Reuse the current replay/session/evaluation helpers where possible. New scripts should be thin adapters, not a second implementation of the tracker or scoring rules.
5. Write the development experiment choices and acceptance rules below to the run manifest before looking at new outcome tables. Keep the original round-2 preregistration unchanged.

Deliverable: a frozen configuration and input manifest, plus a list of completed work being reused. No new look at the final test is needed for this step.

**Step 1 — Fix measurement and label provenance**

Likely touchpoints: `src/dua_recognition/{corpus,labels,splits}.py`, `scripts/{user_label,word_truth,word_eval,follow_eval,evaluate}.py`, and the new test-set helpers. Inspect the actual interfaces; names proposed below are not assumed to exist.

1. Separate line-label provenance from word-label provenance. Preserve the existing `auto`/tier interface for compatibility, but add explicit metadata for each granularity: source kind, model/revision if automatic, reference hash, review state, and reviewed intervals. Missing provenance means unknown, not human-reviewed.
2. Importing a reviewed line SRT promotes **line** truth only. Newly force-aligned words remain automatic. A partially reviewed recording must expose masks/intervals so unreviewed words cannot enter a human-reviewed score.
3. Preserve actual substitutions, omissions, repetitions, and uncertain timings. Separate the canonical position being followed from a verbatim transcript of what was spoken. Give repeated occurrences distinguishable identities where needed.
4. Add a review-bundle exporter and importer using existing subtitle/annotation tools. Include audio excerpts, candidate word starts/ends, canonical text, source provenance, and an editable annotation file. Prepare at least 60 development excerpts: 10 each for ordinary within-line speech, first words after pauses, long vowels, repeated/skipped words, low-level/room audio, and low-confidence/rejected alignment. If material is unavailable, report the shortfall; do not replace missing real cases with claims based on synthetic examples.
5. Keep this deliberately difficult diagnostic sample separate from a randomly sampled audit used to estimate label quality. Record uncertainty or acceptable onset intervals when a precise boundary cannot be heard.
6. Make the new amateur/majlis manifests distinguish development from final test before evaluating candidates. Split by person and venue, grouping source recordings/reuploads together. Retain existing test designations. Treat the repeatedly examined old test as a historical regression suite, not fresh independent evidence.
7. Report supplied hours, successfully labelled hours, scored hours/words/transitions, and exclusions by reason. Low teacher agreement should route audio to review, not make the failures disappear from the dataset inventory. Projected slide timings remain proposals until checked against the audio.
8. Audit cache keys before reuse. Include corpus/reference hashes, acoustic model/export, preprocessing, gate configuration, tracker/follower configuration, label revision, and assumed latency. Reject incompatible caches. In particular, inspect the current follower cache key before relying on it across config changes.

Meaningful tests: corrected line labels cannot silently promote word labels; partial review masks are respected; legacy loaders retain existing behavior; split groups do not cross development/final test; cache invalidation catches a changed configuration/reference.

Deliverables: provenance-aware evaluation, a local review bundle, a frozen split manifest, and coverage reporting. Human-reviewed metrics remain unavailable until a real reviewer completes the relevant annotations.

**Step 2 — Measure and correct browser speech gating**

Likely touchpoints: `web/asr-worker.js`, `web/app.js`, `web/session-log.js`, `scripts/session_report.py`, and `src/dua_recognition/asr.py`. Keep experimental controls in developer/debug configuration, not the main product flow.

1. Instrument each hop with gate policy, inference ran/skipped, exact skip reason, input level, VAD evidence, energy evidence, trailing-quiet estimate, captured-audio end, enqueue/dequeue times, inference duration, and visible-update time. Distinguish no sound, rejected speech, empty ASR result, filtered hallucination, and an inference failure. Preserve existing session-file compatibility.
2. Use the audio sample clock for capture positions and a documented mapping to the browser monotonic clock for display latency. Do not subtract unsynchronized server/browser wall clocks. Separate cold-start time from steady-state timing.
3. Add three explicitly selected policies while leaving the initial default unchanged:

   | Policy | Behavior |
   |---|---|
   | `legacy` | Exact current browser gate, retained as the baseline |
   | `energy_assisted` | A bounded energy/speech-activity fallback can permit transcription when Silero rejects melodic speech |
   | `ungated` | Bypass the extra Silero gate while retaining the separately documented silence floor and existing output safeguards |

4. Treat the -45 dBFS floor as a separate factor. First compare the three gates with it fixed. Then, only if low-level failures warrant it, compare the leading policy with a conservative near-zero-input check or gain-aware floor. Freeze that rule on development data. Do not describe -45 dBFS as literal digital silence.
5. Inspect browser hallucination handling before claiming parity with Python. Avoid changing filtering and gating simultaneously in the first comparison. If a safety filter is missing, give that change a separate ablation.
6. Build a replay harness around the actual browser worker and display path, reusing the existing browser test setup if available. Use the same phone checkpoint and audio for all browser policies. Compare CTranslate2 separately to identify runtime differences; do not treat a turbo server versus base browser comparison as an isolated gate test.
7. First use a fixed smoke subset to validate the harness, then run each policy once over the selected development set. Include natural speech, sustained vowels, pauses, repeats, noise-only stretches, and lower-gain copies of known speech. Keep transformed variants grouped with their source recording. Reuse existing pause data and out-of-corpus recordings.
8. Measure discarded annotated-speech seconds, hallucinated/wrong-du'a exposure during nonspeech, exact/within-one line, onset-lag distribution, missed transitions, update cadence, and p50/p90/p95 audio-to-visible-output latency. Separate human-labelled results from automatic-label diagnostics.

Predeclare the selection rule: choose among policies that reduce false-silence exposure without a material regression in line accuracy or wrong-du'a/nonspeech behavior. For this experiment, use a 1 percentage-point line-accuracy margin and a 0.5 percentage-point wrong-du'a exposure margin; require p95 evidence age to be no more than 0.2 s worse. These are engineering tolerances, not statistical guarantees. Report paired per-recording results and grouped uncertainty; if the data cannot resolve the tradeoff, retain the default and keep the candidate opt-in.

Meaningful tests: recent acoustic activity plus a Silero rejection reaches the intended policy branch; true quiet remains correctly reported; skipped inference is distinct from an empty transcript; old session files still load. Test the browser path rather than duplicating its decision logic in Python.

Deliverable: `docs/results/browser_gate.md` plus machine-readable results, reproducible run instructions, and a candidate/default decision supported by evidence.

**Step 3 — Finish interpreting follower round 2 before extending it**

1. Read the completed round-2 report/logs and preserve its original verdict. Do not rerun its grid or spend another test look to choose a parameter.
2. Verify the latency accounting. Every online decision must use only samples available at that moment. Extra right context, chunk accumulation, queueing, inference, and render delay all count. Keep oracle/future-context diagnostics in a clearly separate table.
3. Compare lag against the human-audited onset sample as well as legacy line marks and automatic word timings. Report signed median, p90 positive lag, >0.3 s early entries, and missed transitions. Show coverage so a method cannot improve lag by failing to enter difficult lines.
4. If the new evidence confirms that emissions arrive late and the follower acts promptly, leave the decision rule alone and proceed to Step 4. The default next experiment is a faster/more suitable acoustic model, not another onset heuristic.
5. Only if decision/display latency remains after that accounting, build one opt-in comparison that keeps the next line previewed while committing the active highlight using fresh local text evidence. Reuse the existing preview UI. Compare continuation/repetition of the current line with the next line's allowed opening; activity alone must not identify the next word.
6. Explicitly test the new `_onset()` rule against sustained final vowels, repetition after a breath, a cough/noise burst, and crowd response. Folded CTC blank absorbs more than acoustic silence; never use it as a calibrated silence probability. Preserve recovery from real backward repeats and forward skips.
7. Keep preview availability and active-highlight correctness as separate metrics. Do not improve the “line entered” score by counting preview as commitment.

For a new follower candidate, predeclare development gates: preserve at least a +5 percentage-point exact-word gain over page mode where valid labels exist; next-line exposure during pauses at most 10%; jerks and early-entry rate no worse than page mode; median human-onset lag no more than 0.2 s worse than page mode, with no increase in missed transitions. Compare against the round-2 incumbent as well. Automatic-word scores remain secondary if the independent audit is incomplete. Keep a candidate experimental if any required evidence is missing.

Deliverable: `docs/results/follower_reliability.md`, explaining which latency component is actionable, whether a new decision rule was justified, and what passed or failed. Do not alter the earlier round-2 success criteria after its results are known.

**Step 4 — Run one bounded small-CTC comparison**

Proceed after the measurement harness is working. Do not start by training a new model.

1. Inspect [Tilawa](https://github.com/yazinsai/tilawa), particularly `packages/core/src/recitation/{engine,tracker,ctcDecoder}.ts`, and its [experiment notes](https://github.com/yazinsai/tilawa/blob/main/lab/EXPERIMENTS.md). Pin the inspected revision. Identify whether a reported number is live tracking or includes a final/fallback identification pass.
2. Compare the existing wav2vec2 follower with **one** practical Arabic text-CTC candidate, using a documented public ONNX export where possible. Tilawa's FastConformer option is the first artifact to inspect. Verify its vocabulary, quantization, input normalization, frame stride, and streaming contract instead of assuming they match wav2vec2.
3. Create a narrow posterior adapter with explicit blank ID, token vocabulary, frame timestamps, valid lengths, and any state/cache. The current single-character folding code must not silently discard BPE pieces. If necessary, align native token sequences and map them back to canonical words. Check Arabic normalization and word ownership with meaningful examples.
4. Distinguish buffered-window inference from actual cached streaming. For streaming, carry encoder caches and CTC decode state across chunks; validate repeated letters at chunk boundaries. Include lookahead and end flushing in latency. Do not hardcode wav2vec2's 20 ms stride for another model.
5. Validate a few development clips for correctness before the full comparison. Record download size, memory, warm/cold latency, real-time factor, word/line metrics, pauses, and acquisition/recovery. Use the same tracker anchors, labels, and time accounting as the incumbent.
6. Only pursue [Quran-Lab Zipformer](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3) if the text-CTC experiment leaves a clear unmet need and its published access/usage terms fit the task. It uses a Quranic phoneme vocabulary, not ordinary Arabic character IDs. First demonstrate a valid reference mapping for a few du'as; otherwise document the missing adapter and stop that branch. Do not bypass gated access or silently accept account terms.
7. Use actual phone measurements before claiming phone suitability. If no phone is available, provide a local browser benchmark route and label desktop results as provisional. Do not port the whole architecture merely because inference succeeds.

Deliverable: `docs/results/small_ctc.md` with a go/no-go recommendation. No checkpoint becomes the default unless its quality gates and device latency both pass. If neither candidate helps, retain wav2vec2 for server experiments and document the failure.

**Step 5 — Close the work with a measured next action**

1. Run `.venv/Scripts/python.exe -m pytest -q` and the relevant browser checks once the owned changes are stable. Compare failures with the starting state; do not modify another session's unfinished code merely to make the full suite green. Repeat checks only when changes or failures warrant it.
2. Reproduce one completed run from its manifest, verifying compatible caches and deterministic scoring. Report nondeterminism if inference varies; do not select the best run.
3. If a final comparison is justified and untouched labelled data are available, freeze the chosen candidate first and evaluate that candidate versus the incumbent once. Do not use those results to tune another variant. Otherwise clearly separate development evidence from final validation.
4. Write `docs/results/reliability_summary.md`: starting behavior, implemented changes, result tables, label coverage/provenance, device measurements, negative results, and exact commands. Update the live status note after reconciling the other sessions' work. Keep README headline claims unchanged until the relevant validation supports them.
5. Prepare a balanced next-training manifest only if the measured failures point to acoustic generalization. Cap contributions by speaker and du'a, preserve held-out venues/voices, and report unique speakers/hours rather than overlapping-window counts. Do not launch that training run in this plan.
6. Final response: link the reports; state the chosen behavior and why; list checks run; identify any human review/phone measurements still needed; distinguish completed implementation from provisional experimental conclusions. Leave all changes uncommitted and nothing pushed.

**Research references for specific decisions**

- [Less-peaky CTC through label priors](https://arxiv.org/abs/2406.02560): alignment-aware training, relevant if boundary quality remains the bottleneck.
- [Delay-penalized CTC](https://www.isca-archive.org/interspeech_2023/yao23b_interspeech.html): training-side emission latency, not a justification for arbitrary display lead.
- [Delayed-KD](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.pdf): a later student-training option if streaming acoustic delay is confirmed.
- [Stateful Conformer with cached inference](https://arxiv.org/abs/2312.17279): distinguish true streaming state from repeated windows.
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner): an optional second source of offline timing proposals; its outputs remain automatic labels.
- [Adaptive holdout reuse](https://arxiv.org/abs/1506.02629): why a new run on an already examined test set is not fresh validation.

All changes above are intended to be concrete, reversible, and reviewable. Continue independent work when a compute slot, external asset, human annotation, or physical device is unavailable. Do not replace missing evidence with assumed passes, and do not stop after producing another plan.
