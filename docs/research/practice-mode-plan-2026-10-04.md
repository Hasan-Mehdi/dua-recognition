# Practice mode: a phased plan

Drafted 2026-10-04 by a planning agent (Fable) from [tarteel-2026-10-04.md](tarteel-2026-10-04.md)
and the code. The names it uses were checked against the tree that day. Two corrections are
applied: the repeat gap rule lives in `sc_repeat` in `bench.py`, not in a `repeat_points`
function; and `phone_latency.md`'s "word exact" (59.3% at 0.3 s) measures display timing, not
how well the model hears. Nothing here is decided. Hasan's decisions are listed at the end.

**Status, 2026-10-04:** Phases 0 and 1 are done ([practice_bench.md](../results/practice_bench.md)).
The display alone failed the bars. The v1 checker (`src/dua_recognition/practice.py`,
`web/practice.js`) judges each line by a with/without-line alignment and passes them on dev and
on test: 0.12 false left-out /10 min, 93% skipline, 97% ending, 1.7 / 3.8 s. It differs from
the plan below in several ways:
- It decides when the display has stayed two lines past the line, not at line close.
- It scores lines only (no per-word "different").
- It pins the display with no far jumps.
- It adds a sound gate.
Phase 2 is on the page too ("practise" on the home screen, `?practice=<id>&from=<n>`; rosettes carry
the verdicts; hide-text; summary and per-du'a memory). Next: Hasan's own sessions in practice mode.

## Where the codebase already stands

- **The follower is the head start.** `StreamFollower` (`src/dua_recognition/stream_follower.py`,
  mirrored in `web/stream-follower.js`) holds a probability over every letter of the du'a. `L`
  and `B` are per letter, `F` is the talk filler per line, and `IL`/`IB` is the salawat chain.
  It already explains:
  - pauses (blank)
  - restarts (`c_restart`)
  - going back 1-3 lines (`c_back`)
  - skips (`c_skip`)
  - talk (`fill_cost`, `c_fill_in`)
  - the salawat (`interjections`)

  `posterior()` gives `pw` (probability per word) and `pf` (the share off the text). The
  next-line gate (`_next_margin`) makes a line crossing an evidence-backed event. The detector
  should observe all this, not rebuild it.
- **The bench is the measuring instrument.** `scripts/bench.py`:
  - builds items as programs over source audio (`Prog.play/tone/clip`, the `["other", ...]` op)
  - renders them (`render`) and caches model outputs (`cmd_asr`)
  - replays the page (`replay` → `stream_updates`)
  - scores (`metrics`, `aggregate`, `BARS`) and checks pre-registered bars (`cmd_guard`)

  Its 24 scenarios already include every non-error the research lists: `pause`, `talk`,
  `salawat`, `repeat`, `back`, `stumble` (self-correction), `jumps`.
- **Word timings exist** for source lines in `sources.jsonl`, as `lines[].words = [[global word,
  start, end], ...]`. DuaPlayer timings come from forced alignment (`scripts/word_truth.py`),
  harvest timings from the CTC teacher. A planted skipped word is cut with these.
- **Synthetic voices exist.** `scripts/synth_voices.py` clones a voice with OmniVoice
  (`create_voice_clone_prompt`), and `check()` round-trips the result through turbo-ft. The same
  machinery can read an *edited* line in the source's own voice.
- **The page already has these pieces:**
  - per-word spans on the current line (`setWords`, `.wd`, `.said`)
  - `paintWords`, which marks every word before the token as `.said`. This is Tarteel's
    "painted as correct" problem and must change in practice mode.
  - "Hide other lines" (`body.focus`)
  - line taps meaning "I'm here" (`seekTo`)
  - `stored()` over localStorage, with `saveProgress`
  - the `today`/`hijri()` chips
  - kids mode (`kids.js`), the model for a skin that leaves the engine untouched
  - the session log (`session-log.js`, `scripts/session_report.py`)
- **The hearing limit.** The phone CTC student's character error rate on everyday voices is
  24.1% (README). It mishears ط as ت and drops ع (`phone_latency.md`). A single near-word swap
  (رحيم/حليم) is likely out of reach on phones. Measure that before promising it.

## Phase 0: the error bench, before any UI

### New scenarios in `scripts/bench.py`

They run on lanes `studio`, `harvest`, `mafatih` and `user`. Ordinary voices are the point;
`majlis` is crowd audio and not relevant to solo practice.

| scenario | how it's planted | truth |
|---|---|---|
| `skipword` | Every ~4th line, one word is cut out: two `["src", a, b]` spans around the word's `[start, end]` with a ~15 ms crossfade (new op form). The word has ≥3 letters, isn't first or last in its line, doesn't recur within ±1 line, and has clean gaps to its neighbours, like `sc_repeat`'s gap check. | `errors: [{kind:"skipword", dua, word, t}]`; the word is left out of `words` |
| `swapword` / `nearword` | Every ~4th line is replaced by a TTS clip of the same line with one word changed, in the source's own voice (new op `["tts", clip]`, clips from `scripts/synth_errors.py`). `nearword` changes one letter, using a word from the corpus vocabulary. `swapword` uses another word of the same du'a. | `errors: [{kind, dua, word, said, t_a, t_b}]`; `words` carries the TTS words at the round-trip alignment's times |
| `skipline` | `_runs_with` with `between` returning `i + 2` after a breath tone, every 3-5 lines | `errors: [{kind:"skipline", dua, seg}]` |
| `foreignline` | After a breath, one line of another du'a that shares ≥3 words with the line just read (found via `tracker._passage_keys`; audio via the `["other", ...]` op), then the reading continues | `errors: [{kind:"foreignline", said_dua, said_seg, t_a, t_b}]` |
| `ending` | The run stops 1-3 lines before the end, followed by `tone(8.0)` | `errors: [{kind:"ending", first_unread_seg}]` |
| `swapfix` | A `swapword` TTS line, a breath, then the correct line: a self-corrected slip | no error; must stay unflagged |
| `ttsctrl` | The unedited line resynthesised and spliced in | no error; checks that the splice itself isn't read as talk |

The existing cells give non-errors for free: `flow`, `pause`, `talk`, `salawat`, `repeat`,
`back`, `stumble`, `slow`, `fast`, plus the recording conditions. `skip` counts as a skipped line
the detector should catch. `jumps` is left out of flag scoring.

Items get a new field `errors`, filled by a `Prog.error(kind, **fields)` helper. Its times are
scaled by `tempo`, like `events`.

### `scripts/synth_errors.py` (TTS venv)

Modelled on `synth_voices.py`. For each planted line:

1. Clone the voice from the source's own cleanest line (2-8 s).
2. Generate the edited line.
3. Run `check()` twice. The transcript must align to the edited text at CER ≤ 0.3, and the
   changed word must be heard as changed: it is closer to the planted word than to the original.

Clips that fail are dropped. Word times come from CTC forced alignment, as in `word_truth.py`.
GPU order: TTS first, then `bench.py asr`. One CUDA job at a time.

### Scoring: `metrics_flags`

Add a `PracticeConfig` (`--pc k=v`) and run the detector as an **observer** after each `sf.step`
in `stream_updates`. It collects `flags: [(t, dua, seg, word_lo, word_hi, state)]`, where state
is `different`, `unsure` or `unheard`. Metrics:

- **False flags per 10 min of correct reading.** `different`/`unheard` on words in the truth
  that aren't within ±1 word of a planted cut. `unsure` is reported as its own rate.
- **Catch rate per error type.** A mark must land on the planted word (±1 word) or line before
  the item ends.
- **Time to decide.** The flag time minus the end of the error's line, as median and p90.
- **Flips.** A word that goes from `different`/`unheard` back to `heard` without the line being
  read again. Bar: 0.
- **Display unchanged.** The observer doesn't change display ticks; assert this in a test, so
  every existing `cmd_guard` bar passes by construction.

`cmd_guard` gets a `--prereg practice` group.

### v0 detector: no new model code

Score a trivial rule from the follower's trajectory already in the replay:

- A word is `heard` if the display rested on it, or `pw ≥ show_p`, while its line was shown.
- A word is `unheard` if the reader moved two lines past it without that.
- Nothing is ever `different`.

This gives the first false-flag rate and skipped-line catch rate on real ordinary voices in an
afternoon. It also shows how much a real detector has to add.

### Proposed bars (Hasan sets them; write them to `data/cache/practice/prereg.md` before the run)

- **False `different`:** ≤ 0.3 per 10 min pooled; ≤ 0.5 on `harvest`+`user`; ≤ 0.2 over
  pause, talk, salawat, repeat, back, stumble, swapfix and ttsctrl.
- **False `unheard` (line):** ≤ 0.2 per 10 min.
- **`unsure`:** ≤ 3 per 10 min pooled, reported per lane.
- **Catch rates that gate shipping:** skipline ≥ 85%, ending ≥ 85%, foreignline ≥ 70%.
  skipword, swapword and nearword are reported but don't gate v1.
- **Time to decide:** median ≤ 2.5 s after the line ends, p90 ≤ 5 s.
- **Flips:** 0.
- **Guards:** every existing bar unchanged; `follow_ms` median ≤ +1 ms.
- **Shipping rule:** word-level `different` ships only if it meets its bar on `harvest`+`user`.
  Otherwise ship `heard`/`unsure` for words and `unheard` for lines.

### Files, tests, size, risks

- **Change:** `scripts/bench.py`.
- **Add:** `scripts/synth_errors.py`, `data/cache/practice/prereg.md`,
  `docs/results/practice_bench.md`, and `tests/test_bench_errors.py`. The tests build each
  scenario from a fake source and check `words`/`errors` consistency, and check
  `metrics_flags` on hand-made flags.
- **Risks:**
  - Automatic word times may clip neighbouring words. Spot-check 20 cuts by ear.
  - The TTS splice may read as talk; `ttsctrl` measures this.
  - Harvest truth is "confident lines only", so report `harvest` and `studio` separately.
- **Size:** 3-5 days including GPU passes and the write-up.

## Phase 1: the mistake detector

### Evidence already there

- The stream decoder: `pw`, `pf`, committed frames, line entry/exit, the filler `F`, the salawat
  chain, `quiet_now`.
- The tracker: `segment_confidence`, `line_masses()`, `null`. Use these as a second vote for
  `foreignline` and `ending`.
- Whisper text via `CorpusIndex.word_costs`. At 24% CER it can only confirm, never flag alone.

### Where the decision lives

New `src/dua_recognition/practice.py` (`PracticeConfig`, `PracticeChecker`), an observer that
never writes to the follower. Decisions are made once:

- **At line close**, when the follower has committed the move to another line. Line k's words
  are judged on the frames buffered from entry to exit.
  - A restart of the line resets the buffer, so the last attempt counts and a self-correction is
    not an error.
  - A line re-entered later (back or repeat) is judged again. That is the reader correcting, not
    the app retracting.
- **At session end** (stop, or `lapse_s`), lines after the furthest one read are `unheard`.
  Lines skipped earlier become `unheard` once the reader is two lines past them.
- **Never** at a pause or breath, while `pf > 0.5`, or while the salawat holds the probability.

Per word, a small CTC DP compares "said" against two alternatives: deleted (blank) and replaced
(the filler). Its shape follows `ctc_align.end_scores` / `_end_scores_loop`; `endScores` in
`web/follower.js` is the JS start.

- `heard`: the log-likelihood ratio is ≥ `theta_heard`.
- `different`: the ratio is ≤ `−theta_diff`, the replaced path beats deleted, and the word had
  ≥ `min_frames` of letter frames.
- `unsure`: anything else.

Thresholds are tuned on dev to the false-flag bar. A foreign line shows as `pf` high for a
line's length while no line of the du'a gains probability. v1 marks it as a gap between lines.

Open question: extend `endScores`, or write a separate `wordScores` in both languages? The
second is cleaner for parity.

### Mirroring and parity

- **`web/practice.js`** is hooked in `app.js` after the device engine's `follower.step`. It takes
  `?pc=k:v` via `urlCfg`, and logs `flag` and `line_close` events that `session_report.py
  --timeline` prints.
- **`tests/test_practice_parity.py`**, modelled on `test_stream_parity.py`, uses synthetic frames
  with a deletion, a substitution, a restart, a salawat and a pause. Python and JS must match
  step for step.
- **`tests/test_practice.py`** covers the rules: the last attempt counts, no decision while
  `pf > 0.5`, the ending, no flips.
- **No `TrackerConfig` change**, so `bench._LATE_FIELDS` is untouched.

### Risks and size

- `different` may not meet the precision bar on phones. Then v1 ships heard/unsure/unheard and
  says so in the UI.
- Long lines are costly on slow phones. Bound the buffer to ~20 s, use `squeeze_blanks`, and
  measure `follow_ms`.
- **Size:** 4-7 days.

## Phase 2: the practice UI

A mode like kids mode (body class `practice`), with the engine unchanged.

- **Three word states, never "painted correct".** In practice mode `paintWords` stops marking
  `.said` by position. Words get `.heard`, `.unsure` or `.diff` at line close, and stay neutral
  until then. No animation, sound or vibration. Flags also paint the transliteration.
- **Hidden mode.** Extend `body.focus`. A word is revealed when `pw ≥ show_p`, except:
  - never while `pf > 0.5`
  - never into line k+1 until 2 of its words are confirmed or `_next_margin` has passed
  - never on a word from the common-word list alone. The list is built by `export_web.py` into
    `corpus.json`: الله, اللهم, يا, رب, و, على, plus each du'a's refrain lines.

  Starting mid-du'a reveals nothing before the start line.
- **Graded hints.** First letter, then word, then line. Each is logged as a hint, not a mistake.
- **Quiet feedback.** Marks appear at line close (0.3 s fade). The session-end summary includes
  "Checked: words and lines. Not vowels or tajweed."
- **Dismiss.** Tapping a mark offers "That was right". Dismissals are logged, and become a live
  false-flag estimate in `session_report.py`.
- **Fading per-line marks.** `web/marks.js` keeps `stored("practice:<duaId>")`. Each clean
  recitation halves a line's weight. Weak lines show faintly at session start, and the picker
  shows "N weak lines".
- **Hear a line.** A speaker glyph in practice mode only, because a tap already means "I'm here".
  The DuaPlayer recordings carry `slide_start_ms` per segment (read in `corpus.py`), and the
  server serves them at `/audio/{audio_id}`. That is the server engine only: the static page has
  no recordings ("not ours to redistribute").
- **Files.** Change `web/app.js`, `web/index.html`, `web/style.css`, `scripts/session_report.py`
  and `scripts/export_web.py`. Add `web/marks.js`, the practice UI, and a "Practice mode"
  section in `docs/how-it-works.md`.
- **Tests.** Fading arithmetic in Node. A `page_replay.mjs` run with `?practice=1` whose logged
  flags must match Python.
- **Risks.** Reveals may feel slow on refrain du'as. The mark popover must not fight the
  "I'm here" tap.
- **Size:** 4-6 days.

## Phase 3: daily use

- **Plans** (`web/plans.js`). Examples: "Du'a Ahd, day 12 of 40" (`dua-aahad`), the weekday du'as,
  Ziyarat Ashura, Sabah, and Kumayl over four weeks by line ranges. A plan starts with an
  intention and is committed by press-and-hold. No badges.
- **Reminders.** Extend `today`/`hijri()` with weekday texts, after Fajr and Thursday night.
  Browsers can't schedule notifications cleanly; iOS needs the page installed, and
  `coi-serviceworker.js` already registers a service worker. So v1 shows reminders when the app
  opens, plus an `.ics` calendar export.
- **"Continue from this line" quiz.** Hidden mode plus `tracker.seek(dua, seg)` (which pins the
  line) on a random line. Weak lines are chosen more often.
- **Silent variant** (`web/quiz.js`). Pick the next words from distractors in the same du'a, or
  type them, normalised like `text.normalize`.
- **Counting a phrase said aloud.** Run the stream decoder on the phrase as N identical segments:
  the forward move beats `c_restart`, and `c_skip` stops it racing ahead. Measure first on
  `mafatih-dua-asharat`, whose repeated phrases are each a line. Bar: count within ±1 on ≥ 95%
  of items.
- **Size:** 5-10 days over several sessions; the counter and the plan card are about a day each.

## Decisions only Hasan can make

1. **The bars.** What ceiling for false `different` on ordinary voices? Does word-level
   `different` ship in v1 at all?
2. **Which lanes gate:** `harvest`+`user` only, or `studio` too?
3. **When flags appear:** at the end of each line, or only in a session summary?
4. **Hearing a line:** run the practice page from his own server, ask DuaPlayer, record
   reference readings himself, or leave it out of the static page?
5. **Testing on his own voice:** synthetic voices and cut audio avoid deliberate misrecitation
   on the bench, but a phone test with his own voice needs a stance.
6. **The common-word and refrain list for hidden mode:** by corpus frequency, or by hand?
7. **Reminders:** calendar export for v1, or a service worker now?
8. **Recordings:** the wording of the opt-in donation, and whether practice sessions upload.

## Fastest safe path to something on the phone

1. **Days 1-2.** Phase 0 v0: the `skipline`, `ending`, `skipword` and `swapfix` scenarios (no
   TTS yet), `asr`, and the trajectory detector. This produces real numbers and a `prereg.md`.
2. **Days 3-5.** Phase 1 at line level only (`heard`/`unsure`/`unheard`, no `different`), in
   Python + JS with a parity test, behind `?practice=1`.
3. **Days 6-7.** The Phase 2 minimum: the toggle, three word states, hidden mode with the
   reveal rule, hints, dismiss, and the local marks. That is usable for Du'a Ahd, and his
   sessions become the first real-learner data.
4. **Then:** the TTS swap items, and the word-level `different` decision, judged on the bench.
