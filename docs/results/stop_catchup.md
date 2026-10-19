# Catching up at a stop (2026-10-08/09)

Hasan's session `20261008-020516-aq31` (Ziyarat Waritha from line 18, desktop Chrome, headset mic):
"it stopped following at some point even tho i was saying the next lines; it ultimately fixed
itself but some mistakes after as well". Two separate causes. The mistakes after were our text:
line 32 lacked the second "bi-abī anta wa ummī" (fixed with the text audit, `text_audit.md`). The
stall is this write-up.

## The stall

Line 31 is "wa ʿalā ghāʾibikum, wa ʿalā ẓāhirikum, wa ʿalā bāṭinikum". The highlight reached the
second "wa ʿalā" at 152.8 s and held it until 165.7 s:

| time | Hasan | page |
|---|---|---|
| 153.1-157.0 | ẓāhirikum, wa ʿalā, bāṭinikum | on the second "wa ʿalā" |
| 157-162 | stop | on the second "wa ʿalā" |
| 162.1-167.3 | the same three words again | caught up at 165.7-166.1 |

(Word times from the fine-tuned turbo with word timestamps.) Every model hears his "wa ʿalā" here
as "wa-anā" and the ẓ as z; turbo wrote "وأنا زاحركم وأنا باتنكم", and the streaming letter model
heard least: greedy "ازححك … انابعتن". A Python replay of the session (the page's q8 stream model
through `stream_ctc.PageStream`, the tracker over the logged transcripts, `StreamFollower` at the
logged step times; 9 of 1,847 steps differ from the log) shows the decoder's probabilities through
the stop: 0.63 on the second "wa ʿalā", 0.32 on "bāṭinikum", the line's last word. The display
shows the likeliest word, so it held; with the stop detector hearing no voice, letters cost 4 nats
more and nothing changed for 5 s.

## The rule

`StreamConfig.stop_p` (JS `stopP`; 0 = off), `stop_s`, `stop_last`: once the stop detector has heard
no voice for `stop_s` (0.5 s), a word ahead of the shown one in its line that holds `stop_p` of
the probability is shown, and the display doesn't step back from it until the voice returns. With
`stop_last`, only the line's last word. `stop_tracker` (the tracker's word, ahead in the line by at
most `stop_tracker_words`, as follower.py's quiet catch-up) was tried too. Python and JavaScript
step for step: `tests/test_stream_parity.py`, four new cases that each change the display.

## On the bench

Phone cadence (Whisper every 2 s, 1.8 s late; the stream every 0.1 s, 0.1 s late), `--display
stream --same-text 8`, against a fresh baseline at the defaults of the day (`stop_base2_dev`, equal to
`full_ph_G1_dev` to the tick). Two new per-item counts in `bench.metrics`: reading time with the
highlight on the reader's line but ahead of or behind their word, and, at stops (1 s or more before
the next word, from 0.5 s in, up to 5 s), whether the highlight is on the word the reader stopped
after, ahead of it or behind it.

| dev, 1,373 items | right word | at stops: on the word | ahead | behind | reading ticks: right / ahead |
|---|---:|---:|---:|---:|---:|
| defaults | 74.33% | 85.52% | 1.35% | 2.50% | |
| any word ahead, 0.25 | 74.37% | 85.55% | 1.36% | 2.46% | +568 / +48 |
| any word ahead, 0.2 | 74.39% | 85.58% | 1.37% | 2.42% | +896 / +81 |
| any word ahead, 0.15 | 74.42% | 85.66% | 1.38% | 2.33% | +1,499 / +137 |
| any word ahead, 0.1 | 74.44% | 85.71% | 1.40% | 2.25% | +1,938 / +275 |
| 0.2 + the tracker's word | 74.38% | 85.58% | 1.42% | 2.36% | +809 / +495 |
| the tracker's word alone | 74.33% | 85.53% | 1.41% | 2.42% | -59 / +419 |
| **line's last word, 0.15** | 74.38% | **85.67%** | **1.36%** | **2.33%** | +872 / +37 |
| line's last word, 0.1 | 74.40% | 85.76% | 1.37% | 2.24% | +1,130 / +101 |

The tracker's word at a stop is too often ahead of the reader: not taken. Any word ahead, 0.15,
failed one guard bar on test: majlis "stays put" 90.0% -> 89.0%, all from one item
(`pause-acb4cc13f3`): the reader stopped after "qadīm", the end of their line; the display showed
another text reading the same passage, where "qadīm" is followed by "wa" in the same line, and a
stray letter gave the one-letter "wa" 15%: the catch-up showed it, which in the reader's text is
the next line. Readers stop at line ends; `stop_last` keeps the gain at stops and drops most of
the wrong moves (ahead +37 ticks against +137).

`bench.py guard`, line's last word 0.15 against the defaults: every bar passes on dev but "user lane
on line >= 78% (absolute)", 76.6% before and after (it failed the same way on 2026-10-07). On
lines, jumps, early moves, stays put and line lag nothing changes: the rule moves within a line.

Test, once (617 items; the test voices were looked at in the jump work):

| test | right word | at stops: on the word | ahead | behind |
|---|---:|---:|---:|---:|
| defaults (`stop_base_test`) | 75.73% | 87.09% | 1.17% | 2.22% |
| line's last word, 0.15 (`stop_l15_test`) | 75.76% | 87.25% | 1.17% | 2.06% |

Guard on test: every bar passes. The gain is small because the stall is rare on the bench: at
today's defaults the highlight is behind the reader's word for 2.5% of stop time. Hasan's own
sessions (user lane, dev): behind at stops 6.2% -> 5.8%, on the word 74.6% -> 75.0%.

## On the real page

The session through the real page (`page_replay.mjs`, three plays of the same audio): the
letter model's frames differ from play to play (the page takes the audio in real time), and so does
the probability on "bāṭinikum" at the stop: 0.32 in Hasan's session, 0.22 and 0.06 in two replays.
At 0.15 the catch-up comes at 157.5 s in the first two and not in the third. This stop sits at the
edge of what the letter model hears in his voice; the rule fixes it most of the time, the letter
model would fix it every time.

One test item per scenario (24) through the real page at phone speed (`bench_page.py --per-scenario 1
--split test --asr-ms 1200 --ctc-ms 150`), the defaults of the day against `stopP:0.15,stopLast:true`
(`pg_stop_base`, `pg_stop_l15`; on the texts as corrected on 2026-10-09): on line 90% and right word
67% both; jumps 2.35 -> 2.72, early 0.72 -> 0.91 /10 min, stays put 94% -> 88%, unknown du'a shown
31% -> 38%. None of that is the rule: each of the 24 sessions replayed in Python with the rule as
played and off gives the same display at every one of 19,930 steps (the replay matches the page's
log at 99.2% of steps). The rule didn't come into play in these 24 items (on the Python bench it
touched 34 of 617 test items), and two plays of the same items differ this much by themselves:
the page hears its audio in real time, and its models' frames differ from play to play.

On since 2026-10-09: `stop_p` 0.15, `stop_last` on (py and js). The server engine's profile keeps it
off (`stopP: 0` in app/server.py's SERVER_PROFILE): not scored with the server's models. `?sc=stopP:0`
turns it off on the page.

## Where things are

Files: `src/dua_recognition/stream_follower.py` and `web/stream-follower.js` (`_catch_up`/`_catchUp`,
the display's move split out as `_move`), `tests/test_stream_parity.py`, `scripts/bench.py` (the new
counts). Bench results `stop_*` in data/testbed/results, logs in data/testbed/logs/stop, bars in
`data/cache/stop/prereg.md` (written before the first result).
