# docs/demo

The tools that make the README's GIF and the two demo films. On every phone in them is the real
app (`web/`) following a real recording by a voice held out of training.

| | made from | lands in |
|---|---|---|
| README GIF (31 s, no sound): one phone, beside what Whisper and the letter model hear | `inside.html`, `inside-runs/` | `docs/demo.gif`; with sound, `data/cache/media/inside.mp4` |
| Five-phone demo (93 s): five du'as, five reciters | `stage.html`, `runs/` | `data/cache/media/demo_with_audio.mp4` (+ `_small.mp4`, under 10 MB) |
| Story film (70 s): a boy lost in Du'a Kumayl at a majlis takes out the app | `story.html`, `story-art.js`, `story-runs/` | `data/cache/media/story.mp4` (+ `_small.mp4`) |

The films with sound stay in `data/cache/media/`, which git ignores: the recitations are
DuaPlayer's and other people's uploads.

## The .html files are film sets

`film.mjs` loads a stage in headless Chrome, hands it a run and moves its clock one frame at a
time. Opened in a browser by itself, a stage gets neither and stays black. To look at one without
filming all of it:

```
node docs/demo/film.mjs --stills 3,9,20       # PNGs at those film seconds, in data/cache/media/demo/stills/
node docs/demo/film.mjs --fast                # the whole film at 30 fps, no motion blur
```

Both take `--stage inside.html` or `--stage story.html` (with that film's `--runs`, below).

## How a film is made

1. `capture.mjs` plays a stretch of a recording through the page in headless Chrome with the
   engine a phone runs, each step held to the time it takes on Hasan's Galaxy Z Flip 6 (Whisper
   1.73 s a window, so it updates every 2 s; the letter model 60 ms a step;
   `docs/results/phone_engine.md`). It saves every update the app was handed, with the moment in
   the recording it arrived: one run, `runs/N-name.json`.
2. `film.mjs` loads a stage, puts the app on each phone and replays the run into it at 60 fps.
   `vclock.js` holds the page's clock, so no frame is late however slow a screenshot is; moving
   frames are blurred over four samples. Then it mixes the recordings under the picture, each at
   the gain and pan the stage gives it, and encodes.

A film replays its runs, so after the engine changes (a new model, new decoder settings) capture
the runs again, or the film shows the old app.

| file | |
|---|---|
| `capture.mjs` | a recording through the page, to a run; its header has the five phones' capture commands |
| `clip.mjs` | where a clip of a recording is cached (`data/cache/media/demo/clip-*.wav`) |
| `film.mjs` | a stage and its runs, to a film |
| `vclock.js` | the virtual clock: timers, animation frames and CSS animations step with the film |
| `stage.html` | the five-phone demo: the camera, the credits, the intro and outro |
| `inside.html` | the README GIF: the phone, and the recording flowing out of it |
| `story.html`, `story-art.js` | the story: its shots, and the people, hall and cat, drawn each frame |
| `story-sounds.mjs` | the story's own sounds, synthesized: the night outside, a page turning, a tap, the cat (`data/cache/media/story/`) |
| `runs/`, `inside-runs/`, `story-runs/` | the captured runs each film replays |

## Rebuilding

The README GIF:

```
node docs/demo/capture.mjs --seconds 40 --out docs/demo/inside-runs/1-tawassul.json
node docs/demo/film.mjs --stage inside.html --runs docs/demo/inside-runs --fps 30 --samples 1
ffmpeg -y -i data/cache/media/inside.mp4 -vf "fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none" docs/demo.gif
```

The five-phone demo: the five `capture.mjs` commands in its header, then `node docs/demo/film.mjs`
(26 min on this PC).

The story (10 min):

```
node docs/demo/capture.mjs --sid majlis:mj-rSPgtn4Km_w-2226:0 --line 4 --into 2.4 --seconds 84 --credit "Congregation at KSIJ, Dar es Salaam" --out docs/demo/story-runs/1-kumayl.json
node docs/demo/story-sounds.mjs
node docs/demo/film.mjs --stage story.html --runs docs/demo/story-runs --pre 30 --bed data/cache/media/story/night.wav --sfx data/cache/media/story
```

`film.mjs` keeps its masters and soundtrack in `data/cache/media/demo/` (or `story/`, `inside/`),
so two renders of one stage at once spoil each other: give one of them `--work DIR`.

Needs Chrome, ffmpeg on the PATH, puppeteer-core in `data/cache/reliability/node` (as for
`scripts/page_replay.mjs`), and the recordings: the bench's held-out sources for `--sid`
(`data/testbed/sources.jsonl`), and DuaPlayer's for the default Tawassul
(`python scripts/fetch_duaplayer.py`).
