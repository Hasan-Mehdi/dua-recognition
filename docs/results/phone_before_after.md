# The phone app before and after 2026-09-30

Three changes, each tested on its own (the pages linked below), together:

1. **Display rules for ordinary voices** ([display_stability.md](display_stability.md)):
   - a 0.25 s display lead instead of 0.5 s;
   - a tapped line stays put;
   - a step back needs two updates in a row;
   - once locked, "not in the corpus" needs worse windows to take over.
2. **An 8 s audio context** ([phone_speed.md](phone_speed.md)): four times less work per
   update, so the phone can update every second instead of every 2-3 s.
3. **Synthetic ordinary voices in training** ([synthetic_voices.md](synthetic_voices.md)):
   RetaSy CER 30.0% → 24.1%.

"Before" is the previous phone model (whisper-base-aug-v4) at the cadence the phone kept
(an update every 3 s, shown 2.5 s late), with the previous display rules. "After" is
whisper-base-syn-v5-ctx8ft at one update a second, 1 s late, with the new rules. Every
set is scored as `evaluate.py` scores DuaPlayer: the line on screen once per update.

| set | | line | ±1 | refrain | du'a right | wrong du'a | lines missed | jumps/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DuaPlayer test, 16 recordings, 4 h, human labels | before | 79.4% | 96.3% | 82.6% | 96.4% | 0.2% | 8.3% | 2.39 |
| | **after** | **85.1%** | 96.4% | **88.9%** | 96.4% | 0.4% | — | — |
| Majlis dev, 5 streams, 1.7 h, automatic labels | before | 78.1% | 96.7% | 64.3% | 97.4% | 0.0% | 10.1% | 1.87 |
| | **after** | **84.3%** | **97.0%** | **77.2%** | 97.1% | 0.1% | **1.4%** | **0.14** |
| Hasan's own recordings, 2, 2.4 min, automatic labels | before | 27.9% | 48.8% | — | 65.1% | 2.3% | 57.9% | 2.79 |
| | after | 45.8% | 54.2% | — | 80.2% | 3.8% | 47.4% | 2.29 |

A dash means the number wasn't measured.

On the majlis streams, the new model alone at the old cadence and rules scores about the
same as the old one (78.3% line). There, the gain comes from updating every second and
from the display rules.

On Hasan's six debug sessions, replayed and scored against the server model:

| | line | page frozen | steps back | flickers | taps undone |
|---|---:|---:|---:|---:|---:|
| before, as he saw it | 67.1% | 13.3% | 9 | 7 | 7 of 12 |
| after: new rules, new model (at the sessions' own cadence) | 75.0% | 3.7% | 7 | 2 | 2 of 12 |

Caveats:
- The DuaPlayer test set has been looked at once per change tonight, three times in all.
  Read these numbers as a regression check, not a fresh estimate. The majlis numbers are
  fresh, but on 2 voices and automatic labels; Hasan's recordings are minutes long.
- The phone timings assumed in "after" (about 1 s per update) come from desktop Chrome
  (0.58 s). The first session on Hasan's Android will confirm or correct them.

## Reproduce

```
python scripts/transcribe_windows.py --source majlis --model models/whisper-base-syn-v5-ctx8ft-ct2 --tag whisper-base-syn-v5-ctx8ft
python scripts/evaluate.py --quick --source majlis --part dev --asr whisper-base-syn-v5-ctx8ft --latency 1.0
python scripts/evaluate.py --quick --source majlis --part dev --asr whisper-base-aug-v4 --stride 3 --latency 2.5 \
    --set display_lead=0.5 seek_pins_line=false back_confirm=1 null_rate_locked=null
```

`--part dev|final` (new) scores one part of a held-out set's frozen split (`testset_split.py`).
The final part of the majlis set is untouched.
