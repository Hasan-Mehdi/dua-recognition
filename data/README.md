# data

Reference texts live in `duas/<dua_id>.json`. Each file is a du'a split into
ordered segments; the matcher compares ASR output against these.

```
duas/<dua_id>.json     committed   reference text, one file per du'a
raw/<dua_id>/*.wav      ignored     downloaded recitations (see scripts/fetch_audio.py)
processed/<dua_id>/      ignored     resampled / segmented clips
manifest.csv            ignored     source url, reciter, license per raw file
```

`hadith_e_kisa.json` is a **minimal illustrative excerpt** so the pipeline has
something to load — only the opening few segments, and not fully diacritized.
Replace it with a verified full text before doing anything real; sources worth
trusting are al-islam.org and duas.org. Keep the segment ids contiguous.

Audio is never committed. Track every download in `manifest.csv` with its source
URL and license so the corpus stays auditable.
