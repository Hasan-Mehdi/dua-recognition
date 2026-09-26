16 recordings, test split, ASR = whisper-base-quran-dua, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 47.0% | 72.1% | 6.3% | 86.7% | 13.0% | 3.2 s | 14.0% | 4.54 |
| HMM tracker (identifies du'a) | 85.2% | 96.2% | 89.1% | 96.3% | 0.3% | 1.2 s | 1.2% | 0.04 |
| HMM tracker, du'a given | 86.6% | 98.0% | 89.2% | 98.6% | 0.0% | 1.2 s | 0.2% | 0.10 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 84.5% (48.9%) | — |
| Hadith Kisa | 0.22 | 88.2% (47.1%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 66.3% (47.0%) | — |
| Dua Iftitah | 0.39 | 87.5% (60.3%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 87.4% (55.1%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 75.5% (13.0%) | 84.6% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 91.5% (54.9%) | — |
| Munajat Ta'ibeen | 0.14 | 80.6% (51.1%) | — |
| Dua Nudbah | 0.59 | 87.8% (63.6%) | — |
| Dua Simaat | 0.20 | 82.7% (49.3%) | 84.6% (30.8%) |
| Dua Tawassul | 0.50 | 87.3% (18.1%) | 90.7% (6.3%) |
| Dua Tawba | 0.26 | 85.3% (57.2%) | — |
| Ziyarat Aminallah | 0.13 | 69.5% (45.4%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 80.0% | 74.7% |
| 5 s | 79.7% | 87.2% |
| 10 s | 90.9% | 91.9% |
| 20 s | 96.2% | 95.0% |
| 30 s | 97.8% | 96.2% |
