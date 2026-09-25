16 recordings, test split, ASR = large-v3-turbo, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 43.3% | 72.2% | 5.8% | 87.5% | 12.1% | 3.6 s | 15.5% | 4.32 |
| HMM tracker (identifies du'a) | 81.7% | 96.4% | 82.3% | 96.5% | 0.8% | 1.4 s | 1.2% | 0.04 |
| HMM tracker, du'a given | 83.2% | 99.1% | 82.7% | 100.0% | 0.0% | 1.4 s | 0.4% | 0.15 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 82.2% (47.4%) | — |
| Hadith Kisa | 0.22 | 87.9% (43.8%) | 79.2% (33.3%) |
| Dua Hujjah (Faraj) | 0.02 | 68.7% (38.6%) | — |
| Dua Iftitah | 0.39 | 80.7% (52.8%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 85.1% (49.5%) | 78.6% (0.0%) |
| Dua Mujeer | 0.30 | 66.4% (10.0%) | 72.3% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 88.8% (53.1%) | — |
| Munajat Ta'ibeen | 0.14 | 75.2% (42.0%) | — |
| Dua Nudbah | 0.59 | 85.6% (58.4%) | — |
| Dua Simaat | 0.20 | 78.3% (44.8%) | 70.8% (21.5%) |
| Dua Tawassul | 0.50 | 83.9% (18.8%) | 86.2% (6.2%) |
| Dua Tawba | 0.26 | 81.9% (52.8%) | — |
| Ziyarat Aminallah | 0.13 | 68.6% (44.2%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 69.7% | 84.7% |
| 5 s | 69.4% | 91.6% |
| 10 s | 90.9% | 94.7% |
| 20 s | 94.4% | 98.1% |
| 30 s | 97.2% | 98.4% |
