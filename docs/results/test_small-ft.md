16 recordings, test split, ASR = small-ft, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 46.4% | 71.2% | 6.6% | 87.2% | 12.5% | 3.2 s | 14.7% | 5.55 |
| HMM tracker (identifies du'a) | 85.7% | 96.4% | 88.8% | 96.4% | 1.1% | 1.2 s | 1.2% | 0.05 |
| HMM tracker, du'a given | 87.3% | 98.9% | 89.2% | 100.0% | 0.0% | 1.2 s | 0.3% | 0.17 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 84.3% (48.3%) | — |
| Hadith Kisa | 0.22 | 89.2% (47.3%) | 91.7% (33.3%) |
| Dua Hujjah (Faraj) | 0.02 | 69.9% (43.4%) | — |
| Dua Iftitah | 0.39 | 87.3% (62.0%) | 100.0% (25.0%) |
| Dua Kumayl | 0.42 | 87.8% (54.5%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 78.1% (12.0%) | 82.7% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 91.2% (50.0%) | — |
| Munajat Ta'ibeen | 0.14 | 80.4% (48.3%) | — |
| Dua Nudbah | 0.59 | 88.2% (62.1%) | — |
| Dua Simaat | 0.20 | 83.0% (48.2%) | 76.9% (30.8%) |
| Dua Tawassul | 0.50 | 88.0% (19.2%) | 91.3% (6.7%) |
| Dua Tawba | 0.26 | 85.1% (55.4%) | — |
| Ziyarat Aminallah | 0.13 | 70.6% (48.9%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 82.8% | 87.5% |
| 5 s | 82.2% | 92.5% |
| 10 s | 93.1% | 95.0% |
| 20 s | 96.6% | 97.8% |
| 30 s | 98.8% | 98.1% |
