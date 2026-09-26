16 recordings, test split, ASR = large-v3-turbo, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 42.7% | 71.4% | 5.8% | 86.7% | 12.9% | 3.6 s | 15.8% | 4.09 |
| HMM tracker (identifies du'a) | 81.5% | 96.1% | 82.6% | 96.2% | 0.4% | 1.4 s | 1.2% | 0.04 |
| HMM tracker, du'a given | 82.6% | 97.8% | 82.7% | 98.3% | 0.0% | 1.4 s | 0.2% | 0.09 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 81.6% (46.7%) | — |
| Hadith Kisa | 0.22 | 87.2% (43.8%) | 79.2% (33.3%) |
| Dua Hujjah (Faraj) | 0.02 | 67.5% (38.6%) | — |
| Dua Iftitah | 0.39 | 80.7% (51.4%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 85.2% (49.5%) | 78.6% (0.0%) |
| Dua Mujeer | 0.30 | 66.4% (9.9%) | 72.3% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 88.8% (53.0%) | — |
| Munajat Ta'ibeen | 0.14 | 74.1% (41.8%) | — |
| Dua Nudbah | 0.59 | 85.5% (57.6%) | — |
| Dua Simaat | 0.20 | 78.3% (44.8%) | 70.8% (21.5%) |
| Dua Tawassul | 0.50 | 84.2% (18.8%) | 86.8% (6.2%) |
| Dua Tawba | 0.26 | 81.8% (52.7%) | — |
| Ziyarat Aminallah | 0.13 | 67.5% (40.0%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 65.6% | 62.2% |
| 5 s | 67.8% | 81.2% |
| 10 s | 88.4% | 90.6% |
| 20 s | 92.8% | 95.6% |
| 30 s | 94.4% | 96.6% |
