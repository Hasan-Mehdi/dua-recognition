16 recordings, test split, ASR = whisper-base-aug-v4, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 46.3% | 70.9% | 6.5% | 86.3% | 13.5% | 3.2 s | 14.5% | 4.92 |
| HMM tracker (identifies du'a) | 84.8% | 96.2% | 87.8% | 96.2% | 0.5% | 1.3 s | 1.2% | 0.04 |
| HMM tracker, du'a given | 86.1% | 97.9% | 88.2% | 98.4% | 0.0% | 1.2 s | 0.2% | 0.09 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 84.1% (47.8%) | — |
| Hadith Kisa | 0.22 | 87.8% (47.8%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 63.9% (39.8%) | — |
| Dua Iftitah | 0.39 | 87.1% (59.1%) | 87.5% (12.5%) |
| Dua Kumayl | 0.42 | 87.0% (55.3%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 76.6% (12.8%) | 83.8% (0.0%) |
| Munajat Imam Ali (a) | 0.26 | 90.7% (53.0%) | — |
| Munajat Ta'ibeen | 0.14 | 78.2% (49.3%) | — |
| Dua Nudbah | 0.59 | 88.3% (62.4%) | — |
| Dua Simaat | 0.20 | 82.1% (48.5%) | 78.5% (36.9%) |
| Dua Tawassul | 0.50 | 86.3% (18.1%) | 89.5% (6.2%) |
| Dua Tawba | 0.26 | 84.3% (56.3%) | — |
| Ziyarat Aminallah | 0.13 | 69.5% (43.6%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 80.6% | 72.5% |
| 5 s | 78.8% | 87.5% |
| 10 s | 90.9% | 92.2% |
| 20 s | 96.2% | 95.0% |
| 30 s | 96.6% | 95.6% |
