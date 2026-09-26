16 recordings, test split, ASR = whisper-small-dua-v2, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 47.1% | 72.4% | 6.4% | 86.8% | 12.9% | 3.3 s | 14.1% | 4.09 |
| HMM tracker (identifies du'a) | 84.4% | 96.3% | 88.4% | 96.3% | 0.3% | 1.3 s | 1.1% | 0.02 |
| HMM tracker, du'a given | 85.5% | 98.0% | 88.3% | 98.5% | 0.0% | 1.3 s | 0.2% | 0.09 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 83.8% (49.4%) | — |
| Hadith Kisa | 0.22 | 86.4% (48.2%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 67.5% (47.0%) | — |
| Dua Iftitah | 0.39 | 86.2% (60.1%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 86.2% (55.2%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 74.2% (11.9%) | 82.7% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 91.2% (54.8%) | — |
| Munajat Ta'ibeen | 0.14 | 79.4% (49.9%) | — |
| Dua Nudbah | 0.59 | 88.0% (63.9%) | — |
| Dua Simaat | 0.20 | 81.8% (49.7%) | 80.0% (29.2%) |
| Dua Tawassul | 0.50 | 86.9% (19.0%) | 90.6% (6.4%) |
| Dua Tawba | 0.26 | 83.7% (56.7%) | — |
| Ziyarat Aminallah | 0.13 | 68.8% (45.4%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 80.6% | 75.9% |
| 5 s | 78.4% | 87.2% |
| 10 s | 90.9% | 92.2% |
| 20 s | 95.9% | 94.4% |
| 30 s | 97.2% | 96.9% |
