16 recordings, test split, ASR = whisper-base-aug1, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 45.3% | 69.6% | 6.0% | 85.3% | 14.5% | 3.3 s | 15.0% | 5.45 |
| HMM tracker (identifies du'a) | 85.0% | 96.1% | 88.5% | 96.1% | 0.3% | 1.2 s | 1.2% | 0.06 |
| HMM tracker, du'a given | 86.2% | 97.7% | 88.4% | 98.3% | 0.0% | 1.2 s | 0.2% | 0.12 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 83.7% (48.1%) | — |
| Hadith Kisa | 0.22 | 88.3% (47.1%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 65.1% (37.3%) | — |
| Dua Iftitah | 0.39 | 87.1% (58.3%) | 100.0% (0.0%) |
| Dua Kumayl | 0.42 | 86.6% (54.5%) | 92.9% (7.1%) |
| Dua Mujeer | 0.30 | 76.5% (11.8%) | 85.1% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 90.6% (50.4%) | — |
| Munajat Ta'ibeen | 0.14 | 79.8% (47.9%) | — |
| Dua Nudbah | 0.59 | 88.5% (61.1%) | — |
| Dua Simaat | 0.20 | 82.9% (48.9%) | 83.1% (32.3%) |
| Dua Tawassul | 0.50 | 86.5% (16.1%) | 89.8% (5.7%) |
| Dua Tawba | 0.26 | 85.5% (56.3%) | — |
| Ziyarat Aminallah | 0.13 | 69.5% (42.0%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 76.6% | 69.7% |
| 5 s | 77.8% | 85.6% |
| 10 s | 91.9% | 91.6% |
| 20 s | 95.0% | 94.7% |
| 30 s | 96.6% | 95.3% |
