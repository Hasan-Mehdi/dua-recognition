16 recordings, test split, ASR = whisper-base-syn-v5-ctx8ft, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 47.0% | 71.4% | 6.5% | 86.8% | 12.9% | 3.2 s | 14.4% | 4.87 |
| HMM tracker (identifies du'a) | 85.4% | 96.3% | 88.6% | 96.3% | 0.5% | 1.2 s | 1.2% | 0.02 |
| HMM tracker, du'a given | 86.7% | 98.1% | 88.6% | 98.7% | 0.0% | 1.2 s | 0.2% | 0.08 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 83.6% (48.8%) | — |
| Hadith Kisa | 0.22 | 88.8% (49.0%) | 91.7% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 65.1% (39.8%) | — |
| Dua Iftitah | 0.39 | 87.0% (59.7%) | 87.5% (12.5%) |
| Dua Kumayl | 0.42 | 87.6% (55.5%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 77.0% (12.4%) | 84.0% (0.0%) |
| Munajat Imam Ali (a) | 0.26 | 91.1% (54.4%) | — |
| Munajat Ta'ibeen | 0.14 | 81.4% (49.7%) | — |
| Dua Nudbah | 0.59 | 88.8% (63.7%) | — |
| Dua Simaat | 0.20 | 82.5% (48.9%) | 80.0% (35.4%) |
| Dua Tawassul | 0.50 | 87.1% (18.5%) | 90.4% (6.3%) |
| Dua Tawba | 0.26 | 85.7% (57.1%) | — |
| Ziyarat Aminallah | 0.13 | 69.5% (44.7%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 79.1% | 75.6% |
| 5 s | 79.1% | 87.5% |
| 10 s | 90.9% | 92.8% |
| 20 s | 96.6% | 95.3% |
| 30 s | 97.5% | 97.5% |
