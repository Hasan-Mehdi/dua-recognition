16 recordings, test split, ASR = whisper-small-dua-v2, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 47.6% | 73.0% | 6.4% | 87.5% | 12.2% | 3.3 s | 14.0% | 4.29 |
| HMM tracker (identifies du'a) | 84.7% | 96.5% | 88.3% | 96.5% | 0.9% | 1.3 s | 1.1% | 0.05 |
| HMM tracker, du'a given | 86.1% | 98.8% | 88.5% | 100.0% | 0.0% | 1.3 s | 0.5% | 0.16 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 83.8% (49.5%) | — |
| Hadith Kisa | 0.22 | 87.1% (48.2%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 69.9% (49.4%) | — |
| Dua Iftitah | 0.39 | 86.2% (61.2%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 86.2% (55.2%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 74.2% (11.9%) | 82.7% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 92.0% (55.2%) | — |
| Munajat Ta'ibeen | 0.14 | 80.2% (49.9%) | — |
| Dua Nudbah | 0.59 | 88.3% (64.4%) | — |
| Dua Simaat | 0.20 | 81.9% (49.7%) | 80.0% (29.2%) |
| Dua Tawassul | 0.50 | 87.0% (19.1%) | 90.4% (6.4%) |
| Dua Tawba | 0.26 | 83.8% (56.8%) | — |
| Ziyarat Aminallah | 0.13 | 69.7% (50.0%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 84.1% | 88.1% |
| 5 s | 80.0% | 92.2% |
| 10 s | 92.8% | 95.0% |
| 20 s | 96.9% | 97.8% |
| 30 s | 98.8% | 98.8% |
