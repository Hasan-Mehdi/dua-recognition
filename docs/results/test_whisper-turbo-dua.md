16 recordings, test split, ASR = whisper-turbo-dua, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 47.6% | 73.5% | 6.6% | 88.3% | 11.5% | 3.2 s | 14.2% | 3.83 |
| HMM tracker (identifies du'a) | 84.0% | 96.3% | 88.7% | 96.3% | 0.4% | 1.3 s | 1.1% | 0.04 |
| HMM tracker, du'a given | 85.1% | 98.0% | 88.7% | 98.5% | 0.0% | 1.3 s | 0.2% | 0.08 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 83.4% (49.9%) | — |
| Hadith Kisa | 0.22 | 86.2% (48.5%) | 91.7% (41.7%) |
| Dua Hujjah (Faraj) | 0.02 | 66.3% (44.6%) | — |
| Dua Iftitah | 0.39 | 85.8% (60.7%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 85.2% (54.7%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 74.2% (11.7%) | 82.7% (0.0%) |
| Munajat Imam Ali (a) | 0.26 | 91.2% (56.4%) | — |
| Munajat Ta'ibeen | 0.14 | 79.0% (52.1%) | — |
| Dua Nudbah | 0.59 | 88.3% (64.6%) | — |
| Dua Simaat | 0.20 | 80.2% (48.1%) | 78.5% (27.7%) |
| Dua Tawassul | 0.50 | 86.2% (20.6%) | 90.9% (6.8%) |
| Dua Tawba | 0.26 | 83.8% (56.6%) | — |
| Ziyarat Aminallah | 0.13 | 67.3% (45.6%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 81.6% | 75.6% |
| 5 s | 80.9% | 88.1% |
| 10 s | 91.6% | 92.2% |
| 20 s | 96.2% | 95.0% |
| 30 s | 97.5% | 96.9% |
