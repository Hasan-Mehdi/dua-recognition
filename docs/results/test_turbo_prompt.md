16 recordings, test split, ASR = large-v3-turbo+prompt, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 46.3% | 69.9% | 6.7% | 84.4% | 15.2% | 3.3 s | 15.3% | 4.75 |
| HMM tracker (identifies du'a) | 84.8% | 96.2% | 88.1% | 96.2% | 0.4% | 1.2 s | 1.2% | 0.06 |
| HMM tracker, du'a given | 86.0% | 98.0% | 88.4% | 98.4% | 0.0% | 1.2 s | 0.2% | 0.12 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 84.5% (49.1%) | — |
| Hadith Kisa | 0.22 | 86.9% (45.3%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 67.5% (43.4%) | — |
| Dua Iftitah | 0.39 | 86.5% (57.4%) | 87.5% (12.5%) |
| Dua Kumayl | 0.42 | 87.1% (52.5%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 73.3% (10.9%) | 78.0% (0.0%) |
| Munajat Imam Ali (a) | 0.26 | 91.6% (54.0%) | — |
| Munajat Ta'ibeen | 0.14 | 76.0% (50.9%) | — |
| Dua Nudbah | 0.59 | 87.6% (64.8%) | — |
| Dua Simaat | 0.20 | 83.2% (45.7%) | 78.5% (26.2%) |
| Dua Tawassul | 0.50 | 88.1% (21.7%) | 91.8% (7.1%) |
| Dua Tawba | 0.26 | 84.9% (53.5%) | — |
| Ziyarat Aminallah | 0.13 | 69.7% (45.8%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 79.7% | 70.3% |
| 5 s | 75.9% | 86.9% |
| 10 s | 89.4% | 92.5% |
| 20 s | 94.4% | 96.2% |
| 30 s | 96.2% | 96.9% |
