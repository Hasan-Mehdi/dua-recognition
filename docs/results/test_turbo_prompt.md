16 recordings, test split, ASR = large-v3-turbo+prompt, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 46.7% | 70.4% | 6.7% | 85.0% | 14.6% | 3.3 s | 15.2% | 4.98 |
| HMM tracker (identifies du'a) | 84.8% | 96.4% | 88.0% | 96.4% | 0.7% | 1.2 s | 1.2% | 0.07 |
| HMM tracker, du'a given | 86.6% | 99.1% | 88.4% | 100.0% | 0.0% | 1.2 s | 0.4% | 0.17 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 84.4% (49.3%) | — |
| Hadith Kisa | 0.22 | 86.5% (45.3%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 68.7% (44.6%) | — |
| Dua Iftitah | 0.39 | 86.6% (58.6%) | 87.5% (12.5%) |
| Dua Kumayl | 0.42 | 87.1% (52.7%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 73.2% (11.0%) | 78.0% (0.0%) |
| Munajat Imam Ali (a) | 0.26 | 92.0% (54.1%) | — |
| Munajat Ta'ibeen | 0.14 | 77.0% (50.9%) | — |
| Dua Nudbah | 0.59 | 88.0% (65.1%) | — |
| Dua Simaat | 0.20 | 82.3% (45.7%) | 76.9% (26.2%) |
| Dua Tawassul | 0.50 | 88.1% (21.7%) | 91.8% (7.1%) |
| Dua Tawba | 0.26 | 85.0% (53.7%) | — |
| Ziyarat Aminallah | 0.13 | 70.8% (49.1%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 82.8% | 87.2% |
| 5 s | 77.2% | 92.2% |
| 10 s | 91.2% | 95.0% |
| 20 s | 95.9% | 98.1% |
| 30 s | 97.5% | 98.4% |
