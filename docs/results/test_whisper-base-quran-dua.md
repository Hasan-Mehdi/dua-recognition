16 recordings, test split, ASR = whisper-base-quran-dua, update every 1 s

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 47.4% | 72.6% | 6.4% | 87.5% | 12.3% | 3.2 s | 14.0% | 4.87 |
| HMM tracker (identifies du'a) | 85.4% | 96.4% | 89.0% | 96.5% | 1.0% | 1.2 s | 1.1% | 0.05 |
| HMM tracker, du'a given | 87.0% | 98.9% | 89.2% | 100.0% | 0.0% | 1.2 s | 0.4% | 0.17 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 84.4% (49.1%) | — |
| Hadith Kisa | 0.22 | 88.5% (47.1%) | 87.5% (37.5%) |
| Dua Hujjah (Faraj) | 0.02 | 68.7% (48.2%) | — |
| Dua Iftitah | 0.39 | 87.5% (61.4%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 87.4% (55.2%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 75.5% (13.0%) | 84.6% (0.3%) |
| Munajat Imam Ali (a) | 0.26 | 91.8% (55.3%) | — |
| Munajat Ta'ibeen | 0.14 | 81.2% (51.1%) | — |
| Dua Nudbah | 0.59 | 88.3% (64.2%) | — |
| Dua Simaat | 0.20 | 82.7% (49.3%) | 84.6% (30.8%) |
| Dua Tawassul | 0.50 | 87.3% (18.3%) | 90.5% (6.4%) |
| Dua Tawba | 0.26 | 85.2% (57.2%) | — |
| Ziyarat Aminallah | 0.13 | 70.8% (48.5%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 82.8% | 89.4% |
| 5 s | 81.2% | 92.5% |
| 10 s | 93.1% | 95.0% |
| 20 s | 96.9% | 97.8% |
| 30 s | 98.8% | 98.4% |
