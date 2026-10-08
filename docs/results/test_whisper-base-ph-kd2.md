16 recordings, test split, ASR = whisper-base-ph-kd2, update every 1 s
source duaplayer, tier all: 16 recordings, 4.01 h, 6 voices, 0 venues (studio 16)

| method | line acc | line acc ±1 | refrain-line acc | du'a acc | wrong du'a shown | median lag | lines missed | jumps/min |
|---|---|---|---|---|---|---|---|---|
| v0.1 per-window matcher | 46.4% | 73.0% | 6.2% | 89.4% | 10.3% | 3.3 s | 14.2% | 5.14 |
| HMM tracker (identifies du'a) | 86.5% | 96.2% | 89.8% | 96.2% | 0.4% | 1.2 s | 1.2% | 0.03 |
| HMM tracker, du'a given | 87.7% | 98.0% | 90.1% | 98.5% | 0.0% | 1.1 s | 0.2% | 0.08 |

Per du'a, tracker (matcher):

| du'a | hours | line acc | refrain-line acc |
|---|---|---|---|
| Dua Baha | 0.36 | 86.2% (47.1%) | — |
| Hadith Kisa | 0.22 | 90.1% (47.2%) | 87.5% (41.7%) |
| Dua Hujjah (Faraj) | 0.02 | 66.3% (42.2%) | — |
| Dua Iftitah | 0.39 | 87.4% (59.5%) | 100.0% (12.5%) |
| Dua Kumayl | 0.42 | 89.3% (54.1%) | 100.0% (7.1%) |
| Dua Mujeer | 0.30 | 79.0% (11.0%) | 85.6% (0.0%) |
| Munajat Imam Ali (a) | 0.26 | 91.0% (55.9%) | — |
| Munajat Ta'ibeen | 0.14 | 80.0% (49.1%) | — |
| Dua Nudbah | 0.59 | 89.1% (64.3%) | — |
| Dua Simaat | 0.20 | 85.6% (47.8%) | 81.5% (32.3%) |
| Dua Tawassul | 0.50 | 88.2% (18.0%) | 91.6% (5.9%) |
| Dua Tawba | 0.26 | 86.4% (56.1%) | — |
| Ziyarat Aminallah | 0.13 | 70.8% (44.0%) | — |

Identification from a random point mid-recitation (320 starts):

| heard | v0.1 classifier | tracker |
|---|---|---|
| 3 s | 82.5% | 84.4% |
| 5 s | 83.4% | 90.3% |
| 10 s | 92.2% | 94.1% |
| 20 s | 96.2% | 96.2% |
| 30 s | 98.1% | 98.1% |
