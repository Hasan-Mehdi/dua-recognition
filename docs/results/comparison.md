| configuration | line acc | ±1 | refrain | wrong du'a | lag | jumps/min | found @3 s | @10 s |
|---|---|---|---|---|---|---|---|---|
| 6 s windows (current) | 85.4% | 96.4% | 89.0% | 1.0% | 1.2 s | 0.05 | 89% | 95% |
| + VAD | 84.8% | 96.5% | 88.4% | 0.6% | 1.3 s | 0.04 | 87% | 94% |
| + VAD + sticky lock | 84.6% | 96.5% | 88.2% | 1.0% | 1.3 s | 0.06 | 87% | 94% |
| streaming, full hypothesis | 78.4% | 94.7% | 85.5% | 3.2% | 0.5 s | 0.25 | 82% | 86% |
| streaming, committed only | 68.9% | 94.3% | 69.3% | 3.1% | 2.2 s | 0.11 | 82% | 86% |
| noisy room: 6 s windows | 76.9% | 92.1% | 78.1% | 1.1% | 1.5 s | 0.40 | 61% | 88% |
| noisy room: + VAD + sticky lock | 60.9% | 81.7% | 71.2% | 0.9% | 1.7 s | 0.51 | 51% | 80% |
| noisy room: room-trained model | 84.9% | 96.4% | 86.8% | 1.0% | 1.3 s | 0.07 | 84% | 94% |
| noisy room: room-trained + VAD + sticky | 72.4% | 91.5% | 82.1% | 0.7% | 1.5 s | 0.15 | 68% | 89% |
| clean: room-trained model | 85.6% | 96.5% | 88.2% | 0.9% | 1.2 s | 0.06 | 88% | 95% |
| clean: turbo fine-tuned (server) | 84.2% | 96.5% | 88.5% | 0.9% | 1.3 s | 0.05 | 88% | 95% |
| noisy room: turbo fine-tuned + VAD + sticky | 73.0% | 93.5% | 82.6% | 0.8% | 1.6 s | 0.14 | 74% | 92% |
| clean: base v3 (more data + room) | 85.1% | 96.5% | 87.8% | 0.9% | 1.2 s | 0.04 | 87% | 95% |
| noisy room: base v3 + VAD + sticky | 71.9% | 91.3% | 81.0% | 0.6% | 1.5 s | 0.21 | 68% | 90% |
