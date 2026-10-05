# PPO failure analysis

Observed 91 failures in 500 episodes; 0 failed episodes had ground contact.

| Seed | Final distance (m) | Target x, y, z (m) |
|---|---|---|
| 2415 | 0.4847 | 0.393, 0.388, 0.406 |
| 2059 | 0.4710 | 0.384, 0.308, 0.303 |
| 2307 | 0.4633 | 0.397, 0.378, 0.429 |
| 2100 | 0.4511 | 0.390, 0.359, 0.407 |
| 2184 | 0.4502 | 0.399, 0.342, 0.396 |
| 2246 | 0.4340 | 0.408, 0.353, 0.459 |
| 2489 | 0.4311 | 0.401, 0.276, 0.343 |
| 2181 | 0.4111 | 0.414, 0.250, 0.371 |
| 2382 | 0.4094 | 0.405, 0.300, 0.427 |
| 2179 | 0.4089 | 0.315, 0.103, 0.320 |

Diagnostics compare saturation, proximity to joint limits, and stationary distance between successful and failed episodes. A high diagnostic value is evidence to investigate; it is not proof of why a policy failed.

Replay a reported seed using `python evaluate.py --model models/portfolio/legacy_best_model.zip --episodes 1 --seed SEED --render`.

IK succeeds on these benchmark targets. PPO failures therefore cannot simply be dismissed as unreachable targets. Both use the same physics and target distribution, but PPO must learn joint coordination from data. Further controlled experiments are required to attribute improvement to a particular reward term.
