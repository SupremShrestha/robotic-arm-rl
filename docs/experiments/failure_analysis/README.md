# PPO failure analysis

Observed 65 failures in 200 episodes; 0 failed episodes had ground contact.

| Seed | Final distance (m) | Target x, y, z (m) |
|---|---|---|
| 1067 | 0.6590 | 0.576, 0.362, 0.302 |
| 1069 | 0.6381 | 0.598, 0.338, 0.363 |
| 1075 | 0.6366 | 0.536, 0.379, 0.352 |
| 1096 | 0.6016 | 0.589, 0.315, 0.397 |
| 1080 | 0.5711 | 0.572, 0.284, 0.523 |
| 1106 | 0.5630 | 0.535, 0.323, 0.372 |
| 1063 | 0.5221 | 0.589, 0.243, 0.419 |
| 1146 | 0.5094 | 0.425, 0.361, 0.401 |
| 1029 | 0.5047 | 0.551, 0.258, 0.320 |
| 1081 | 0.5030 | 0.434, 0.344, 0.434 |

Diagnostics compare saturation, proximity to joint limits, and stationary distance between successful and failed episodes. A high diagnostic value is evidence to investigate; it is not proof of why a policy failed.

Replay a reported seed using `python evaluate.py --model models/portfolio/best_model.zip --episodes 1 --seed SEED --render`.

IK succeeds on these benchmark targets. PPO failures therefore cannot simply be dismissed as unreachable targets. Both use the same physics and target distribution, but PPO must learn joint coordination from data. Further controlled experiments are required to attribute improvement to a particular reward term.
