> Historical shaped-reward measurements. Current results and the controlled multi-seed experiment are in [docs/experiments](../experiments/README.md). The archived original checkpoint is distributed at `models/reference/initial_checkpoint.zip`.

# Measured benchmark results

These measurements were collected locally using the pinned CPU environment on 2026-10-05 (Asia/Katmandu). Each controller uses identical seeded targets: test seeds 0-199 and held-out seeds 1000-1199. Checkpoint selection uses validation seeds 10000-10019, not these evaluation sets. Success is within 5 cm; the horizon is 200 control steps. Evaluation uses the original distance reward, while the continued PPO run trains with shaped rewards. Reward values therefore must not be compared across training objectives.

| Controller | Test success | Held-out success | Test mean distance (m) | Held-out mean distance (m) | Ground-contact episodes, test / held-out |
|---|---|---|---|---|---|
| Original PPO | 111/200 (55.5%) | 100/200 (50.0%) | 0.2931 | 0.3039 | 0 / 0 |
| Continued PPO | 135/200 (67.5%) | 135/200 (67.5%) | 0.1459 | 0.1254 | 3 / 0 |
| Inverse kinematics | 200/200 (100.0%) | 200/200 (100.0%) | 0.0464 | 0.0466 | 0 / 0 |

Random control scored 0/200 on the test set, with mean distance 0.9408 m. The continued model scored 10/20 on seeds 0-19; improvement on the wider sets does not imply improvement on every subset. Original PPO scored 14/20 on that subset before this continuation.

The PPO continuation requested 100,000 additional steps and collected 100,352. The starting checkpoint had 450,000 training steps. Best validation performance was 17/20 (85%); the selected checkpoint has 540,000 total timesteps. The final policy after its last optimizer update scored 14/20 (70%) on validation; the earlier 85% checkpoint was retained. This is one training run, not evidence of multi-seed robustness. The continued PPO succeeds more often on both 200-target sets, but remains imperfect and has some ground contacts.

IK reached 400/400 across the two sets. This is a classical numerical controller, not reinforcement learning, and this finite test is not a universal guarantee. Ground contact logging excludes the fixed base links and does not check self-collisions. There are no obstacles, orientation objectives, grasping tasks or real-world deployment.

The improved checkpoint is `models/portfolio/best_model.zip`; original models remain available. Checkpoint SHA-256: `4e3c64d745fc0cba6f20faaf38cad2d8d17423d5e81e3bcecf5a53ee0e5b0fd6`.

Per-controller directories contain full JSON metrics with confidence intervals, episode CSVs and distance plots. Training config, validation logs and progress CSV are included here. Summary files preserve the original evaluation path under `runs/`; the distributed portfolio model has the same hash.

## Reproduce

```powershell
python evaluate.py --model models/portfolio/best_model.zip --episodes 200 --seed 0
python evaluate.py --model models/portfolio/best_model.zip --episodes 200 --seed 1000
python evaluate.py --controller ik --episodes 200 --seed 0
python evaluate.py --controller ik --episodes 200 --seed 1000
```

Exact values can vary across dependency versions and platforms. Use the pinned dependencies and compare model hashes.

## CV wording supported by these results

"Developed a KUKA robotic-arm reaching simulator using PyBullet, Gymnasium and PPO; increased held-out reaching success from 50% to 67.5% across 200 seeded targets within a 5 cm tolerance, with reproducible benchmarks, IK/random baselines, graphical demos and automated tests."

Discuss the remaining failure cases honestly in interviews. The IK result can be described separately as 400/400 tested targets, not as PPO's success rate.
