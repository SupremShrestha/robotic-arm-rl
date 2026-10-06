# Robotic Arm Learning in Simulation

A Python semester project using PyBullet, Gymnasium and Stable-Baselines3. It includes PPO target reaching with a KUKA arm, and learned physical-gripper pick-and-place with a Franka Panda arm. The recommended shared Panda checkpoint uses imitation learning and known simulator yaw for grasp alignment. This is a simulation-only project; hardware validation is outside its scope.

## Setup

Use Python 3.11. From the project root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_dependencies.py
```

If activation is unavailable, replace `python` with `.\.venv\Scripts\python.exe`. Dependencies are pinned in `requirements.txt`. New runs go under `outputs/`; saved submission evidence is under `results/`.

## Recommended shared simulation model

`models/panda_simulation_policy.zip` is one checkpoint tested across the implemented Panda task modes. It was trained with balanced behavior cloning across wider-position, diagonal, distance and persistent-object demonstrations, followed by additional upright object-yaw/size and obstacle demonstrations. It is **not a PPO-trained replacement**. Existing specialized checkpoints remain available.

The policy chooses four continuous actions: Cartesian x/y/z increments and finger opening. IK converts position/orientation targets into arm joint commands. Grasp alignment uses the known initial object yaw from the simulator; the neural policy does not infer orientation from an image or choose a yaw action. Evaluation uses policy predictions without the demonstration teacher. Objects are grasped through physical contact/friction, with no attachment constraints or in-episode object teleportation.

| Shared checkpoint check | Result | Scope |
| --- | --- | --- |
| 23/40/65 cm tasks | 3/3 | Exact layouts included in demonstrations |
| Persistent three-object sequence | 3/3 | Same fixed layouts; all three objects remain settled |
| Fresh random diagonal layouts | 20/20 | Seeds 56000-56019 |
| Expanded geometry/physics stress suite | 34/34 | Two paired target seeds per each of 17 cases |

The stress suite covers lighter/heavier objects, slippery surfaces, 4/5/6 cm size cases, box/cylinder shapes, one low/tall obstacle, combined conditions, wider positions, rotated boxes, and a shifted/resized obstacle. It does not test every combination. These finite results establish success on the saved scenarios, not perfect reliability or arbitrary-scene coverage.

[Shared model metadata](models/panda_simulation_selection.json), [distance results](results/shared_simulation_policy/distances/summary.json), [random diagonals](results/shared_simulation_policy/diagonal/summary.json), [persistent sequence](results/shared_simulation_policy/persistent/summary.json), and [stress results](results/shared_simulation_policy/stress/summary.json) contain the evidence and checkpoint hashes.

## Three tasks with different distances

```powershell
python -m robotic_arm evaluate-pick-place --model models/panda_simulation_policy.zip --three-tasks --seed 2000 --render --output outputs/my_three_distances
```

| Task | Source x/y (m) | Destination x/y (m) | Distance |
| --- | --- | --- | --- |
| 1 | (0.45, -0.10) | (0.45, 0.13) | 23 cm |
| 2 | (0.51, -0.18) | (0.39, 0.2015756806) | 40 cm |
| 3 | (0.30, -0.30) | (0.55, 0.30) | 65 cm |

This command keeps one window open and resets the arm/scene between episodes. For actual persistent objects and continuous arm state, use the next command.

## Three objects in one persistent scene

The shortest way to start the recommended simulation from the project root is:

```powershell
python run.py
```

The launcher automatically uses `.venv` when present, selects the shared model, and creates a unique output directory. Activation is not required. Use `python run.py --headless` for a terminal-only run.

```powershell
python -m robotic_arm persistent-pick-place --model models/panda_simulation_policy.zip --output outputs/my_persistent_demo
```

All three physical cubes exist from the beginning. The simulator and arm are not reset between tasks. The active object and destination change after successful placement. Previously placed objects remain in the scene, and the final report verifies that all three are still settled. A failed task stops the sequence and is reported honestly. Add `--headless` for evaluation without a window. Persistent objects have fixed upright 5 cm geometry; other objects are not included in the policy observation, so this is not general clutter-aware manipulation.

[Actual-motion recording](docs/media/persistent_pick_place.gif) shows the shared model. Create another GIF:

```powershell
python -m robotic_arm record-pick-place --model models/panda_simulation_policy.zip --output outputs/media/my_pick_place.gif
```

The camera provides rendered RGB frames for recording. It is not a camera-based learned perception pipeline.

## Random source and destination positions

```powershell
python -m robotic_arm evaluate-pick-place --model models/panda_simulation_policy.zip --three-tasks --random-layouts --render --output outputs/my_random_diagonals
```

The source is in a rear left/right region and the destination in the opposite front region. Sources sample x=0.35-0.40 or 0.50-0.55, y=-0.23 to -0.14; destinations sample the opposite x region, y=0.16-0.26. Sideways separation is at least 10 cm and straight-line distance at least 31 cm. Omitting `--seed` gives a new seed each run; the terminal and report record it. Specify a seed to repeat exactly the same layouts.

`--random-layouts` overrides the fixed distance presets. It cannot be combined with explicit source/destination coordinates or `--fixed-positions`. The three-object persistent command currently uses fixed presets, not this random sampler.

## Object physics, orientation and obstacles

```powershell
python -m robotic_arm evaluate-pick-place --model models/panda_simulation_policy.zip --size .06 --object-yaw .785398 --episodes 3 --render --output outputs/my_rotated_boxes
python -m robotic_arm evaluate-pick-place --model models/panda_simulation_policy.zip --size .06 --obstacle-height .24 --episodes 3 --render --output outputs/my_large_obstacle_demo
python -m robotic_arm benchmark-learned-pick-place --model models/panda_simulation_policy.zip --extended --varied-layouts --episodes-per-case 2 --seed 56000 --output outputs/my_stress_test
```

Evaluation also accepts `--mass`, `--friction`, `--object-shape box|cylinder`, `--source-xy X Y`, `--destination-xy X Y`, `--obstacle-offset-xy X Y`, and `--obstacle-half-xy X Y`. Coordinates are metres and yaw is radians.

`--domain-randomization` samples mass 0.04-0.15 kg, friction 0.5-2.0, size 0.04-0.06 m and box/cylinder shape. `--scene-randomization` samples wider positions, upright yaw in [-pi,pi], obstacle offsets up to 2.5 cm per axis, half-widths 2.5-4.5 cm and heights 0/8/16/24 cm. Explicit scene options take precedence. These samplers are available training distributions and tests; the selected model has not been verified on their entire cross-product.

Accepted object/goal coordinates are x=0.25-0.60 and y=-0.30-0.30. Acceptance does not guarantee IK reachability or policy success at every point. An 83 cm transfer is outside this bounded coordinate range. Roll/pitch object randomization, multiple/moving/unknown obstacles and arbitrary shapes are outside the current supported policy contract.

## Reinforcement learning and training

The KUKA reaching checkpoint is trained through PPO. Its saved evaluation achieved 409/500 target successes (81.8%). IK and random controllers are comparison baselines:

```powershell
python -m robotic_arm evaluate --controller ppo --model models/reaching_policy.zip --episodes 20 --render --output outputs/my_reaching_evaluation
python -m robotic_arm train --timesteps 500000 --seed 42 --run-dir outputs/my_reaching_training
```

The earlier nominal Panda checkpoint `panda_pick_place_policy.zip` uses behavior cloning, DAgger and 1,024 PPO fine-tuning steps; it achieved 50/50 saved limited-workspace placements. That does not establish PPO superiority or learning from scratch. More recent distance/obstacle/shared refinements are imitation-trained.

A 10,240-step PPO continuation of an intermediate shared model was checked at learning rate 1e-5 and target KL 0.001. Diagonal success remained 20/20, but mean error increased from 0.0111 to 0.0185 m and stress success decreased from 33/34 to 32/34. It was rejected as an improvement. The final recommended shared imitation model is a separate later refinement, not the PPO result. [Paired PPO comparison](results/shared_simulation_policy/ppo_comparison.json) and [training logs](results/shared_simulation_policy/ppo_training/validation.jsonl) preserve this experiment.

To run a new, bounded RL experiment:

```powershell
python -m robotic_arm train-pick-place --stage place --obstacle-aware --random-layouts --resume models/panda_simulation_policy.zip --learning-rate .00001 --target-kl .001 --timesteps 10240 --output outputs/my_panda_ppo_trial
```

Evaluate against the starting checkpoint on identical separate test seeds. A PPO run is not automatically an improvement. Timesteps are additional steps when resuming and may round up to a PPO rollout boundary. Closing a GUI stops its evaluation; Ctrl+C during PPO training saves an interrupted checkpoint. Saved environment profiles select compatible IK setups automatically.

Training commands include:

| Command | Purpose |
| --- | --- |
| `train` | PPO KUKA reaching |
| `train-pick-place` | PPO Panda reach/lift/place or resume |
| `learn-pick-place` | Demonstrations, DAgger, optional PPO |
| `refine-obstacle-policy` | Focused imitation refinement; `--distance-tasks`, `--random-layouts`, `--varied-scenes`, `--align-yaw` |
| `learn-persistent-pick-place` | Learned transitions in the fixed three-object scene |
| `train-shared-policy` | Balanced saved demonstrations, optional yaw refinement |

Reproduce the final shared refinement from the preserved intermediate checkpoint:

```powershell
python -m robotic_arm train-shared-policy --model results/shared_simulation_policy/yaw_training/candidate.zip --yaw-refinement --output outputs/my_shared_refinement
```

Default datasets are in `results/scene_generalization`, `results/diagonal_pick_place`, `results/distance_pick_place`, and `results/persistent_pick_place`. The final yaw refinement collected five physical demonstrations per size/yaw or obstacle condition at seeds 52000-52004. The teacher only runs during training. Source paths and parameters are preserved in the training reports.

## Success criteria and reports

Full placement requires prior both-finger grasp, lift above 15 cm while held, release, settled placement within 5 cm, and ten consecutive valid steps. In the obstacle-aware environment, obstacle contact immediately fails the episode. No success flag is forced to make a report look better.

`height` is final object-center height, not maximum lifting height. A settled 6 cm cube has center height 0.030 m. `grasped` and `lifted` describe earlier achievements; `current_grasp` describes present contact. Thread shutdown messages are normal cleanup.

Reports record seeds, model hashes, task positions, outcomes and errors. Evaluation, training and recording commands refuse to overwrite existing output artifacts; choose a new output path for each run. This protects previous successful and failed evidence. Use saved hashes when comparing checkpoints. Persistent-sequence results also check that earlier placements remain settled after subsequent tasks.

## Validation and project layout

```powershell
python -m unittest discover -s tests -v
python scripts/verify_training.py
```

Tests cover Gymnasium contracts, seeded scenes, action/joint limits, physical grasps/lifts/releases, disconnect handling, report preservation, persistent state transitions, camera rendering, and selected checkpoint hashes. Passing software tests does not guarantee every policy rollout succeeds.

```text
robotic-arm-rl/
  robotic_arm/
    environments/   # Physics, observations, scene layouts
    controllers/    # Separate scripted physical-gripper baseline
    training/       # PPO and demonstration learning
    evaluation/     # Learned policies, persistent sequences, recordings
    experiments/    # Benchmarks, reward comparison, failure analysis
    utils/          # Shared output protection
  tests/            # Automated validation
  scripts/          # Dependency and training integration checks
  models/           # Recommended and historical checkpoints + metadata
  results/          # Immutable experiment evidence
  docs/media/       # Actual simulation recordings
  outputs/          # Generated local runs, excluded from Git
  requirements.txt
  README.md
```

Historical model metadata identifies each checkpoint's own scope. `reaching_policy.zip` is KUKA PPO; `panda_pick_place_policy.zip` is earlier demonstration-assisted nominal Panda PPO; `panda_obstacle_policy_v2.zip`, `panda_varied_scene_policy.zip`, `panda_diagonal_policy.zip`, `panda_distance_policy.zip` and `panda_persistent_policy.zip` are specialists. `panda_simulation_policy.zip` is the recommended shared model for the documented current simulation demos.

Other commands: `pick-place` runs the scripted baseline; `benchmark-pick-place` measures its physics variations; `compare-rewards` compares reaching rewards; `analyze-failures` summarizes reaching reports; `record` records KUKA reaching. Run `python -m robotic_arm COMMAND --help` for all options.

## Semester project claims and remaining limits

A suitable title is **Robotic Arm Learning in Simulation: PPO Reaching and Demonstration-Assisted Manipulation**. Present the separation between RL and imitation, physical simulation assumptions, exact evaluation seeds and failure history. The shared policy passed its documented finite test suite and supports an actual persistent three-object demo; this is not universal reliability.

The project does not implement reliable arbitrary tilted-object grasping, arbitrary clutter or multiple/moving obstacle avoidance, camera-based learned state estimation, or real-world transfer. The simulator supplies exact object/obstacle state. More cameras or random scenes alone do not establish these capabilities. Hardware is intentionally excluded from this semester project. Future research should extend observation/action schemas and training, then evaluate new supported conditions independently.
