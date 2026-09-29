import argparse

import numpy as np
from stable_baselines3 import PPO

from environment.reach_env import ReachEnv
from environment.reach_avoid_env import ReachAvoidEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["reach", "avoid"], default="avoid")
parser.add_argument("--model", required=True)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
args = parser.parse_args()

EnvClass = ReachEnv if args.env == "reach" else ReachAvoidEnv
model = PPO.load(args.model)

print(f"{'Noise std':>10} | {'Success':>10} | {'Collision':>10}" if args.env == "avoid" else f"{'Noise std':>10} | {'Success':>10}")
print("-" * (35 if args.env == "avoid" else 24))

for noise_std in args.noise_levels:
    env = EnvClass(render_mode=None)
    successes = 0
    collisions = 0

    for episode in range(args.episodes):
        if episode == 0:
            obs, info = env.reset(seed=args.seed)
        else:
            obs, info = env.reset()
        while True:
            noisy_obs = obs + np.random.normal(0, noise_std, size=obs.shape).astype(np.float32)
            action, _states = model.predict(noisy_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if info.get("is_success", False):
            successes += 1
        if info.get("is_collision", False):
            collisions += 1

    env.close()
    n = args.episodes
    if args.env == "avoid":
        print(f"{noise_std:>10.3f} | {successes/n*100:>9.0f}% | {collisions/n*100:>9.0f}%")
    else:
        print(f"{noise_std:>10.3f} | {successes/n*100:>9.0f}%")
