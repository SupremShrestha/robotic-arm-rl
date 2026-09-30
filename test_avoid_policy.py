import argparse

import numpy as np
from stable_baselines3 import PPO

from environment.reach_avoid_env import ReachAvoidEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/runs/avoid_relvec_s0/best_model")
parser.add_argument("--obs", choices=["basic", "relvec"], default="relvec")
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

env = ReachAvoidEnv(render_mode=None, obs_mode=args.obs, success_threshold=args.threshold)
model = PPO.load(args.model)

successes = 0
collisions = 0
neither = 0
final_distances = []

for episode in range(args.episodes):
    if episode == 0:
        obs, info = env.reset(seed=args.seed)
    else:
        obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    final_distances.append(info["distance"])
    if info.get("is_success", False):
        successes += 1
    elif info.get("is_collision", False):
        collisions += 1
    else:
        neither += 1

n = args.episodes
print(f"Success rate:   {successes}/{n} ({successes/n*100:.0f}%)")
print(f"Collision rate: {collisions}/{n} ({collisions/n*100:.0f}%)")
print(f"Neither (ran out of time): {neither}/{n} ({neither/n*100:.0f}%)")
print(f"Average final distance: {np.mean(final_distances):.4f}")
env.close()
