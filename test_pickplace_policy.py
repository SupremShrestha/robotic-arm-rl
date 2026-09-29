import argparse

import numpy as np
from stable_baselines3 import PPO

from environment.pick_place_env import PickPlaceEnv

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/runs/pickplace_s0/best_model")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

env = PickPlaceEnv(render_mode=None)
model = PPO.load(args.model)

never_grasped = 0
grasped_then_dropped = 0
placed = 0
timed_out_while_holding = 0

for episode in range(args.episodes):
    if episode == 0:
        obs, info = env.reset(seed=args.seed)
    else:
        obs, info = env.reset()
    last_info = {}
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        if terminated or truncated:
            break

    if last_info.get("is_success", False):
        placed += 1
    elif last_info.get("dropped", False):
        grasped_then_dropped += 1
    elif not last_info.get("has_been_grasped", False):
        never_grasped += 1
    else:
        timed_out_while_holding += 1

n = args.episodes
print(f"Placed (success):          {placed}/{n} ({placed/n*100:.0f}%)")
print(f"Never grasped the cube:    {never_grasped}/{n} ({never_grasped/n*100:.0f}%)")
print(f"Grasped then dropped:      {grasped_then_dropped}/{n} ({grasped_then_dropped/n*100:.0f}%)")
print(f"Timed out still holding:   {timed_out_while_holding}/{n} ({timed_out_while_holding/n*100:.0f}%)")
env.close()
