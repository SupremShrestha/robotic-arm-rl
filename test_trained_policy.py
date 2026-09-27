from stable_baselines3 import PPO
from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)
model = PPO.load("models/best_model/best_model")

num_test_episodes = 20
successes = 0
final_distances = []

for episode in range(num_test_episodes):
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    final_distances.append(info["distance"])
    if info.get("is_success", False):
        successes += 1

print(f"Success rate: {successes}/{num_test_episodes} ({successes/num_test_episodes*100:.0f}%)")
print(f"Average final distance: {np.mean(final_distances):.4f}")
print(f"Best final distance: {min(final_distances):.4f}")

env.close()