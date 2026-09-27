from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)
obs, info = env.reset()

initial_distance = env._get_distance_to_target()
print(f"Initial distance to target: {initial_distance:.4f}")
print(f"Target position: {env.target_position}")
print(f"Initial end-effector position: {obs[14:17]}\n")

# Move toward the target's general direction (positive x, y; downward)
# target is at (0.4, 0.3, 0.6), arm starts pointing straight up
action = np.array([0.5, 0.3, 0.5, -0.5, 0.0, 0.0, 0.0], dtype=np.float32)

print("Step | Reward   | Distance")
print("-" * 35)
for i in range(30):
    obs, reward, terminated, truncated, info = env.step(action)
    distance = env._get_distance_to_target()
    if i % 5 == 0:
        print(f"{i:>4} | {reward:>8.4f} | {distance:.4f}")

final_distance = env._get_distance_to_target()
print(f"\nFinal distance: {final_distance:.4f}")
print(f"Distance decreased: {final_distance < initial_distance}")

env.close()