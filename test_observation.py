from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)
obs, info = env.reset()

print("Full observation vector:")
print(obs)
print(f"\nShape: {obs.shape}")

print(f"\nJoint positions (should be ~0): {obs[0:7]}")
print(f"Joint velocities (should be ~0): {obs[7:14]}")
print(f"End-effector position (should be ~(0, 0, 1.281)): {obs[14:17]}")
print(f"Target position (should be (0.4, 0.3, 0.6)): {obs[17:20]}")

action = np.array([0.5, -0.3, 0.2, -0.4, 0.1, 0.3, 0.0], dtype=np.float32)
obs2, reward, terminated, truncated, info = env.step(action)
print(f"\nAfter one step, joint velocities: {obs2[7:14]}")

env.close()
