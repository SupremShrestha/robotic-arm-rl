from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)
obs, info = env.reset()

print("Initial joint positions:", obs[0:7])
print("Initial end-effector position:", obs[14:17])

action = np.array([1.0, -1.0, 0.5, -0.5, 0.3, 0.8, 0.0], dtype=np.float32)
for i in range(50):
    obs, reward, terminated, truncated, info = env.step(action)

print("\nAfter 50 steps of the same action:")
print("Joint positions:", obs[0:7])
print("End-effector position:", obs[14:17])
print("\n(Joint 0 should be roughly 1.0 rad with action_repeat=5.)")

env.close()
