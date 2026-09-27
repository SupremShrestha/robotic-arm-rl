from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)
obs, info = env.reset()

print("Full observation vector:")
print(obs)
print(f"\nShape: {obs.shape}")

# Break it down into its parts to verify correctness
joint_positions = obs[0:7]
joint_velocities = obs[7:14]
end_effector_pos = obs[14:17]
target_pos = obs[17:20]

print(f"\nJoint positions (should be ~0, arm at rest): {joint_positions}")
print(f"Joint velocities (should be ~0, arm not moving yet): {joint_velocities}")
print(f"End-effector position (should be ~(0, 0, 1.281)): {end_effector_pos}")
print(f"Target position (should be (0.4, 0.3, 0.6)): {target_pos}")

# Now take a step with a real action and confirm the observation updates
action = np.array([0.5, -0.3, 0.2, -0.4, 0.1, 0.3, 0.0], dtype=np.float32)
obs2, reward, terminated, truncated, info = env.step(action)

print(f"\nAfter one step, joint velocities (should now be non-zero): {obs2[7:14]}")

env.close()