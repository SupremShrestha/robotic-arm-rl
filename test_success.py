from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)
obs, info = env.reset()

print(f"Target position: {env.target_position}")
print(f"Success threshold: {env.success_threshold}\n")

# Manually place the end-effector suspiciously close to the target
# by teleporting joints directly (cheating on purpose, just to trigger success)
import pybullet as p
target_pos = env.target_position

# Use PyBullet's inverse kinematics to find joint angles that reach the target
joint_angles = p.calculateInverseKinematics(
    env.arm_id,
    env.end_effector_link_index,
    target_pos.tolist(),
    maxNumIterations=200,
    residualThreshold=1e-5,
)

for i, joint_index in enumerate(env.controllable_joints):
    p.resetJointState(env.arm_id, joint_index, joint_angles[i])

distance = env._get_distance_to_target()
print(f"Distance after IK teleport: {distance:.4f}")

# Now take a step - should register as success
action = np.zeros(7, dtype=np.float32)  # no movement, just check success detection
obs, reward, terminated, truncated, info = env.step(action)

print(f"\nAfter step:")
print(f"  reward: {reward:.4f}")
print(f"  terminated: {terminated}")
print(f"  info: {info}")

env.close()