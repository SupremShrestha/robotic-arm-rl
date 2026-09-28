from environment.reach_env import ReachEnv
import numpy as np
import pybullet as p

env = ReachEnv(render_mode=None)
obs, info = env.reset()

print(f"Target position: {env.target_position}")
print(f"Success threshold: {env.success_threshold}\n")

# Testing shortcut only: use inverse kinematics to teleport near the target.
joint_angles = p.calculateInverseKinematics(
    env.arm_id,
    env.end_effector_link_index,
    env.target_position.tolist(),
    maxNumIterations=200,
    residualThreshold=1e-5,
    physicsClientId=env.physics_client,
)

for i, joint_index in enumerate(env.controllable_joints):
    p.resetJointState(env.arm_id, joint_index, joint_angles[i],
                      physicsClientId=env.physics_client)

print(f"Distance after IK teleport: {env._get_distance_to_target():.4f}")

action = np.zeros(7, dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)

print("\nAfter step:")
print(f"  reward: {reward:.4f}")
print(f"  terminated: {terminated}")
print(f"  info: {info}")

env.close()
