import time
import numpy as np
import pybullet as p

from environment.pick_place_env import PickPlaceEnv

env = PickPlaceEnv(render_mode="human")
obs, info = env.reset(seed=42)

print(f"Cube position: {env.cube_position}")
print(f"Goal position: {env.goal_position}")

def move_gripper_to(target_pos, gripper_open, steps=120):
    for _ in range(steps):
        joint_angles = p.calculateInverseKinematics(
            env.panda_id, env.grasp_frame_link, target_pos.tolist(),
            maxNumIterations=50, physicsClientId=env.physics_client,
        )
        for i, j in enumerate(env.arm_joints):
            p.setJointMotorControl2(
                env.panda_id, j, p.POSITION_CONTROL,
                targetPosition=joint_angles[i], force=300,
                physicsClientId=env.physics_client,
            )
        finger_target = 0.04 if gripper_open else 0.0
        for j in env.finger_joints:
            p.setJointMotorControl2(
                env.panda_id, j, p.POSITION_CONTROL,
                targetPosition=finger_target, force=40,
                physicsClientId=env.physics_client,
            )
        p.stepSimulation(physicsClientId=env.physics_client)
        time.sleep(1.0 / 240.0)

cube_pos = env.cube_position.copy()
goal_pos = env.goal_position.copy()

print("\n1. Moving above cube (open gripper)...")
move_gripper_to(cube_pos + np.array([0, 0, 0.15]), gripper_open=True, steps=150)

print("2. Descending to cube...")
move_gripper_to(cube_pos + np.array([0, 0, 0.02]), gripper_open=True, steps=150)

print("3. Closing gripper...")
move_gripper_to(cube_pos + np.array([0, 0, 0.02]), gripper_open=False, steps=100)

is_grasping = env._check_grasping()
print(f"   is_grasping after close: {is_grasping}")

print("4. Lifting...")
move_gripper_to(cube_pos + np.array([0, 0, 0.20]), gripper_open=False, steps=150)

cube_now = env._get_cube_position()
lifted = (cube_now[2] - env.table_height) > env.lift_height
print(f"   cube height above table: {cube_now[2] - env.table_height:.4f}  (lifted={lifted})")

print("5. Moving to above goal...")
move_gripper_to(goal_pos + np.array([0, 0, 0.20]), gripper_open=False, steps=200)

print("6. Descending to goal...")
move_gripper_to(goal_pos + np.array([0, 0, 0.02]), gripper_open=False, steps=150)

print("7. Opening gripper (release)...")
move_gripper_to(goal_pos + np.array([0, 0, 0.02]), gripper_open=True, steps=100)

print("8. Retreating...")
move_gripper_to(goal_pos + np.array([0, 0, 0.2]), gripper_open=True, steps=100)

final_cube = env._get_cube_position()
placed = env._is_placed(final_cube)
print(f"\nFinal cube position: {final_cube}")
print(f"Distance to goal (xy): {np.linalg.norm(final_cube[:2] - goal_pos[:2]):.4f}")
print(f"is_placed: {placed}")

time.sleep(2)
env.close()
