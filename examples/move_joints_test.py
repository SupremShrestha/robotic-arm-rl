from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

"""Standalone learning demo; run explicitly from the project root."""

def main():
    import pybullet as p
    import pybullet_data
    import time

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    plane_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

    controllable_joints = [0, 1, 2, 3, 4, 5, 6]

    # Target angles (radians) for each joint - chosen to visibly bend the arm
    target_angles = [0.5, -0.5, 0.3, -0.8, 0.2, 0.5, 0.0]

    print("Commanding joints to target angles...")
    print(f"Targets: {target_angles}")

    # POSITION_CONTROL: PyBullet's built-in PD controller drives the joint
    # toward the target angle over time, respecting force/velocity limits.
    for joint_index, target_angle in zip(controllable_joints, target_angles):
        p.setJointMotorControl2(
            bodyUniqueId=arm_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=300,  # matches the MaxForce we saw in Step 4
        )

    # Step the simulation so the motors actually move the arm
    for i in range(500):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Read back the actual joint positions achieved
    print("\nActual joint positions after moving:")
    for joint_index in controllable_joints:
        joint_state = p.getJointState(arm_id, joint_index)
        actual_angle = joint_state[0]
        print(f"Joint {joint_index}: target={target_angles[joint_index]:.3f}, actual={actual_angle:.3f}")

    # Check the new end-effector position
    end_effector_link_index = 6
    link_state = p.getLinkState(arm_id, end_effector_link_index)
    print(f"\nNew end-effector position: {link_state[0]}")

    time.sleep(2)  # pause so you can see the final pose before it closes
    p.disconnect()


if __name__ == "__main__":
    main()
