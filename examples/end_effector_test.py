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
    end_effector_link_index = 6


    def get_end_effector_position(body_id, link_index):
        """
        Returns the world-frame (x, y, z) position of the end-effector.
        This will be reused directly inside the Gymnasium environment.
        """
        link_state = p.getLinkState(body_id, link_index)
        world_position = link_state[0]  # (x, y, z) tuple
        return world_position


    # Command a smooth, continuous motion (not one instant jump) so we can
    # sample the end-effector position WHILE it's moving.
    target_angles = [0.8, -0.6, 0.4, -1.0, 0.3, 0.7, 0.2]

    for joint_index, target_angle in zip(controllable_joints, target_angles):
        p.setJointMotorControl2(
            bodyUniqueId=arm_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=300,
        )

    print("Sampling end-effector position during motion:\n")

    for i in range(600):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        # Sample every 100 steps to see the position change over time
        if i % 100 == 0:
            pos = get_end_effector_position(arm_id, end_effector_link_index)
            print(f"Step {i:>4}: end-effector position = "
                  f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    final_pos = get_end_effector_position(arm_id, end_effector_link_index)
    print(f"\nFinal end-effector position: {final_pos}")

    time.sleep(2)
    p.disconnect()


if __name__ == "__main__":
    main()
