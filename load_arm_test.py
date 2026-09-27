import pybullet as p
import pybullet_data
import time

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")

# Load the KUKA iiwa robotic arm, base fixed at the origin
arm_start_pos = [0, 0, 0]
arm_id = p.loadURDF("kuka_iiwa/model.urdf", arm_start_pos, useFixedBase=True)

# Inspect the arm's joints
num_joints = p.getNumJoints(arm_id)
print(f"Robotic arm loaded. Total joints: {num_joints}\n")

for joint_index in range(num_joints):
    joint_info = p.getJointInfo(arm_id, joint_index)
    joint_name = joint_info[1].decode("utf-8")
    joint_type = joint_info[2]
    # Joint type 0 = revolute (rotational, controllable), 4 = fixed
    type_str = "REVOLUTE" if joint_type == 0 else ("FIXED" if joint_type == 4 else str(joint_type))
    print(f"Joint {joint_index}: {joint_name} ({type_str})")

# Let it settle under gravity for a few seconds (fixed base, so it shouldn't move much)
for i in range(300):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

print("\nArm settled. Simulation ending.")
p.disconnect()