import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

num_joints = p.getNumJoints(arm_id)
print(f"Robotic arm loaded. Total joints: {num_joints}\n")

for joint_index in range(num_joints):
    info = p.getJointInfo(arm_id, joint_index)
    name = info[1].decode("utf-8")
    type_str = "REVOLUTE" if info[2] == 0 else ("FIXED" if info[2] == 4 else str(info[2]))
    print(f"Joint {joint_index}: {name} ({type_str})")

for i in range(300):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

print("\nArm settled. Simulation ending.")
p.disconnect()
