import time
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], useFixedBase=True)
panda_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.62], useFixedBase=True)

num_joints = p.getNumJoints(panda_id)
print(f"Panda loaded. Total joints: {num_joints}\n")

for joint_index in range(num_joints):
    info = p.getJointInfo(panda_id, joint_index)
    name = info[1].decode("utf-8")
    joint_type = info[2]
    type_str = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(joint_type, str(joint_type))
    print(f"Joint {joint_index}: {name} ({type_str})")

cube_id = p.loadURDF("cube_small.urdf", [0.5, 0, 0.68], globalScaling=1.0)

print("\nSettling for 1 second...")
for _ in range(240):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

print("\nOpening gripper (fingers to 0.04)...")
for _ in range(120):
    p.setJointMotorControl2(panda_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=20)
    p.setJointMotorControl2(panda_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=20)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

print("Closing gripper (fingers to 0.0)...")
for _ in range(120):
    p.setJointMotorControl2(panda_id, 9, p.POSITION_CONTROL, targetPosition=0.0, force=20)
    p.setJointMotorControl2(panda_id, 10, p.POSITION_CONTROL, targetPosition=0.0, force=20)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

finger1 = p.getJointState(panda_id, 9)[0]
finger2 = p.getJointState(panda_id, 10)[0]
print(f"\nFinal finger positions: {finger1:.4f}, {finger2:.4f}")

print("\nDone. Window stays open 3 more seconds.")
time.sleep(3)
p.disconnect()
