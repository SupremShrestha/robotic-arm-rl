import pybullet as p
import pybullet_data

physics_client = p.connect(p.DIRECT)  # no GUI needed, we're just reading data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

num_joints = p.getNumJoints(arm_id)

print(f"{'Idx':<4} {'Name':<20} {'LowerLim':>10} {'UpperLim':>10} {'MaxForce':>10} {'MaxVel':>10}")
print("-" * 70)

controllable_joints = []

for joint_index in range(num_joints):
    info = p.getJointInfo(arm_id, joint_index)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    lower_limit = info[8]
    upper_limit = info[9]
    max_force = info[10]
    max_velocity = info[11]

    if joint_type == p.JOINT_REVOLUTE:
        controllable_joints.append(joint_index)

    print(f"{joint_index:<4} {joint_name:<20} {lower_limit:>10.3f} {upper_limit:>10.3f} {max_force:>10.1f} {max_velocity:>10.3f}")

print(f"\nControllable (revolute) joint indices: {controllable_joints}")

# The last link is typically the end-effector for KUKA iiwa (link index = num_joints - 1)
end_effector_link_index = num_joints - 1
link_state = p.getLinkState(arm_id, end_effector_link_index)
end_effector_pos = link_state[0]  # world position (x, y, z)

print(f"End-effector link index: {end_effector_link_index}")
print(f"End-effector world position at rest: {end_effector_pos}")

p.disconnect()