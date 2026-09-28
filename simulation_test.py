import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
cube_id = p.loadURDF("cube_small.urdf", [0, 0, 1])

print("Simulation started. Watch the cube fall and settle on the ground.")

for i in range(500):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

final_pos, _ = p.getBasePositionAndOrientation(cube_id)
print(f"Cube final position: {final_pos}")

p.disconnect()
print("Simulation ended cleanly.")
