from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

"""Standalone learning demo; run explicitly from the project root."""

def main():
    import pybullet as p
    import pybullet_data
    import time

    # Connect to the physics server with a GUI window
    physics_client = p.connect(p.GUI)

    # Tell PyBullet where to find its built-in URDF models (ground plane, etc.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity (Earth gravity, pointing down on the Z axis)
    p.setGravity(0, 0, -9.8)

    # Load a ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Load a simple falling object (a cube) above the ground
    cube_start_pos = [0, 0, 1]
    cube_id = p.loadURDF("cube_small.urdf", cube_start_pos)

    print("Simulation started. Watch the cube fall and settle on the ground.")

    # Step the simulation for a few seconds
    for i in range(500):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # PyBullet's default physics timestep is 1/240s

    # Check where the cube ended up
    final_pos, final_orientation = p.getBasePositionAndOrientation(cube_id)
    print(f"Cube final position: {final_pos}")

    p.disconnect()
    print("Simulation ended cleanly.")


if __name__ == "__main__":
    main()
