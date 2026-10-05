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


    def create_target(position, radius=0.03, color=(1, 0, 0, 1)):
        """
        Creates a small floating sphere as a visual target marker.
        It has NO collision shape - the arm should be able to pass through it
        visually; we detect "reaching" via distance calculation, not physical
        collision.
        """
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        target_id = p.createMultiBody(
            baseMass=0,  # mass 0 = doesn't fall or get affected by gravity
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
        )
        return target_id


    def get_target_position(target_id):
        position, _orientation = p.getBasePositionAndOrientation(target_id)
        return position


    def set_target_position(target_id, new_position):
        # Keep orientation neutral since the sphere is symmetric
        p.resetBasePositionAndOrientation(target_id, new_position, [0, 0, 0, 1])


    # Place a target somewhere within the arm's reasonable reach
    # (KUKA iiwa's reach is roughly 0.8m from its base)
    target_start_pos = [0.4, 0.3, 0.6]
    target_id = create_target(target_start_pos)

    print(f"Target created at: {get_target_position(target_id)}")

    # Let the simulation sit for a moment so you can see the target and arm together
    for i in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Test moving the target to a new position
    new_target_pos = [-0.3, 0.4, 0.5]
    set_target_position(target_id, new_target_pos)
    print(f"Target moved to: {get_target_position(target_id)}")

    for i in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    time.sleep(2)
    p.disconnect()


if __name__ == "__main__":
    main()
