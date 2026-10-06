"""Physical-gripper pick and place using Panda IK; separate from PPO reaching."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pybullet as p
import pybullet_data


def run(render=True, source=(0.45, -0.15), destination=(0.45, 0.20),
        mass=0.08, friction=2.0, gif_path=None, verbose=True):
    source, destination = np.asarray(source,dtype=float), np.asarray(destination,dtype=float)
    if source.shape != (2,) or destination.shape != (2,) or not np.all(np.isfinite([source,destination])):
        raise ValueError("source and destination must each contain two finite coordinates")
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(friction) or friction < 0:
        raise ValueError("mass must be positive and friction nonnegative")
    if gif_path is not None and Path(gif_path).exists():
        raise ValueError("Output GIF exists; choose a new path")
    frames, step_count, ground_contact = [], 0, False
    phase = "Settling"
    scene = {"source":source.tolist(),"destination":destination.tolist(),"mass_kg":float(mass),
             "friction":float(friction),"controller":"scripted_ik_physical_gripper"}
    client = p.connect(p.GUI if render else p.DIRECT)
    if client < 0:
        raise RuntimeError("Cannot connect to PyBullet")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(1/240, physicsClientId=client)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=client)
        plane = p.loadURDF("plane.urdf", physicsClientId=client)
        arm = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=client)
        home = [0, -0.4, 0, -2.0, 0, 1.6, 0.8]
        for joint, angle in enumerate(home):
            p.resetJointState(arm, joint, angle, physicsClientId=client)
        for joint in (9, 10):
            p.resetJointState(arm, joint, 0.04, physicsClientId=client)
            p.changeDynamics(arm, joint, lateralFriction=friction, spinningFriction=0.01, physicsClientId=client)
        half = 0.025
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3, physicsClientId=client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=[0.9,0.35,0.1,1], physicsClientId=client)
        cube = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual, basePosition=[*source,half], physicsClientId=client)
        p.changeDynamics(cube, -1, lateralFriction=friction, spinningFriction=0.01, rollingFriction=0.001, physicsClientId=client)
        marker = p.createVisualShape(p.GEOM_CYLINDER, radius=0.065, length=0.002,
                                    rgbaColor=[0.1,0.8,0.3,0.7], physicsClientId=client)
        p.createMultiBody(baseVisualShapeIndex=marker, basePosition=[*destination,0.002], physicsClientId=client)
        if render:
            p.resetDebugVisualizerCamera(1.5, 55, -35, [0.35,0,0.3], physicsClientId=client)
        orientation = p.getQuaternionFromEuler([np.pi,0,0])
        finger_target = 0.04
        limits=[p.getJointInfo(arm,j,physicsClientId=client) for j in range(7)]
        lower=np.array([info[8] for info in limits])
        upper=np.array([info[9] for info in limits])

        def announce(message):
            nonlocal phase
            phase=message
            if verbose:
                print(message)

        def tick(count):
            nonlocal step_count, ground_contact
            for _ in range(count):
                p.setJointMotorControlArray(arm, [9,10], p.POSITION_CONTROL,
                        targetPositions=[finger_target]*2, forces=[10,10], physicsClientId=client)
                p.stepSimulation(physicsClientId=client)
                step_count += 1
                ground_contact = ground_contact or any(c[3]>0 for c in p.getContactPoints(arm,plane,physicsClientId=client))
                if gif_path and step_count % 24 == 0:
                    from PIL import Image, ImageDraw
                    view=p.computeViewMatrixFromYawPitchRoll([0.35,0,0.3],1.5,55,-35,0,2)
                    projection=p.computeProjectionMatrixFOV(60,640/480,0.1,10)
                    pixels=p.getCameraImage(640,480,viewMatrix=view,projectionMatrix=projection,
                            renderer=p.ER_TINY_RENDERER,physicsClientId=client)[2]
                    frame=Image.fromarray(np.asarray(pixels,dtype=np.uint8).reshape(480,640,4)[:,:,:3])
                    draw=ImageDraw.Draw(frame)
                    draw.rectangle((0,0,640,32),fill="white")
                    draw.text((8,8),f"Scripted IK pick and place | {phase}",fill="black")
                    frames.append(frame)
                if render:
                    time.sleep(1/240)

        def move(target, steps=240):
            start = np.array(p.getLinkState(arm,11,computeForwardKinematics=True,physicsClientId=client)[4])
            target = np.array(target)
            for index in range(steps):
                fraction = (index+1)/steps
                fraction = fraction*fraction*(3-2*fraction)
                position = start+(target-start)*fraction
                angles = p.calculateInverseKinematics(arm,11,position.tolist(),orientation,
                            maxNumIterations=100,residualThreshold=1e-5,physicsClientId=client)
                p.setJointMotorControlArray(arm,list(range(7)),p.POSITION_CONTROL,
                            targetPositions=np.clip(angles[:7],lower,upper).tolist(), forces=[87,87,87,87,12,12,12], physicsClientId=client)
                tick(1)
            tick(60)

        tick(120)
        announce("Approaching source...")
        move([*source,0.25])
        move([*source,0.035])
        announce("Closing fingers around the cube...")
        finger_target = 0.0
        tick(240)
        contacts = p.getContactPoints(arm,cube,physicsClientId=client)
        finger_contacts = {contact[3] for contact in contacts if contact[3] in (9,10)}
        announce("Lifting...")
        move([*source,0.28],steps=480)
        lifted_position = p.getBasePositionAndOrientation(cube,physicsClientId=client)[0]
        lifted = lifted_position[2] > 0.15
        if not lifted:
            result = {**scene, "success": False,"simulated_time_s":step_count/240,
                      "ground_contact":bool(ground_contact), "grasped_with_both_fingers": finger_contacts == {9,10},
                      "lifted": False, "failure": "Cube was not lifted; no attachment constraint was used."}
            if verbose:
                print(json.dumps(result,indent=2))
            return result
        announce("Moving to destination...")
        move([*destination,0.28],steps=360)
        move([*destination,0.035])
        announce("Releasing...")
        finger_target = 0.04
        tick(180)
        move([*destination,0.25])
        tick(240)
        position = p.getBasePositionAndOrientation(cube,physicsClientId=client)[0]
        velocity = p.getBaseVelocity(cube,physicsClientId=client)[0]
        error = float(np.linalg.norm(np.array(position[:2])-np.array(destination)))
        resting = abs(position[2]-half) < 0.015 and np.linalg.norm(velocity) < 0.03
        released = not p.getContactPoints(arm,cube,physicsClientId=client)
        success = bool(lifted and finger_contacts == {9,10} and error < 0.05 and resting and released)
        result = {**scene,"success":success,"simulated_time_s":step_count/240,
                  "ground_contact":bool(ground_contact),
                  "grasped_with_both_fingers":finger_contacts == {9,10}, "lifted":bool(lifted),
                  "released":bool(released), "resting":bool(resting), "placement_error_m":error,
                  "final_position":list(position), "source":list(source), "destination":list(destination)}
        if not success:
            result["failure"]="Final grasp/placement checks failed; inspect the reported stage metrics."
        if verbose:
            print(json.dumps(result,indent=2))
        return result
    except p.error:
        if not p.isConnected(client):
            print("Simulation window closed; pick-and-place stopped.")
            return {"success":False,"interrupted":True}
        raise
    finally:
        if p.isConnected(client):
            try:
                p.disconnect(client)
            except p.error:
                if p.isConnected(client):
                    raise
        if gif_path and frames:
            path=Path(gif_path)
            path.parent.mkdir(parents=True,exist_ok=True)
            frames[0].save(path,save_all=True,append_images=frames[1:],duration=100,loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source",type=float,nargs=2,default=[0.45,-0.15],metavar=("X","Y"))
    parser.add_argument("--destination",type=float,nargs=2,default=[0.45,0.20],metavar=("X","Y"))
    parser.add_argument("--mass",type=float,default=0.08)
    parser.add_argument("--friction",type=float,default=2.0)
    parser.add_argument("--gif",type=Path)
    args=parser.parse_args()
    if any(path is not None and path.exists() for path in (args.output,args.gif)):
        parser.error("Output file already exists; choose a new path")
    try:
        result=run(render=not args.headless,source=args.source,destination=args.destination,
                   mass=args.mass,friction=args.friction,gif_path=args.gif)
    except ValueError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Pick-and-place interrupted.")
        return
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not result.get("success") and not result.get("interrupted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
