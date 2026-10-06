"""Evaluate a fixed-orientation learned policy with three objects in one persistent scene."""
import argparse
import hashlib
import json
from pathlib import Path
from stable_baselines3 import PPO
from robotic_arm.environments.persistent_pick_place_env import PersistentPickPlaceEnv
from robotic_arm.environments.reaching_env import SimulationDisconnected
from robotic_arm.utils.output_paths import prepare_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path("models/panda_simulation_policy.zip"))
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--seed",type=int,default=2000)
    parser.add_argument("--output",type=Path,default=Path("outputs/persistent_pick_place"))
    args=parser.parse_args()
    if not args.model.is_file(): parser.error("Model does not exist")
    try: prepare_output(args.output)
    except ValueError as error: parser.error(str(error))
    model=PPO.load(args.model,device="cpu")
    if getattr(model,"environment_profile",None) not in ("quarter_turn_v2","yaw_aligned_v3"):
        parser.error("Use a quarter_turn_v2 or yaw_aligned_v3 policy for this persistent scene")
    env=PersistentPickPlaceEnv(render_mode=None if args.headless else "human")
    rows=[]
    try:
        obs,_=env.reset(seed=args.seed)
        for task in range(3):
            while True:
                action,_=model.predict(obs,deterministic=True)
                obs,_,t,tr,info=env.step(action)
                if t or tr: break
            rows.append({"task":task+1,**info,"steps":env.steps})
            print(f"Task {task+1}: success={info['is_success']}, error={info['placement_error_m']:.4f} m",flush=True)
            if not info["is_success"]: break
            if task<2: obs,_=env.next_task()
        settled=[]
        from robotic_arm.environments.scene_layouts import THREE_TASKS
        import numpy as np
        import pybullet as p
        for body,(_,goal) in zip(env.objects,THREE_TASKS):
            position=p.getBasePositionAndOrientation(body,physicsClientId=env.physics_client)[0]
            velocity=p.getBaseVelocity(body,physicsClientId=env.physics_client)[0]
            settled.append(bool(np.linalg.norm(np.array(position[:2])-goal)<.05 and abs(position[2]-.025)<.015 and np.linalg.norm(velocity)<.03))
    except (KeyboardInterrupt,SimulationDisconnected):
        print("Simulation interrupted; no complete result saved.")
        return
    finally: env.close()
    successes=sum(row["is_success"] for row in rows)
    report={"model":str(args.model),"sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"seed":args.seed,
        "persistent_scene":True,"planned_tasks":3,"successes":successes,"success_rate":successes/3,
        "all_objects_finally_settled":settled,"sequence_success":successes==3 and all(settled),"tasks":rows,
        "limits":"Other objects are physical but absent from the policy observation; collisions with them are not explicitly penalized. No obstacle or orientation generalization claim."}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print(f"Persistent placement: {successes}/3; all objects settled={all(settled)}")


if __name__=="__main__": main()
