"""Stress-test a frozen policy on paired scenes and explicit physical variations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from stable_baselines3 import PPO
from robotic_arm.environments.pick_place_env import PickPlaceEnv
from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
from robotic_arm.environments.yaw_aligned_pick_place_env import YawAlignedPickPlaceEnv

CASES={
    "reference":{}, "light":{"mass":.04}, "heavy":{"mass":.15},
    "slippery":{"friction":.5}, "small":{"size":.04}, "large":{"size":.06},
    "cylinder":{"object_shape":"cylinder"},
    "low_obstacle":{"obstacle_height":.08}, "blocking_obstacle":{"obstacle_height":.24},
    "heavy_slippery":{"mass":.15,"friction":.5},
}


def benchmark(model_path,output,episodes=3,seed=6000,extended=False,varied_layouts=False):
    if episodes<1: raise ValueError("episodes must be positive")
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output folder to preserve previous reports")
    model=PPO.load(str(model_path),device="cpu")
    env_class=OrientedPickPlaceEnv if getattr(model,"environment_profile",None)=="quarter_turn_v2" else (ObstaclePickPlaceEnv if model.observation_space.shape==(32,) else PickPlaceEnv)
    if getattr(model,"environment_profile",None)=="yaw_aligned_v3": env_class=YawAlignedPickPlaceEnv
    env=env_class(stage="place")
    cases_to_test=dict(CASES)
    if extended:
        cases_to_test.update(large_blocking_obstacle={"size":.06,"obstacle_height":.24},
            cylinder_blocking_obstacle={"object_shape":"cylinder","obstacle_height":.24})
    if varied_layouts:
        cases_to_test.update(wide_left={"source_xy":[.36,-.22],"destination_xy":[.38,.25]},
            wide_right={"source_xy":[.53,-.20],"destination_xy":[.52,.24]},
            rotated_small_box={"size":.04,"object_yaw":.785398},
            rotated_large_box={"size":.06,"object_yaw":.785398},
            shifted_obstacle={"obstacle_height":.16,"obstacle_offset_xy":[-.025,.02],"obstacle_half_xy":[.03,.045]})
    rows=[]
    try:
        if model.action_space!=env.action_space or model.observation_space!=env.observation_space:
            raise ValueError("Incompatible policy")
        for name,options in cases_to_test.items():
            for episode in range(episodes):
                obs,_=env.reset(seed=seed+episode,options=options)
                while True:
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,t,tr,info=env.step(action)
                    if t or tr:
                        rows.append({"case":name,"seed":seed+episode,**info,"steps":env.steps});break
            selected=[r for r in rows if r["case"]==name]
            print(f"{name}: {sum(r['is_success'] for r in selected)}/{episodes}; obstacle contacts={sum(r['obstacle_contact'] for r in selected)}",flush=True)
    finally:
        env.close()
    cases={}
    for name in cases_to_test:
        selected=[r for r in rows if r["case"]==name]
        cases[name]={"episodes":episodes,"successes":sum(r["is_success"] for r in selected),
            "collision_free_successes":sum(r["is_success"] and not r["obstacle_contact"] for r in selected),
            "obstacle_contact_episodes":sum(r["obstacle_contact"] for r in selected),
            "mean_placement_error_m":sum(r["placement_error_m"] for r in selected)/episodes}
    summary={"model":str(model_path),"sha256":hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "environment_profile":getattr(model,"environment_profile","obstacle_aware_v1" if model.observation_space.shape==(32,) else "nominal"),"observation_schema":int(model.observation_space.shape[0]),"max_episode_steps":env.max_episode_steps,"extended_suite":extended,"varied_layouts":varied_layouts,"seed":seed,"cases":cases,"episodes":len(rows),"successes":sum(r["is_success"] for r in rows),
        "scope":"Finite stress test; paired target seeds across conditions. No camera noise or hardware; two simple upright geometries; obstacle contact is distinct from placement success."}
    output.mkdir(parents=True,exist_ok=True)
    with (output/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path("models/panda_pick_place_policy.zip"))
    parser.add_argument("--episodes-per-case",type=int,default=10)
    parser.add_argument("--seed",type=int,default=6000)
    parser.add_argument("--extended",action="store_true")
    parser.add_argument("--varied-layouts",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("outputs/learned_robustness"))
    args=parser.parse_args()
    try: benchmark(args.model,args.output,args.episodes_per_case,args.seed,args.extended,args.varied_layouts)
    except ValueError as error: parser.error(str(error))


if __name__=="__main__": main()
