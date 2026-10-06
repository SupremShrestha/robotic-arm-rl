"""Evaluate only learned actions, without scripted grasp or placement assistance."""
import argparse
import csv
import hashlib
import json
import secrets
import math
from pathlib import Path
from robotic_arm.utils.output_paths import prepare_output
from stable_baselines3 import PPO
from robotic_arm.environments.pick_place_env import PickPlaceEnv
from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
from robotic_arm.environments.yaw_aligned_pick_place_env import YawAlignedPickPlaceEnv
from robotic_arm.environments.reaching_env import SimulationDisconnected


from robotic_arm.environments.scene_layouts import THREE_TASKS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--three-tasks",action="store_true",help="Three consecutive tasks in one window, each with distinct source/destination")
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--random-layouts",action="store_true",help="Random diagonal layouts across different table regions; new layouts per run unless seeded")
    parser.add_argument("--stage",choices=["reach","lift","place"],default="place")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--fixed-positions",action="store_true")
    parser.add_argument("--start-near-cube",action="store_true")
    parser.add_argument("--domain-randomization",action="store_true")
    parser.add_argument("--scene-randomization",action="store_true")
    parser.add_argument("--mass",type=float)
    parser.add_argument("--friction",type=float)
    parser.add_argument("--size",type=float)
    parser.add_argument("--object-shape",choices=["box","cylinder"])
    parser.add_argument("--obstacle-height",type=float)
    parser.add_argument("--object-yaw",type=float)
    parser.add_argument("--source-xy",type=float,nargs=2)
    parser.add_argument("--destination-xy",type=float,nargs=2)
    parser.add_argument("--obstacle-offset-xy",type=float,nargs=2)
    parser.add_argument("--obstacle-half-xy",type=float,nargs=2)
    parser.add_argument("--output",type=Path,default=Path("outputs/pick_place_rl_evaluation"))
    args=parser.parse_args()
    if args.seed is None:
        args.seed=secrets.randbelow(1000000) if args.random_layouts else 2000
    if args.random_layouts and (args.fixed_positions or args.source_xy is not None or args.destination_xy is not None):
        parser.error("--random-layouts cannot be combined with fixed or explicit positions")
    print(f"Scene seed: {args.seed}",flush=True)
    if args.episodes<1 or not args.model.is_file():
        parser.error("Provide an existing model and positive episode count")
    if args.three_tasks:
        if args.source_xy is not None or args.destination_xy is not None:
            parser.error("--three-tasks supplies its own source and destination positions")
        args.episodes=3
    try:
        prepare_output(args.output)
    except ValueError as error:
        parser.error(str(error))
    model=PPO.load(str(args.model),device="cpu")
    env_class=OrientedPickPlaceEnv if getattr(model,"environment_profile",None)=="quarter_turn_v2" else (ObstaclePickPlaceEnv if model.observation_space.shape==(32,) else PickPlaceEnv)
    if getattr(model,"environment_profile",None)=="yaw_aligned_v3": env_class=YawAlignedPickPlaceEnv
    env=env_class(render_mode="human" if args.render else None,stage=args.stage,randomize=not args.fixed_positions,start_near_cube=args.start_near_cube,domain_randomization=args.domain_randomization,scene_randomization=args.scene_randomization,cross_workspace=args.random_layouts)
    scene_options={key:getattr(args,key) for key in ("mass","friction","size","object_shape","obstacle_height","object_yaw","source_xy","destination_xy","obstacle_offset_xy","obstacle_half_xy") if getattr(args,key) is not None}
    rows=[]
    try:
        if model.action_space != env.action_space or model.observation_space != env.observation_space:
            parser.error("Model spaces do not match the Panda environment; use a pick-and-place checkpoint")
        for episode in range(args.episodes):
            task_options=dict(scene_options)
            if args.three_tasks and not args.random_layouts:
                source,destination=THREE_TASKS[episode]
                task_options.update(source_xy=list(source),destination_xy=list(destination))
                print(f"Task {episode+1}/3: source={source}, destination={destination}, distance={math.dist(source,destination):.3f} m",flush=True)
            obs,_=env.reset(seed=args.seed+episode,options=task_options)
            if args.random_layouts:
                print(f"Task {episode+1}: source={env.scene_config['source_xy']}, destination={env.scene_config['destination_xy']}",flush=True)
            done=False
            while not done:
                action,_=model.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action)
                done=terminated or truncated
            rows.append({"seed":args.seed+episode,**info,"steps":env.steps})
            print(f"Episode {episode+1}: success={info['is_success']}, grasp={info['grasped']}, lift={info['lifted']}, height={info['cube_height_m']:.3f}, contact={info['current_grasp']}, placement_error={info['placement_error_m']:.4f} m")
    except ValueError as error:
        parser.error(str(error))
    except (KeyboardInterrupt,SimulationDisconnected):
        print("Evaluation interrupted; no complete benchmark saved.")
        return
    finally:
        env.close()
    n=len(rows)
    summary={"controller":"ppo_panda","stage":args.stage,"episodes":n,"seed":args.seed,
        "environment_profile":getattr(model,"environment_profile","obstacle_aware_v1" if model.observation_space.shape==(32,) else "nominal"),"observation_schema":int(model.observation_space.shape[0]),"max_episode_steps":env.max_episode_steps,"three_tasks":args.three_tasks,"random_layouts":args.random_layouts,"task_positions":THREE_TASKS if args.three_tasks and not args.random_layouts else [[r["source_xy"],r["destination_xy"]] for r in rows] if args.random_layouts else None,"scene_options":scene_options,"domain_randomization":args.domain_randomization,"scene_randomization":args.scene_randomization,"randomized_positions":not args.fixed_positions,"start_near_cube":args.start_near_cube,"reward_version":3,"model":str(args.model),
        "model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "successes":sum(r["is_success"] for r in rows),
        "success_rate":sum(r["is_success"] for r in rows)/n,
        "collision_free_success_rate":sum(r["is_success"] and not r["obstacle_contact"] for r in rows)/n,
        "obstacle_contact_episodes":sum(r["obstacle_contact"] for r in rows),
        "grasp_rate":sum(r["grasped"] for r in rows)/n,"lift_rate":sum(r["lifted"] for r in rows)/n,
        "release_rate":sum(r["released"] for r in rows)/n,
        "mean_placement_error_m":sum(r["placement_error_m"] for r in rows)/n}
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"{args.stage} success rate: {summary['successes']}/{n} ({summary['success_rate']:.1%})")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
