"""Focused imitation refinement on large objects and obstacle routes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
from robotic_arm.training.learn_pick_place import collect,fit,evaluate,ReleasePhaseObservation
from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
from robotic_arm.environments.yaw_aligned_pick_place_env import YawAlignedPickPlaceEnv


class FocusEnv(ObstaclePickPlaceEnv):
    def __init__(self,options):
        self.fixed_options=dict(options)
        cross=self.fixed_options.pop("_cross_workspace",False)
        super().__init__(stage="place",domain_randomization=False,cross_workspace=cross)

    def reset(self,seed=None,options=None):
        return super().reset(seed=seed,options={**self.fixed_options,**(options or {})})


class OrientedFocusEnv(OrientedPickPlaceEnv):
    def __init__(self,options):
        self.fixed_options=dict(options)
        cross=self.fixed_options.pop("_cross_workspace",False)
        super().__init__(stage="place",domain_randomization=False,cross_workspace=cross)

    def reset(self,seed=None,options=None):
        return super().reset(seed=seed,options={**self.fixed_options,**(options or {})})


class AlignedFocusEnv(YawAlignedPickPlaceEnv):
    def __init__(self,options):
        self.fixed_options=dict(options)
        cross=self.fixed_options.pop("_cross_workspace",False)
        super().__init__(stage="place",domain_randomization=False,cross_workspace=cross)

    def reset(self,seed=None,options=None):
        return super().reset(seed=seed,options={**self.fixed_options,**(options or {})})


def add_release_features(old,env):
    features=old.policy.features_extractor
    new=PPO("MlpPolicy",env,seed=42,device="cpu",n_steps=1024,batch_size=64,
        policy_kwargs={"net_arch":[256,256],"features_extractor_class":ReleasePhaseObservation,
        "features_extractor_kwargs":{"mean":features.mean.cpu().tolist(),"scale":features.scale.cpu().tolist()}})
    old_state=old.policy.state_dict();new_state=new.policy.state_dict()
    for key,target in new_state.items():
        source=old_state.get(key)
        if source is None: continue
        if target.shape==source.shape:
            target.copy_(source)
        elif target.ndim==2 and target.shape[0]==source.shape[0] and target.shape[1]==source.shape[1]+4:
            target.zero_();target[:,:source.shape[1]].copy_(source)
    new.policy.load_state_dict(new_state)
    return new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--random-layouts",action="store_true")
    parser.add_argument("--distance-tasks",action="store_true",help="Refine the 23/40/65 cm presets plus random diagonal layouts")
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--demonstrations",type=int,default=10)
    parser.add_argument("--quarter-turn",action="store_true")
    parser.add_argument("--varied-scenes",action="store_true")
    parser.add_argument("--align-yaw",action="store_true",help="Known simulator initial yaw drives low-level gripper alignment")
    parser.add_argument("--phase-features",action="store_true")
    parser.add_argument("--fresh-dataset",action="store_true")
    parser.add_argument("--training-seed",type=int,default=20000)
    parser.add_argument("--validation-seed",type=int,default=21000)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): parser.error("Use a new output folder")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    model=PPO.load(args.model,device="cpu")
    if model.observation_space.shape!=(32,): parser.error("Provide an obstacle-aware model")
    if min(args.epochs,args.rounds,args.demonstrations)<1: parser.error("Training budgets must be positive")
    with np.load(args.dataset) as data:
        x=data["observations"].copy(); y=data["actions"].copy()
    if x.ndim!=2 or x.shape[1]!=32 or y.shape!=(len(x),4) or not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        parser.error("Dataset must contain finite 32-value observations and four-value actions")
    env_class=AlignedFocusEnv if args.align_yaw else (OrientedFocusEnv if args.quarter_turn else FocusEnv)
    if args.phase_features:
        bootstrap=env_class({})
        try: model=add_release_features(model,bootstrap)
        finally: bootstrap.close()
    if args.quarter_turn: model.environment_profile="quarter_turn_v2"
    if args.align_yaw: model.environment_profile="yaw_aligned_v3"
    if args.fresh_dataset:
        x=np.empty((0,32),dtype=np.float32);y=np.empty((0,4),dtype=np.float32)
    conditions={"reference":{},"large":{"size":.06},"tall":{"obstacle_height":.24},
                "large_tall":{"size":.06,"obstacle_height":.24}}
    if args.varied_scenes:
        conditions.update(wide_left={"source_xy":[.36,-.22],"destination_xy":[.38,.25]},
            wide_right={"source_xy":[.53,-.20],"destination_xy":[.52,.24]},
            rotated_box={"size":.04,"object_yaw":.785398},
            shifted_obstacle={"obstacle_height":.16,"obstacle_offset_xy":[-.025,.02],"obstacle_half_xy":[.03,.045]})
    if args.random_layouts:
        conditions={"reference":{},"diagonal":{"_cross_workspace":True}}
    if args.distance_tasks:
        from robotic_arm.environments.scene_layouts import THREE_TASKS
        conditions={f"distance_{index+1}":{"source_xy":list(source),"destination_xy":list(goal)}
            for index,(source,goal) in enumerate(THREE_TASKS)}
        conditions["diagonal"]={"_cross_workspace":True}
    history=[]
    best=(-1,-1)
    for iteration in range(args.rounds):
        additions=[]
        for name,options in conditions.items():
            env=env_class(options)
            try:
                newx,newy,n=collect(env,list(range(args.training_seed+iteration*args.demonstrations,args.training_seed+(iteration+1)*args.demonstrations)),
                    model=None if iteration==0 else model,teacher_probability=.25,transport_height=.34)
                if len(newx): additions.append((newx,newy))
                print(f"Round {iteration} {name}: {n}/{args.demonstrations} training rollouts",flush=True)
            finally: env.close()
        for newx,newy in additions:
            x=np.concatenate([x,newx,newx]);y=np.concatenate([y,newy,newy])
        if not len(x):
            raise RuntimeError("No successful demonstrations were collected")
        np.savez_compressed(args.output/"demonstrations.npz",observations=x,actions=y)
        loss=fit(model,x,y,args.epochs,42+iteration)
        results={}
        for name,options in conditions.items():
            env=env_class(options)
            try:
                rows=evaluate(model,env,list(range(args.validation_seed,args.validation_seed+5)))
                results[name]=sum(r["is_success"] for r in rows)
            finally: env.close()
        score=(min(results.values()),sum(results.values()))
        history.append({"round":iteration,"loss":loss,"successes_out_of_5":results,"samples":len(x)})
        print(f"Focused validation {results}",flush=True)
        model.save(args.output/f"round_{iteration}")
        if score>best:
            best=score;model.save(args.output/"best_model")
    (args.output/"report.json").write_text(json.dumps({"method":"focused behavior cloning/DAgger; no additional PPO",
        "source":str(args.model),"history":history,"validation_seeds":[args.validation_seed,args.validation_seed+4],"config":vars(args)},indent=2,default=str),encoding="utf-8")


if __name__=="__main__": main()
