"""Demonstration refinement for a fixed persistent three-object scene."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from robotic_arm.environments.persistent_pick_place_env import PersistentPickPlaceEnv
from robotic_arm.training.learn_pick_place import expert_action,fit
from robotic_arm.utils.output_paths import prepare_output


def rollout(env,model=None,teacher_probability=1.,seed=2000):
    rng=np.random.default_rng(seed)
    obs,_=env.reset(seed=seed)
    xs,ys,rows=[],[],[]
    for task in range(3):
        while True:
            teacher=expert_action(obs,"place",.34)
            xs.append(obs.copy());ys.append(teacher)
            action=teacher if model is None or rng.random()<teacher_probability else model.predict(obs,deterministic=True)[0]
            obs,_,t,tr,info=env.step(action)
            if t or tr:break
        rows.append({"task":task+1,**info})
        if not info["is_success"]:break
        if task<2:obs,_=env.next_task()
    return np.asarray(xs),np.asarray(ys),rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path("models/panda_distance_policy.zip"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=150)
    args=parser.parse_args()
    if args.epochs<1:parser.error("epochs must be positive")
    try:prepare_output(args.output)
    except ValueError as error:parser.error(str(error))
    torch.set_num_threads(1)
    model=PPO.load(args.model,device="cpu")
    if getattr(model,"environment_profile",None)!="quarter_turn_v2":parser.error("Use a quarter_turn_v2 model")
    env=PersistentPickPlaceEnv()
    history=[];best=-1
    try:
        x,y,rows=rollout(env,seed=46000)
        if len(rows)!=3 or not all(row["is_success"] for row in rows):raise RuntimeError("Physical teacher failed persistent sequence")
        for iteration in range(3):
            if iteration:
                nx,ny,_=rollout(env,model,teacher_probability=.5,seed=46000+iteration)
                x=np.concatenate([x,nx]);y=np.concatenate([y,ny])
            loss=fit(model,x,y,args.epochs,46000+iteration)
            _,_,rows=rollout(env,model,teacher_probability=0.,seed=47000)
            count=sum(row["is_success"] for row in rows)
            history.append({"round":iteration,"loss":loss,"tasks_completed":count,"samples":len(x)})
            print(f"Round {iteration}: {count}/3 persistent tasks",flush=True)
            model.save(args.output/f"round_{iteration}")
            if count>best:best=count;model.save(args.output/"best_model")
            if count==3:break
        np.savez_compressed(args.output/"demonstrations.npz",observations=x,actions=y)
        (args.output/"report.json").write_text(json.dumps({"method":"behavior cloning with optional mixed teacher/learner state aggregation, no new PPO","source":str(args.model),"history":history,"scope":"Same fixed three-object scene; seeds do not change these object positions."},indent=2))
    finally:env.close()


if __name__=="__main__":main()
