"""Demonstration-assisted manipulation: behavior cloning, DAgger, then optional PPO."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robotic_arm.environments.pick_place_env import PickPlaceEnv
from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
from robotic_arm.training.train_pick_place import PickPlaceValidation


class StandardizedObservation(BaseFeaturesExtractor):
    """Saved physical-unit scaling; evaluation uses the identical fixed transform."""
    def __init__(self, observation_space, mean, scale):
        super().__init__(observation_space,features_dim=observation_space.shape[0])
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))

    def forward(self, observations):
        return (observations-self.mean)/self.scale


class ReleasePhaseObservation(StandardizedObservation):
    """Observable task predicates, never action overrides or a scripted phase controller."""
    def __init__(self,observation_space,mean,scale):
        super().__init__(observation_space,mean,scale)
        self._features_dim=36

    def forward(self,obs):
        base=super().forward(obs)
        near=torch.linalg.vector_norm(obs[:,3:5]-obs[:,6:8],dim=1)<.028
        low=obs[:,5]<obs[:,30]/2+.07
        ready=near & low & (obs[:,21]>.5)
        opened=obs[:,18:20].min(dim=1).values>.035
        contact=(obs[:,22]>.5)&(obs[:,23]>.5)
        clearance=obs[:,2]>2*obs[:,29]+.075
        predicates=torch.stack([ready,opened,contact,clearance],dim=1).float()*2-1
        return torch.cat([base,predicates],dim=1)


def expert_action(obs, stage="lift", transport_height=.21):
    """Training-only feedback teacher; no environment modifications or attachments."""
    ee,cube=obs[:3],obs[3:6]
    contact=bool(obs[22] and obs[23])
    goal=obs[6:9]
    size=obs[30] if len(obs)==32 else .05
    grasp_height=max(.035,size/2+.01)
    close_height=grasp_height+.008
    at_goal=np.linalg.norm(cube[:2]-goal[:2])<(.028 if len(obs)==32 else .025)
    if stage=="place" and obs[21]:
        if at_goal and cube[2]<size/2+(.07 if len(obs)==32 else .025):
            if len(obs)==32 and min(obs[18:20])<.035:
                target=ee.copy()  # Open fully before withdrawing; do not drag the released object.
            else:
                target=np.array([ee[0],ee[1],.22])
            grip=1.
        elif contact:
            if not at_goal and ee[2]<transport_height-.025:
                target=np.array([cube[0],cube[1],transport_height])
            else:
                target=np.array([goal[0],goal[1],grasp_height if at_goal else transport_height])
            grip=-1.
        else:
            # Recover an accidental drop away from the destination using ordinary grasp actions.
            aligned=np.linalg.norm(ee[:2]-cube[:2])<.008
            target=np.array([cube[0],cube[1],grasp_height if aligned else max(ee[2],.08)])
            grip=-1. if aligned and ee[2]<close_height else 1.
    elif contact:
        target=np.array([cube[0],cube[1],.21])
        grip=-1.
    else:
        aligned=np.linalg.norm(ee[:2]-cube[:2])<.008
        low=ee[2]<close_height
        target=np.array([cube[0],cube[1],grasp_height if aligned else max(ee[2],.08)])
        grip=-1. if aligned and low else 1.
    if len(obs)==32 and stage=="place" and obs[21] and contact and obs[29]>0 and not at_goal:
        # Training-only corridor around the obstacle; the learner receives obstacle geometry.
        clearance=max(transport_height,2*obs[29]+.10)
        if ee[2]<clearance-.025:
            target=np.array([cube[0],cube[1],clearance])
        elif ee[1]<obs[25]+obs[28]+.07:
            target=np.array([.62,ee[1] if ee[0]<.60 else goal[1],clearance])
        else:
            target=np.array([goal[0],goal[1],clearance])
        grip=-1.
    return np.r_[np.clip((target-ee)/.01,-1,1),grip].astype(np.float32)


def collect(env, seeds, model=None, teacher_probability=1., transport_height=.21):
    observations,actions=[],[]
    successes=0
    rng=np.random.default_rng(seeds[0])
    for seed in seeds:
        obs,_=env.reset(seed=seed)
        episode_start=len(observations)
        while True:
            target=expert_action(obs,env.stage,transport_height)
            observations.append(obs.copy()); actions.append(target)
            action=target
            if model is not None and rng.random()>teacher_probability:
                action,_=model.predict(obs,deterministic=True)
            obs,_,t,tr,info=env.step(action)
            if t or tr:
                successes+=info["is_success"]
                if model is None and not info["is_success"]:
                    del observations[episode_start:]
                    del actions[episode_start:]
                break
    return np.asarray(observations),np.asarray(actions),int(successes)


def fit(model, observations, actions, epochs, seed):
    rng=np.random.default_rng(seed)
    x=torch.as_tensor(observations,device=model.device)
    y=torch.as_tensor(actions,device=model.device)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-3)
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        permutation=rng.permutation(len(x))
        for start in range(0,len(x),256):
            ids=permutation[start:start+256]
            mean=model.policy.get_distribution(x[ids]).distribution.mean
            loss=(mean-y[ids]).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.)
            optimizer.step()
    model.policy.set_training_mode(False)
    with torch.no_grad():
        model.policy.log_std.fill_(-2.5)
    return float(loss.detach().cpu())


def evaluate(model, env, seeds):
    results=[]
    for seed in seeds:
        obs,_=env.reset(seed=seed)
        while True:
            action,_=model.predict(obs,deterministic=True)
            obs,_,t,tr,info=env.step(action)
            if t or tr:
                results.append({"seed":seed,**info}); break
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["lift","place"],default="lift")
    parser.add_argument("--output",type=Path,default=Path("outputs/pick_place_assisted"))
    parser.add_argument("--demonstrations",type=int,default=30)
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--dagger-rounds",type=int,default=3)
    parser.add_argument("--ppo-steps",type=int,default=10240)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--start-near-cube",action="store_true")
    parser.add_argument("--domain-randomization",action="store_true")
    parser.add_argument("--transport-height",type=float,default=.21)
    parser.add_argument("--normalize-inputs",action="store_true")
    parser.add_argument("--obstacle-aware",action="store_true")
    args=parser.parse_args()
    if not .18<=args.transport_height<=.4:
        parser.error("transport-height must be between .18 and .4 metres")
    if args.demonstrations<1 or args.epochs<1 or args.dagger_rounds<0 or args.ppo_steps<0:
        parser.error("Invalid training budget")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder must be new or empty to preserve existing runs")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    env_class=ObstaclePickPlaceEnv if args.obstacle_aware else PickPlaceEnv
    env=env_class(stage=args.stage,randomize=True,start_near_cube=args.start_near_cube,domain_randomization=args.domain_randomization)
    validation=env_class(stage=args.stage,randomize=True,start_near_cube=args.start_near_cube,domain_randomization=args.domain_randomization)
    try:
        x,y,successes=collect(env,list(range(args.seed,args.seed+args.demonstrations)),transport_height=args.transport_height)
        if successes==0:
            raise RuntimeError("Teacher failed: no successful demonstrations to train on")
        kwargs={"net_arch":[256,256]}
        if args.normalize_inputs:
            kwargs.update(features_extractor_class=StandardizedObservation,
                features_extractor_kwargs={"mean":x.mean(axis=0).tolist(),"scale":np.maximum(x.std(axis=0),.01).tolist()})
        model=PPO("MlpPolicy",env,seed=args.seed,device="cpu",n_steps=1024,batch_size=64,
            learning_rate=3e-5,ent_coef=0.,target_kl=.01,policy_kwargs=kwargs,verbose=1)
        print(f"Teacher: {successes}/{args.demonstrations} {args.stage} successes; {len(x)} samples",flush=True)
        if successes==0:
            raise RuntimeError("Teacher failed: refusing to train on an unsuccessful demonstration setup")
        history=[]
        best_imitation_rate=-1.
        for iteration in range(args.dagger_rounds+1):
            loss=fit(model,x,y,args.epochs,args.seed+iteration)
            rows=evaluate(model,validation,list(range(10000,10010)))
            rate=sum(r["is_success"] for r in rows)/len(rows)
            history.append({"round":iteration,"samples":len(x),"loss":loss,"validation_success_rate":rate})
            print(f"Imitation round {iteration}: {rate:.1%} {args.stage} success",flush=True)
            model.save(str(args.output/f"imitation_round_{iteration}"))
            if rate>best_imitation_rate:
                best_imitation_rate=rate
                model.save(str(args.output/"best_imitation_model"))
            if iteration<args.dagger_rounds:
                new_x,new_y,_=collect(env,list(range(100+iteration*args.demonstrations,100+(iteration+1)*args.demonstrations)),model,
                    teacher_probability=.5/(iteration+1),transport_height=args.transport_height)
                x=np.concatenate([x,new_x]); y=np.concatenate([y,new_y])
        np.savez_compressed(args.output/"demonstrations.npz",observations=x,actions=y)
        model=PPO.load(str(args.output/"best_imitation_model.zip"),env=env,device="cpu")
        model.save(str(args.output/"imitation_model"))
        model.set_logger(configure(str(args.output),["stdout","csv"]))
        callback=PickPlaceValidation(validation,args.output,args.stage)
        callback.init_callback(model); callback.validate()
        if args.ppo_steps:
            model.learn(total_timesteps=args.ppo_steps,callback=callback)
            callback.validate()
        model.save(str(args.output/"final_model"))
        report={"method":"behavior_cloning_dagger_then_ppo" if args.ppo_steps else "behavior_cloning_dagger","reward_version":3,
            "config":{**vars(args),"output":str(args.output)},"teacher_successes":successes,
            "imitation_history":history,"note":"Validation targets are development data, not independent final results."}
        (args.output/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"Saved policies under {args.output}. Evaluate best_model.zip on fresh seeds without the teacher.")
    finally:
        env.close(); validation.close()


if __name__=="__main__":
    main()
