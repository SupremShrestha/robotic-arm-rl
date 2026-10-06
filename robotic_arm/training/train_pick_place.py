"""Train a separate PPO policy for physical Panda manipulation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from robotic_arm.utils.output_paths import prepare_output
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from robotic_arm.environments.pick_place_env import PickPlaceEnv
from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv


class PickPlaceValidation(BaseCallback):
    """Select by stage success, then held height (lift) or placement error."""
    def __init__(self, env, output, stage):
        super().__init__()
        self.env,self.output,self.stage=env,output,stage
        self.best=(-1.,-float("inf"))

    def _on_step(self):
        if self.n_calls % 10000 == 0:
            self.validate()
        return True

    def validate(self):
        rows=[]
        for seed in range(10000,10010):
            obs,_=self.env.reset(seed=seed)
            total=0
            while True:
                action,_=self.model.predict(obs,deterministic=True)
                obs,reward,t,tr,info=self.env.step(action)
                total+=reward
                if t or tr: break
            rows.append({**info,"reward":total})
        rate=float(np.mean([r["is_success"] for r in rows]))
        height=float(np.mean([r["max_held_height_m"] for r in rows]))
        distance=float(np.mean([r["reach_distance_m"] if self.stage=="reach" else r["placement_error_m"] for r in rows]))
        score=(rate,height if self.stage=="lift" else -distance)
        record={"timesteps":self.num_timesteps,"success_rate":rate,"mean_max_held_height_m":height,
            "mean_reward":float(np.mean([r["reward"] for r in rows])),"reward_version":3}
        with (self.output/"validation.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
        self.logger.record("eval/success_rate",rate)
        self.logger.record("eval/max_held_height_m",height)
        print(f"Validation: success={rate:.1%}; max held height={height:.3f} m")
        if score>self.best:
            self.best=score
            self.model.save(str(self.output/"best_model"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps",type=int,default=1000000)
    parser.add_argument("--learning-rate",type=float)
    parser.add_argument("--target-kl",type=float)
    parser.add_argument("--stage",choices=["reach","lift","place"],default="place")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--fixed-positions",action="store_true")
    parser.add_argument("--domain-randomization",action="store_true")
    parser.add_argument("--scene-randomization",action="store_true")
    parser.add_argument("--random-layouts",action="store_true",help="Train on opposite table-region source/destination pairs")
    parser.add_argument("--obstacle-aware",action="store_true")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--start-near-cube",action="store_true",help="Reset near the cube to practice grasping; no automatic grasp")
    parser.add_argument("--reset-exploration",action="store_true",help="Restore action exploration when resuming a stalled policy")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if any(value is not None and (not np.isfinite(value) or value<=0) for value in (args.learning_rate,args.target_kl)):
        parser.error("learning-rate and target-kl must be positive and finite")
    if args.timesteps<1:
        parser.error("timesteps must be positive")
    output=args.output or Path("outputs/pick_place_training")/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        output=prepare_output(output)
    except ValueError as error:
        parser.error(str(error))
    config={**vars(args),"output":str(output),"controller":"ppo_panda", "randomize":not args.fixed_positions,"reward_version":3}
    (output/"config.json").write_text(json.dumps(config,indent=2,default=str),encoding="utf-8")
    overrides={key:value for key,value in {"learning_rate":args.learning_rate,"target_kl":args.target_kl}.items() if value is not None}
    resumed=PPO.load(str(args.resume),device="cpu",custom_objects=overrides) if args.resume else None
    from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
    env_class=OrientedPickPlaceEnv if resumed is not None and getattr(resumed,"environment_profile",None)=="quarter_turn_v2" else (ObstaclePickPlaceEnv if args.obstacle_aware else PickPlaceEnv)
    if resumed is not None and getattr(resumed,"environment_profile",None)=="yaw_aligned_v3":
        from robotic_arm.environments.yaw_aligned_pick_place_env import YawAlignedPickPlaceEnv
        env_class=YawAlignedPickPlaceEnv
    train=Monitor(env_class(stage=args.stage,randomize=not args.fixed_positions,start_near_cube=args.start_near_cube,domain_randomization=args.domain_randomization,scene_randomization=args.scene_randomization,cross_workspace=args.random_layouts),str(output/"train"))
    validation=Monitor(env_class(stage=args.stage,randomize=not args.fixed_positions,start_near_cube=args.start_near_cube,domain_randomization=args.domain_randomization,scene_randomization=args.scene_randomization,cross_workspace=args.random_layouts))
    validation.reset(seed=10000)
    try:
        if args.resume:
            model=resumed
            model.set_env(train)
            model.set_random_seed(args.seed)
            # Saved reaching statistics must not appear as new lifting successes.
            model.ep_info_buffer = None
            model.ep_success_buffer = None
            if args.reset_exploration:
                import torch
                with torch.no_grad():
                    model.policy.log_std.fill_(-.5)
        else:
            model=PPO("MlpPolicy",train,seed=args.seed,device="cpu",verbose=1,
                n_steps=1024,batch_size=64,learning_rate=args.learning_rate or 3e-4,target_kl=args.target_kl,policy_kwargs={"net_arch":[256,256]})
        model.set_logger(configure(str(output),["stdout","csv"]))
        callback=PickPlaceValidation(validation,output,args.stage)
        callback.init_callback(model)
        callback.validate()
        try:
            model.learn(total_timesteps=args.timesteps,callback=callback,reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            model.save(str(output/"interrupted_model"))
            print("Interrupted; checkpoint saved.")
            return
        callback.validate()
        model.save(str(output/"final_model"))
        print(f"Saved {output/'final_model.zip'}; evaluate on unseen seeds before claiming success.")
    finally:
        train.close()
        validation.close()


if __name__=="__main__":
    main()
