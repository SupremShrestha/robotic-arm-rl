"""Train one candidate across saved modern simulation demonstrations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from robotic_arm.training.learn_pick_place import fit
from robotic_arm.utils.output_paths import prepare_output


DATASETS=["results/scene_generalization/training/demonstrations.npz",
    "results/diagonal_pick_place/training/demonstrations.npz",
    "results/distance_pick_place/training/demonstrations.npz",
    "results/persistent_pick_place/training/demonstrations.npz"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path("models/panda_distance_policy.zip"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--yaw-refinement",action="store_true")
    parser.add_argument("--datasets",type=Path,nargs="+",default=[Path(p) for p in DATASETS])
    args=parser.parse_args()
    if args.epochs<1:parser.error("epochs must be positive")
    if not all(path.is_file() for path in [args.model,*args.datasets]):parser.error("Model/dataset path does not exist")
    try:prepare_output(args.output)
    except ValueError as error:parser.error(str(error))
    rng=np.random.default_rng(49000)
    xs,ys=[],[]
    for path in args.datasets:
        with np.load(path) as data:x=data["observations"];y=data["actions"]
        if x.shape[1:]!=(32,) or y.shape!=(len(x),4) or not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            parser.error("Datasets must have finite 32-value observations and four-value actions")
        ids=rng.choice(len(x),6000,replace=len(x)<6000)
        xs.append(x[ids]);ys.append(y[ids])
    x=np.concatenate(xs);y=np.concatenate(ys)
    torch.set_num_threads(1)
    model=PPO.load(args.model,device="cpu")
    if getattr(model,"environment_profile",None) not in ("quarter_turn_v2","yaw_aligned_v3"):parser.error("Use a quarter_turn_v2 model")
    demonstrations=[]
    if args.yaw_refinement:
        from robotic_arm.training.refine_obstacle_policy import AlignedFocusEnv
        from robotic_arm.training.learn_pick_place import collect
        cases=[{"size":size,"object_yaw":yaw} for size in (.04,.06) for yaw in (.4,.785398,1.2,-.785398)]
        cases += [{"size":.06,"obstacle_height":.24},{"object_shape":"cylinder","obstacle_height":.24}]
        for options in cases:
            env=AlignedFocusEnv(options)
            try:nx,ny,count=collect(env,list(range(52000,52005)),transport_height=.34)
            finally:env.close()
            demonstrations.append({"options":options,"successes":count,"episodes":5})
            print(f"Yaw refinement {options}: {count}/5",flush=True)
            if len(nx):
                x=np.concatenate([x,nx,nx]);y=np.concatenate([y,ny,ny])
        model.environment_profile="yaw_aligned_v3"
    np.savez_compressed(args.output/"demonstrations.npz",observations=x,actions=y)
    loss=fit(model,x,y,args.epochs,49000)
    model.save(args.output/"candidate")
    (args.output/"report.json").write_text(json.dumps({"method":"balanced behavior cloning across saved datasets; no PPO","source":str(args.model),"datasets":[str(p) for p in args.datasets],"samples_per_dataset":6000,"yaw_refinement_demonstrations":demonstrations,"seed":49000,"epochs":args.epochs,"loss":loss,"selection":"Candidate only; requires independent evaluation"},indent=2))
    print(f"Saved shared candidate; loss={loss:.6f}")


if __name__=="__main__":main()
