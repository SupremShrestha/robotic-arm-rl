"""Record learned pick-and-place in a persistent three-object scene as a GIF."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from stable_baselines3 import PPO
from robotic_arm.environments.persistent_pick_place_env import PersistentPickPlaceEnv
from robotic_arm.environments.reaching_env import SimulationDisconnected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path("models/panda_simulation_policy.zip"))
    parser.add_argument("--seed",type=int,default=2000)
    parser.add_argument("--output",type=Path,default=Path("outputs/media/persistent_pick_place.gif"))
    args=parser.parse_args()
    if args.output.exists():parser.error("Output already exists; choose a new path")
    if not args.model.is_file():parser.error("Model does not exist")
    model=PPO.load(args.model,device="cpu")
    if getattr(model,"environment_profile",None) not in ("quarter_turn_v2","yaw_aligned_v3"):parser.error("Use a compatible fixed/yaw-aligned model")
    env=PersistentPickPlaceEnv(render_mode="rgb_array")
    frames=[];count=0
    try:
        obs,_=env.reset(seed=args.seed)
        for task in range(3):
            while True:
                action,_=model.predict(obs,deterministic=True)
                obs,_,t,tr,info=env.step(action)
                if env.steps%6==0 or t or tr:
                    frame=Image.fromarray(env.render())
                    draw=ImageDraw.Draw(frame);draw.rectangle((0,0,320,24),fill="white")
                    draw.text((5,5),f"Task {task+1}/3 | error {info['placement_error_m']:.3f} m | success {info['is_success']}",fill="black")
                    frames.append(frame)
                if t or tr:break
            count+=int(info["is_success"])
            frames.extend([frames[-1].copy() for _ in range(4)])
            if not info["is_success"]:break
            if task<2:obs,_=env.next_task()
    except (KeyboardInterrupt,SimulationDisconnected):
        print("Recording interrupted; no complete GIF saved.");return
    finally:env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=300,loop=0)
    print(f"Saved {args.output}; {count}/3 completed tasks")


if __name__=="__main__":main()
