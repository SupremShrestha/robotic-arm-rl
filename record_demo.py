"""Record actual simulated motion as a GIF for your portfolio."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from environment.reach_env import ReachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["ppo", "ik"], default="ik")
    parser.add_argument("--model", default="models/portfolio/legacy_best_model.zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/demo.gif"))
    args = parser.parse_args()
    if not 1 <= args.episodes <= 5 or args.seed < 0:
        parser.error("Use 1-5 episodes and a nonnegative seed")
    model = PPO.load(args.model, device="cpu") if args.controller == "ppo" else None
    env = ReachEnv(render_mode="rgb_array")
    frames = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            for step in range(env.max_episode_steps):
                action = model.predict(obs, deterministic=True)[0] if model is not None else env.ik_action()
                obs, _, terminated, truncated, info = env.step(action)
                if step % 4 == 0 or terminated or truncated:
                    frame = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 40), fill="white")
                    draw.text((10, 8), f"{args.controller.upper()} | target {episode+1} | distance {info['distance']:.3f} m | success {info['is_success']}", fill="black")
                    frames.append(frame)
                if terminated or truncated:
                    frames.extend([frames[-1].copy() for _ in range(6)])
                    break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=83, loop=0)
    print(f"Saved actual {args.controller.upper()} simulation to {args.output}")


if __name__ == "__main__":
    main()
