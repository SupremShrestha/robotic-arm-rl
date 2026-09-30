import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from environment.reach_avoid_env import ReachAvoidEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", choices=["basic", "relvec"], default="relvec")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    torch.set_num_threads(1)

    thresh_tag = f"t{int(args.threshold * 1000)}"
    name = f"avoid_{args.obs}_{thresh_tag}_s{args.seed}"
    run_dir = os.path.join("models", "runs", name)
    os.makedirs(run_dir, exist_ok=True)

    env_kwargs = {"obs_mode": args.obs, "success_threshold": args.threshold}
    vec_env = make_vec_env(ReachAvoidEnv, n_envs=1, seed=args.seed, env_kwargs=env_kwargs)
    eval_env = make_vec_env(ReachAvoidEnv, n_envs=1, seed=args.seed + 1000, env_kwargs=env_kwargs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )

    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=2048,
                batch_size=64, n_epochs=10, seed=args.seed)

    print(f"Run {name}: {args.steps} steps, threshold={args.threshold}")
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Training complete. Best model: {run_dir}/best_model.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
