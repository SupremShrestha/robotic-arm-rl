import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from environment.pick_place_env import PickPlaceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    args = parser.parse_args()

    torch.set_num_threads(1)

    name = f"pickplace_s{args.seed}"
    run_dir = os.path.join("models", "runs", name)
    os.makedirs(run_dir, exist_ok=True)

    vec_env = make_vec_env(PickPlaceEnv, n_envs=1, seed=args.seed)
    eval_env = make_vec_env(PickPlaceEnv, n_envs=1, seed=args.seed + 1000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        n_steps=2048, batch_size=64, n_epochs=10, seed=args.seed,
        ent_coef=0.01,
    )

    print(f"Run {name}: {args.steps} steps")
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Training complete. Best model: {run_dir}/best_model.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
