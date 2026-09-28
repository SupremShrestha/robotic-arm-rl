import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment.reach_env import ReachEnv


def make_env():
    env = ReachEnv(render_mode=None)
    env = Monitor(env)
    return env


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/eval_logs", exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=1)
    eval_env = make_vec_env(make_env, n_envs=1)

    # Every 5,000 steps, pause and run 10 DETERMINISTIC episodes to measure
    # real (noise-free) performance, and keep the best checkpoint.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_model",
        log_path="models/eval_logs",
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    print("Starting training run (300,000 timesteps) with deterministic eval callback...\n")
    model.learn(total_timesteps=500_000, callback=eval_callback)

    model.save("models/ppo_reach_random_target")
    print("\nTraining complete.")
    print("Final model: models/ppo_reach_fixed_target.zip")
    print("Best deterministic-eval model: models/best_model/best_model.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
