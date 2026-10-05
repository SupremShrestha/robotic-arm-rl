"""Train/resume PPO with fixed validation seeds and isolated run artifacts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.reach_env import ReachEnv


class ValidationCallback(BaseCallback):
    def __init__(self, directory, frequency, episodes, reward_mode):
        super().__init__()
        self.directory, self.frequency, self.episodes = directory, frequency, episodes
        self.env = ReachEnv(reward_mode=reward_mode)
        self.best = (-1.0, -np.inf)

    def _on_step(self):
        if self.n_calls % self.frequency:
            return True
        return self.validate()

    def validate(self):
        successes, rewards, distances = [], [], []
        for episode in range(self.episodes):
            obs, info = self.env.reset(seed=10000 + episode)
            total = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            successes.append(info["is_success"])
            rewards.append(total)
            distances.append(info["distance"])
        rate, distance = float(np.mean(successes)), float(np.mean(distances))
        score = (rate, -distance)
        record = {"timesteps": self.num_timesteps, "success_rate": rate,
                  "mean_distance_m": distance, "mean_reward": float(np.mean(rewards))}
        with (self.directory / "validation.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Validation: {rate:.1%} success; mean distance {distance:.4f} m")
        self.logger.record("eval/success_rate", rate)
        self.logger.record("eval/mean_distance", distance)
        if score > self.best:
            self.best = score
            self.model.save(str(self.directory / "best_model"))
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--reward-mode", choices=["legacy", "shaped"], default="legacy")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if min(args.timesteps, args.eval_freq, args.eval_episodes) < 1:
        parser.error("timesteps, eval-freq and eval-episodes must be positive")
    directory = args.run_dir or Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty run directory to avoid overwriting results")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: Monitor(ReachEnv(reward_mode=args.reward_mode))])
    env.seed(args.seed)
    callback = ValidationCallback(directory, args.eval_freq, args.eval_episodes, args.reward_mode)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"run_dir": str(directory), "validation_seed_start": 10000,
                   "success_threshold_m": 0.05, "torch_version": torch.__version__})
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = None
    try:
        if args.resume:
            model = PPO.load(str(args.resume), device="cpu", seed=args.seed)
            if model.observation_space.shape != env.observation_space.shape or model.action_space != env.action_space:
                raise ValueError("Checkpoint has an incompatible observation/action layout")
            # Legacy checkpoints declare +/-10 bounds although velocities may exceed them.
            # Only metadata changes; the 20 raw observation values keep their meaning.
            model.observation_space = env.observation_space
            model.set_env(env)
        else:
            model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device="cpu",
                        n_steps=2048, batch_size=64, n_epochs=10)
        model.set_logger(configure(str(directory), ["stdout", "csv"]))
        # Evaluate the starting model, so a worse update cannot replace it as best.
        callback.init_callback(model)
        callback.num_timesteps = model.num_timesteps
        callback.validate()
        model.learn(args.timesteps, callback=callback, reset_num_timesteps=not bool(args.resume))
        callback.num_timesteps = model.num_timesteps
        callback.validate()
        model.save(str(directory / "final_model"))
        print(f"Completed. Best checkpoint: {directory / 'best_model.zip'}")
    except KeyboardInterrupt:
        if model is not None:
            model.save(str(directory / "interrupted_model"))
        print("Interrupted; checkpoint saved when model was available.")
    finally:
        callback.env.close()
        env.close()


if __name__ == "__main__":
    main()
