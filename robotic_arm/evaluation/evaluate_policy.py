"""Repeatable evaluation and simulation for PPO, random, and IK controllers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from robotic_arm.utils.output_paths import prepare_output
import time
import numpy as np
from stable_baselines3 import PPO
from robotic_arm.environments.reaching_env import ReachEnv, SimulationDisconnected


def evaluate(controller="ppo", model_path="models/reaching_policy.zip", episodes=20,
             seed=0, render=False, output=None, reward_mode="legacy"):
    if controller not in ("ppo", "ik", "random"):
        raise ValueError("controller must be ppo, ik, or random")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if output:
        prepare_output(output)
    model = None
    if controller == "ppo":
        path = Path(model_path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {path}. Train first or supply --model.")
        model = PPO.load(str(path), device="cpu")
    env = ReachEnv(render_mode="human" if render else None, reward_mode=reward_mode)
    rows = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs, info = env.reset(seed=episode_seed)
            rng = np.random.default_rng(episode_seed)
            total_reward, collision = 0.0, False
            initial_distance=info["distance"]
            closest_distance=initial_distance
            saturated_steps=limit_steps=stationary_steps=0
            previous_distance=initial_distance
            while True:
                if controller == "ppo":
                    action, _ = model.predict(obs, deterministic=True)
                elif controller == "ik":
                    action = env.ik_action()
                else:
                    action = rng.uniform(-1, 1, 7).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                closest_distance=min(closest_distance,info["distance"])
                saturated_steps += bool(np.any(np.abs(action)>=0.99))
                limit_steps += bool(np.any(np.minimum(obs[:7]-env.joint_lower,env.joint_upper-obs[:7])<0.05))
                stationary_steps += abs(info["distance"]-previous_distance)<0.0001
                previous_distance=info["distance"]
                collision = collision or info["ground_collision"]
                if render:
                    time.sleep(env.action_repeat / 240)
                if terminated or truncated:
                    break
            rows.append({"seed": episode_seed, "target_x": info["target"][0],
                         "target_y": info["target"][1], "target_z": info["target"][2],
                         "success": info["is_success"], "distance_m": info["distance"],
                         "steps": info["steps"], "reward": total_reward,
                         "ground_collision": collision, "timeout": truncated,
                         "initial_distance_m":initial_distance,"closest_distance_m":closest_distance,
                         "action_saturation_fraction":saturated_steps/info["steps"],
                         "joint_limit_fraction":limit_steps/info["steps"],
                         "stationary_distance_fraction":stationary_steps/info["steps"]})
            if render:
                print(f"Episode {episode + 1}: success={info['is_success']}, distance={info['distance']:.4f} m")
                time.sleep(0.4)
    except SimulationDisconnected:
        if not render:
            raise
        print(f"Simulation window closed or disconnected. Stopped after {len(rows)} completed episodes.")
        print("Evaluation incomplete; no benchmark report saved.")
        return None, rows
    finally:
        env.close()
    successes = sum(r["success"] for r in rows)
    rate = successes / episodes
    z = 1.96
    center = (rate + z*z / (2*episodes)) / (1 + z*z / episodes)
    margin = z * np.sqrt(rate*(1-rate)/episodes + z*z/(4*episodes*episodes)) / (1+z*z/episodes)
    summary = {"controller": controller, "episodes": episodes, "seed_start": seed,
               "successes": successes, "success_rate": rate,
               "success_rate_95pct_wilson": [float(center-margin), float(center+margin)],
               "mean_distance_m": float(np.mean([r["distance_m"] for r in rows])),
               "best_distance_m": min(r["distance_m"] for r in rows),
               "mean_steps": float(np.mean([r["steps"] for r in rows])),
               "ground_collision_episodes": sum(r["ground_collision"] for r in rows),
               "success_threshold_m": env.success_threshold, "reward_mode": reward_mode}
    if model is not None:
        summary["model"] = str(path)
        summary["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Success rate: {successes}/{episodes} ({rate:.1%})")
    print(f"Average final distance: {summary['mean_distance_m']:.4f} m")
    print(f"Best final distance: {summary['best_distance_m']:.4f} m")
    print(f"Ground-contact episodes (excluding fixed base): {summary['ground_collision_episodes']}")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(1, episodes+1), [r["distance_m"] for r in rows],
               color=["seagreen" if r["success"] else "coral" for r in rows])
        ax.axhline(env.success_threshold, color="black", linestyle="--", label="5 cm threshold")
        ax.set(xlabel="Episode", ylabel="Final distance (m)", title=f"{controller.upper()}: {rate:.1%} success")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "distances.png", dpi=150)
        plt.close(fig)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["ppo", "ik", "random"], default="ppo")
    parser.add_argument("--model", default="models/reaching_policy.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--reward-mode", choices=["legacy", "shaped"], default="legacy")
    args = parser.parse_args()
    try:
        evaluate(args.controller, args.model, args.episodes, args.seed, args.render, args.output, args.reward_mode)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Simulation/evaluation interrupted; environment closed.")


if __name__ == "__main__":
    main()
