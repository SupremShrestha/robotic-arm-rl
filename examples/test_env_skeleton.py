from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

"""Standalone learning demo; run explicitly from the project root."""

def main():
    from environment.reach_env import ReachEnv

    env = ReachEnv(render_mode=None)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, info = env.reset()
    print(f"\nreset() returned observation of shape: {obs.shape}")

    action = env.action_space.sample()
    print(f"Sampled random action: {action}")

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nstep() returned:")
    print(f"  observation shape: {obs.shape}")
    print(f"  reward: {reward}")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")

    env.close()
    print("\nEnvironment closed cleanly.")


if __name__ == "__main__":
    main()
