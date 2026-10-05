from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

"""Standalone learning demo; run explicitly from the project root."""

def main():
    from environment.reach_env import ReachEnv

    env = ReachEnv(render_mode=None)

    print("Sampling 5 resets to confirm target randomization:\n")
    for i in range(5):
        obs, info = env.reset()
        print(f"Reset {i+1}: target position = {env.target_position}")

    env.close()


if __name__ == "__main__":
    main()
