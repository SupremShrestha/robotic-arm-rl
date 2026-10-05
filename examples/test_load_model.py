from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

"""Standalone learning demo; run explicitly from the project root."""

def main():
    from stable_baselines3 import PPO

    model = PPO.load("models/ppo_reach_smoketest")
    print("Model loaded successfully.")
    print(f"Policy: {model.policy}")


if __name__ == "__main__":
    main()
