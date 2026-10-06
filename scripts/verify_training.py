"""Short fresh/resumed training integration check, without bundled weights."""
from pathlib import Path
import subprocess
import sys
import tempfile
from stable_baselines3 import PPO


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        common=["--timesteps","2048","--eval-freq","2048","--eval-episodes","2"]
        subprocess.run([sys.executable,"-m","robotic_arm","train",*common,"--run-dir",str(root/"fresh")],check=True)
        initial=PPO.load(root/"fresh/final_model.zip",device="cpu")
        subprocess.run([sys.executable,"-m","robotic_arm","train",*common,"--resume",str(root/"fresh/final_model.zip"),
                        "--run-dir",str(root/"resumed")],check=True)
        continued=PPO.load(root/"resumed/final_model.zip",device="cpu")
        if continued.num_timesteps <= initial.num_timesteps:
            raise AssertionError("Resume did not add training steps")
        for stage in ["fresh","resumed"]:
            if not (root/stage/"best_model.zip").is_file() or not (root/stage/"validation.jsonl").is_file():
                raise AssertionError("Missing training artifacts")
    print("Fresh training, final validation, checkpoint saving and resume passed.")


if __name__ == "__main__":
    main()
