from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

"""Standalone learning demo; run explicitly from the project root."""

def main():
    from environment.reach_env import ReachEnv
    import numpy as np

    env = ReachEnv(render_mode=None)
    obs, info = env.reset()

    print("Initial joint positions:", obs[0:7])
    print("Initial end-effector position:", obs[14:17])

    # Apply the SAME action repeatedly for 50 steps and watch the arm move
    action = np.array([1.0, -1.0, 0.5, -0.5, 0.3, 0.8, 0.0], dtype=np.float32)

    for i in range(50):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print(f"\nAfter 50 steps of the same action:")
    print("Joint positions:", obs[0:7])
    print("End-effector position:", obs[14:17])

    # Sanity check: with max_action_delta=0.05, action=1.0 for 50 steps
    # should move joint 0 close to (but not exceeding, due to PD lag) 50 * 0.05 = 2.5 rad
    print(f"\nExpected joint 0 to have moved toward ~2.5 rad (its limit is 2.967)")

    env.close()


if __name__ == "__main__":
    main()
