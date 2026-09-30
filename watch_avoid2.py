import time

from stable_baselines3 import PPO

from environment.reach_avoid2_env import ReachAvoid2Env

env = ReachAvoid2Env(render_mode="human")
model = PPO.load("models/runs/avoid2_relvec_s0/best_model")

for episode in range(5):
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1 / 60)
        if terminated or truncated:
            outcome = "SUCCESS" if info["is_success"] else ("COLLISION" if info["is_collision"] else "timeout")
            print(f"Episode {episode + 1}: {outcome}, distance={info['distance']:.3f}")
            time.sleep(1)
            break

env.close()
