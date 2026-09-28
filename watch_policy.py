import time
from stable_baselines3 import PPO
from environment.reach_env import ReachEnv

env = ReachEnv(render_mode="human")
model = PPO.load("models/best_model/best_model")

for episode in range(5):
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1 / 60)
        if terminated or truncated:
            print(f"Episode {episode + 1}: success={info['is_success']}, distance={info['distance']:.3f}")
            time.sleep(1)
            break

env.close()
