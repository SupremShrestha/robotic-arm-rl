from stable_baselines3 import PPO

model = PPO.load("models/ppo_reach_smoketest")
print("Model loaded successfully.")
print(f"Policy: {model.policy}")