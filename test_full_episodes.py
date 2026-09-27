from environment.reach_env import ReachEnv
import numpy as np

env = ReachEnv(render_mode=None)

num_episodes = 10
success_count = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0.0
    steps_taken = 0

    while True:
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps_taken += 1

        if terminated or truncated:
            break

    if info.get("is_success", False):
        success_count += 1

    print(f"Episode {episode + 1:>2}: steps={steps_taken:>3}, "
          f"total_reward={episode_reward:>8.2f}, "
          f"final_distance={info['distance']:.4f}, "
          f"terminated={terminated}, truncated={truncated}")

print(f"\nRan {num_episodes} episodes successfully, no crashes.")
print(f"Successes with random actions: {success_count}/{num_episodes} "
      f"(expect 0 or very few - random actions rarely find the target)")

env.close()