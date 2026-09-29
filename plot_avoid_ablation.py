import numpy as np
import matplotlib.pyplot as plt

seeds = [0, 1, 2]

all_t = []
all_success = []
for s in seeds:
    d = np.load(f"models/runs/avoid_relvec_s{s}/evaluations.npz")
    all_t.append(d["timesteps"])
    if "successes" in d.files:
        all_success.append(d["successes"].mean(axis=1) * 100)

t = all_t[0]
arr = np.stack(all_success)
mean = arr.mean(axis=0)
std = arr.std(axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, mean, label="Success rate (mean of 3 seeds)", color="tab:green")
ax.fill_between(t, mean - std, mean + std, alpha=0.2, color="tab:green")
ax.set_xlabel("Training timesteps")
ax.set_ylabel("Success rate (%)")
ax.set_title("Obstacle avoidance: PPO with relative-vector observation (3 seeds)")
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig("obstacle_avoidance_curve.png", dpi=150)
print("Saved obstacle_avoidance_curve.png")
print(f"\nFinal success rate: {mean[-1]:.1f}% +/- {std[-1]:.1f}% (n=3 seeds)")
