import numpy as np
import matplotlib.pyplot as plt

seeds = [0, 1, 2]
modes = ["basic", "relvec"]
colors = {"basic": "tab:orange", "relvec": "tab:blue"}

fig, ax = plt.subplots(figsize=(8, 5))
summary_lines = []

for mode in modes:
    all_t = []
    all_success = []
    for s in seeds:
        d = np.load(f"models/runs/ppo_{mode}_s{s}/evaluations.npz")
        all_t.append(d["timesteps"])
        if "successes" in d.files:
            all_success.append(d["successes"].mean(axis=1) * 100)
        else:
            all_success.append(None)

    t = all_t[0]
    if all_success[0] is not None:
        arr = np.stack(all_success)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(t, mean, label=f"{mode} (mean of 3 seeds)", color=colors[mode])
        ax.fill_between(t, mean - std, mean + std, alpha=0.2, color=colors[mode])
        summary_lines.append(f"{mode}: final success rate = {mean[-1]:.1f}% +/- {std[-1]:.1f}% (n=3 seeds)")

ax.set_xlabel("Training timesteps")
ax.set_ylabel("Success rate (%)")
ax.set_title("Observation ablation: basic vs relative-vector (PPO, 3 seeds each)")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("ablation_comparison.png", dpi=150)
print("Saved ablation_comparison.png")
print()
for line in summary_lines:
    print(line)
