import numpy as np
import matplotlib.pyplot as plt

d = np.load("models/eval_logs/evaluations.npz")
t = d["timesteps"]
reward = d["results"].mean(axis=1)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(t, reward, color="tab:blue", label="Eval reward")
ax1.set_xlabel("Training timesteps")
ax1.set_ylabel("Mean eval reward", color="tab:blue")

if "successes" in d.files:
    ax2 = ax1.twinx()
    ax2.plot(t, d["successes"].mean(axis=1) * 100, color="tab:green", label="Success %")
    ax2.set_ylabel("Success rate (%)", color="tab:green")

plt.title("PPO on random-target reaching (KUKA iiwa)")
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print("Saved training_curve.png")
