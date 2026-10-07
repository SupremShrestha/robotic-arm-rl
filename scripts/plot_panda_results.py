"""Create report figures from archived Panda experiments; no training required."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = []


def read_json(relative):
    path = ROOT / relative
    SOURCES.append(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(relative):
    path = ROOT / relative
    SOURCES.append(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(relative):
    path = ROOT / relative
    SOURCES.append(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(fig, directory, name, note):
    fig.text(.5, .012, note, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(directory / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(directory / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New or empty figure directory")
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / "panda_figures" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory to preserve earlier figures.")

    # Read all inputs before creating output artifacts.
    training = read_json("results/learned_pick_place/training/training_report.json")
    ppo = read_jsonl("results/shared_simulation_policy/ppo_training/validation.jsonl")
    stress = read_json("results/shared_simulation_policy/stress/summary.json")
    diagonal = read_rows("results/shared_simulation_policy/diagonal/episodes.csv")
    distances = read_json("results/shared_simulation_policy/distances/summary.json")
    paired = read_json("results/shared_simulation_policy/ppo_comparison.json")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .2,
    })
    generated = []

    history = training["imitation_history"]
    rounds = [r["round"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rounds, [100*r["validation_success_rate"] for r in history], "o-", color="#237849")
    axes[0].set(xlabel="Imitation / DAgger round", ylabel="Validation success (%)",
                ylim=(-5, 105), xticks=rounds, title="Nominal Panda learning: validation")
    axes[1].plot(rounds, [r["loss"] for r in history], "o-", color="#315da8")
    axes[1].set(xlabel="Imitation / DAgger round", ylabel="Last training minibatch action MSE",
                xticks=rounds, title="Recorded action-fitting loss")
    name = "panda_imitation_learning.png"
    save(fig, output, name, "Earlier nominal model; development validation. Loss is not held-out error or an epoch-average loss.")
    generated.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = [r["timesteps"] for r in ppo]
    axes[0].plot(steps, [100*r["success_rate"] for r in ppo], "o-", color="#237849")
    axes[0].set(xlabel="Additional PPO environment steps", ylabel="Validation success (%)",
                ylim=(0, 105), title="Intermediate Panda PPO continuation")
    axes[1].plot(steps, [r["mean_reward"] for r in ppo], "o-", color="#315da8")
    axes[1].set(xlabel="Additional PPO environment steps", ylabel="Mean episode reward",
                title="Recorded validation reward")
    name = "panda_ppo_training_curve.png"
    save(fig, output, name, "Single PPO seed; three recorded evaluations, joined for readability. This is not final shared-model training.")
    generated.append(name)

    cases = stress["cases"]
    names = list(cases)
    labels = [n.replace("_", " ") for n in names]
    rates = [100*cases[n]["successes"]/cases[n]["episodes"] for n in names]
    errors = [100*cases[n]["mean_placement_error_m"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    y = np.arange(len(names))
    axes[0].barh(y, rates, color="#237849")
    axes[0].set(yticks=y, yticklabels=labels, xlim=(0, 115), xlabel="Observed success (%)",
                title="Final shared Panda: stress outcomes")
    for i, n in enumerate(names):
        axes[0].text(rates[i]+1, i, f'{cases[n]["successes"]}/{cases[n]["episodes"]}', va="center", fontsize=9)
    axes[1].barh(y, errors, color="#315da8")
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Mean final XY error (cm)",
                title="Placement precision by condition")
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    name = "panda_robustness_comparison.png"
    save(fig, output, name, "17 conditions, two paired target seeds per condition; finite coverage, not 34 independent random scene configurations.")
    generated.append(name)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    errors = [100*float(row["placement_error_m"]) for row in diagonal]
    colours = ["#237849" if row["is_success"].lower()=="true" else "#b43a38" for row in diagonal]
    ax.bar(np.arange(1, len(errors)+1), errors, color=colours)
    ax.axhline(5, color="#b43a38", linestyle="--", label="XY tolerance: 5 cm")
    ax.set(xlabel="Evaluation episode (seeds 56000 onward)", ylabel="Final XY placement error (cm)",
           xticks=np.arange(1, len(errors)+1), title="Shared Panda policy: fresh diagonal layouts")
    ax.legend()
    name = "panda_placement_errors.png"
    save(fig, output, name, "Success additionally requires prior grasp/lift, release, settling and ten valid steps; distance alone is insufficient.")
    generated.append(name)

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#237849", "#315da8", "#8b3b98"]
    for i, (source, goal) in enumerate(distances["task_positions"]):
        source, goal = np.asarray(source), np.asarray(goal)
        cm = 100*np.linalg.norm(goal-source)
        ax.scatter(*source, marker="s", s=100, color=colours[i])
        ax.scatter(*goal, marker="*", s=180, color=colours[i])
        ax.annotate("", xy=goal, xytext=source, arrowprops={"arrowstyle":"->", "color":colours[i], "lw":1.8})
        ax.plot([], [], color=colours[i], label=f"Task {i+1}: {cm:.0f} cm")
        ax.annotate(f"S{i+1}", source, xytext=(5, -14), textcoords="offset points")
        ax.annotate(f"D{i+1}", goal, xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Workspace x (m)", ylabel="Workspace y (m)", xlim=(.24, .61), ylim=(-.35, .35),
           title="Panda transfer layouts: 23, 40 and 65 cm")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=.95)
    name = "panda_task_layouts.png"
    save(fig, output, name, "Squares: sources; stars: destinations. Arrows show straight-line separation, not measured arm trajectories.")
    generated.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    def percentage(value):
        success, total = map(int, value.split("/"))
        return 100*success/total
    rates = [percentage(paired["imitation_stress"]), percentage(paired["ppo_stress"])]
    axes[0].bar(["Before PPO", "After PPO"], rates, color=["#237849", "#d49032"])
    axes[0].set(ylabel="Paired stress success (%)", ylim=(0, 105), title="Intermediate-model comparison")
    for i, value in enumerate([paired["imitation_stress"], paired["ppo_stress"]]):
        axes[0].text(i, rates[i]+2, value, ha="center")
    errors = [100*paired["imitation_diagonal_error_m"], 100*paired["ppo_diagonal_error_m"]]
    axes[1].bar(["Before PPO", "After PPO"], errors, color=["#237849", "#d49032"])
    axes[1].set(ylabel="Mean diagonal XY error (cm)", title="Both policies: 20/20 diagonal success")
    name = "panda_paired_ppo_comparison.png"
    save(fig, output, name, "10,240 additional PPO steps; same test seeds before/after. Final shared BC refinement is a separate checkpoint.")
    generated.append(name)

    manifest = {
        "purpose": "Report figures from archived Panda measurements; no new training or rollouts.",
        "figures": generated,
        "source_files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
        "limits": "Development curves are not final tests. Different task sets are not directly comparable ablations.",
    }
    (output/"figure_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(generated)} Panda figures and figure_sources.json.")


if __name__ == "__main__":
    main()
