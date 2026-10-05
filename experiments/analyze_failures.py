"""Summarize observed failures without treating correlations as proven causes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def analyze(input_path, output):
    with Path(input_path).open(newline="",encoding="utf-8") as f:
        raw=list(csv.DictReader(f))
    if not raw:
        raise ValueError("Input report has no episodes")
    rows=[{**r,"success":r["success"]=="True","ground_collision":r["ground_collision"]=="True",
           "distance_m":float(r["distance_m"]),"target_x":float(r["target_x"]),
           "target_y":float(r["target_y"]),"target_z":float(r["target_z"])} for r in raw]
    failed=[r for r in rows if not r["success"]]
    summary={"episodes":len(rows),"failures":len(failed),"failed_with_ground_contact":sum(r["ground_collision"] for r in failed),
             "by_height":{},"diagnostics":{},"interpretation":"These are observed associations, not proof of a causal explanation."}
    for label,select in [("z_below_0.5",lambda r:r["target_z"]<0.5),("z_at_least_0.5",lambda r:r["target_z"]>=0.5)]:
        group=[r for r in rows if select(r)]
        summary["by_height"][label]={"episodes":len(group),"success_rate":float(np.mean([r["success"] for r in group])) if group else None}
    for key in ["action_saturation_fraction","joint_limit_fraction","stationary_distance_fraction"]:
        if key in rows[0]:
            summary["diagnostics"][key]={}
            for label,group in [("successful",[r for r in rows if r["success"]]),("failed",failed)]:
                summary["diagnostics"][key][label]=float(np.mean([float(r[key]) for r in group])) if group else None
    worst=sorted(failed,key=lambda r:r["distance_m"],reverse=True)[:10]
    summary["worst_episode_seeds"]=[int(r["seed"]) for r in worst]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    text=["# PPO failure analysis", "",f"Observed {len(failed)} failures in {len(rows)} episodes; {summary['failed_with_ground_contact']} failed episodes had ground contact.",
          "", "| Seed | Final distance (m) | Target x, y, z (m) |", "|---|---|---|"]
    for r in worst:
        text.append(f"| {r['seed']} | {r['distance_m']:.4f} | {r['target_x']:.3f}, {r['target_y']:.3f}, {r['target_z']:.3f} |")
    text += ["", "Diagnostics compare saturation, proximity to joint limits, and stationary distance between successful and failed episodes. A high diagnostic value is evidence to investigate; it is not proof of why a policy failed.",
             "", "Replay a reported seed using `python evaluate.py --model models/portfolio/legacy_best_model.zip --episodes 1 --seed SEED --render`.",
             "", "IK succeeds on these benchmark targets. PPO failures therefore cannot simply be dismissed as unreachable targets. Both use the same physics and target distribution, but PPO must learn joint coordination from data. Further controlled experiments are required to attribute improvement to a particular reward term."]
    (output/"README.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplot(projection="3d")
    for success,label,color in [(True,"Reached","seagreen"),(False,"Failed","coral")]:
        group=[r for r in rows if r["success"]==success]
        ax.scatter([r["target_x"] for r in group],[r["target_y"] for r in group],[r["target_z"] for r in group],c=color,label=label,s=18)
    ax.set(xlabel="Target x (m)",ylabel="Target y (m)",zlabel="Target z (m)",title="PPO failures by target position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"targets.png",dpi=150)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output",default="reports/failure_analysis")
    args=parser.parse_args()
    analyze(args.input,args.output)
