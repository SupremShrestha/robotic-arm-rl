"""Balanced grid of masses/friction, repeated across seeded source/destination scenes."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pick_place import run


def benchmark(output, scenes=6, seed=2026):
    if scenes < 1 or seed < 0:
        raise ValueError("Use positive scene count and nonnegative seed")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    rows=[]
    for index in range(scenes):
        source=rng.uniform([0.40,-0.18],[0.50,-0.08])
        destination=rng.uniform([0.40,0.10],[0.50,0.22])
        for mass in [0.04,0.08,0.15]:
            for friction in [0.5,1.0,2.0]:
                result=run(render=False,source=source,destination=destination,mass=mass,friction=friction,verbose=False)
                rows.append({"scene":index,"seed":seed,"source_x":float(source[0]),"source_y":float(source[1]),
                    "destination_x":float(destination[0]),"destination_y":float(destination[1]),"mass_kg":mass,
                    "friction":friction,"success":result["success"],"lifted":result.get("lifted",False),
                    "grasped_with_both_fingers":result.get("grasped_with_both_fingers",False),
                    "released":result.get("released",False),"resting":result.get("resting",False),
                    "placement_error_m":result.get("placement_error_m"),"ground_contact":result.get("ground_contact",False),
                    "failure":result.get("failure","")})
        print(f"Completed scene {index+1}/{scenes}",flush=True)
    summary={"trials":len(rows),"seed":seed,"geometric_scenes":scenes,"successes":sum(r["success"] for r in rows),
        "success_rate":float(np.mean([r["success"] for r in rows])),
        "ground_contact_trials":sum(r["ground_contact"] for r in rows),
        "scope":"5 cm cubes, 6 seeded geometric scenes by default; mass/friction grid, no obstacles or sensor noise.",
        "by_friction":{},"by_mass":{}}
    for key,groups in [("friction",[0.5,1.0,2.0]),("mass_kg",[0.04,0.08,0.15])]:
        target="by_friction" if key=="friction" else "by_mass"
        for value in groups:
            selected=[r for r in rows if r[key]==value]
            summary[target][str(value)]={"trials":len(selected),"successes":sum(r["success"] for r in selected),
                "success_rate":float(np.mean([r["success"] for r in selected]))}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (output/"trials.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    values=np.array([[np.mean([r["success"] for r in rows if r["mass_kg"]==mass and r["friction"]==friction])
                    for friction in [0.5,1.0,2.0]] for mass in [0.04,0.08,0.15]])
    chart=ax.imshow(values,vmin=0,vmax=1,cmap="YlGn")
    ax.set(xticks=range(3),xticklabels=[0.5,1.0,2.0],yticks=range(3),yticklabels=[0.04,0.08,0.15],
           xlabel="Friction coefficient",ylabel="Cube mass (kg)",title=f"Pick and place: {summary['success_rate']:.1%} over {len(rows)} trials")
    for y in range(3):
        for x in range(3):
            ax.text(x,y,f"{values[y,x]:.0%}",ha="center",va="center",color="white" if values[y,x]>0.5 else "black")
    fig.colorbar(chart,ax=ax,label="Success fraction")
    fig.tight_layout()
    fig.savefig(output/"robustness.png",dpi=150)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="reports/pick_place_robustness")
    parser.add_argument("--scenes",type=int,default=6)
    parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args()
    benchmark(args.output,args.scenes,args.seed)
