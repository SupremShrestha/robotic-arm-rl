"""Equal-budget, paired-seed PPO reward ablation from a common checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7,19,42])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, choices=[1,2], default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model/best_model.zip"))
    parser.add_argument("--output", type=Path, default=Path("reports/ablation"))
    args = parser.parse_args()
    if args.timesteps < 1 or args.episodes < 1 or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive budgets and unique nonnegative seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"requested_additional_steps":args.timesteps, "training_seeds":args.seeds,
                "evaluation_episodes_per_set":args.episodes, "test_seed_start":0,
                "holdout_seed_start":1000, "validation_seed_start":10000,
                "initial_checkpoint":str(args.checkpoint),
                "initial_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "design":"Paired continuation seeds, identical starting checkpoint and step budgets; not independent training from scratch."}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    def job(mode, seed):
        directory = args.output/f"{mode}_{seed}"
        directory.mkdir()
        with (directory/"console.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-m","training.train","--resume",str(args.checkpoint),
                            "--timesteps",str(args.timesteps),"--seed",str(seed),"--reward-mode",mode,
                            "--eval-freq","10000","--eval-episodes","20","--run-dir",str(directory/"run")],
                            stdout=log,stderr=subprocess.STDOUT,check=True)
            for name, start in [("test",0),("holdout",1000)]:
                subprocess.run([sys.executable,"evaluate.py","--model",str(directory/"run/best_model.zip"),
                                "--episodes",str(args.episodes),"--seed",str(start),"--output",str(directory/name)],
                                stdout=log,stderr=subprocess.STDOUT,check=True)
        return {"reward_mode":mode,"training_seed":seed,
                "test":json.loads((directory/"test/summary.json").read_text()),
                "holdout":json.loads((directory/"holdout/summary.json").read_text())}

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job,mode,seed) for seed in args.seeds for mode in ["legacy","shaped"]]
        for future in as_completed(futures):
            record=future.result()
            records.append(record)
            print(f"Completed {record['reward_mode']} seed {record['training_seed']}: held-out {record['holdout']['success_rate']:.1%}",flush=True)
            (args.output/"runs.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    summary={"design":manifest,"conditions":{},"paired_holdout_difference_shaped_minus_legacy":[]}
    for mode in ["legacy","shaped"]:
        group=[r for r in records if r["reward_mode"]==mode]
        summary["conditions"][mode]={}
        for split in ["test","holdout"]:
            rates=[r[split]["success_rate"] for r in group]
            summary["conditions"][mode][split]={"mean_success_rate":float(np.mean(rates)),
                "sample_std_across_training_seeds":float(np.std(rates,ddof=1)) if len(rates)>1 else None,
                "per_seed":{str(r["training_seed"]):r[split]["success_rate"] for r in group}}
    for seed in args.seeds:
        pair={r["reward_mode"]:r for r in records if r["training_seed"]==seed}
        summary["paired_holdout_difference_shaped_minus_legacy"].append({"seed":seed,
            "difference":pair["shaped"]["holdout"]["success_rate"]-pair["legacy"]["holdout"]["success_rate"]})
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary["conditions"],indent=2),flush=True)


if __name__ == "__main__":
    main()
