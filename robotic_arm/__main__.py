"""One command-line entry point for the semester project."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    "record-pick-place": "robotic_arm.evaluation.record_pick_place",
    "train-shared-policy": "robotic_arm.training.train_shared_policy",
    "learn-persistent-pick-place": "robotic_arm.training.learn_persistent_pick_place",
    "persistent-pick-place": "robotic_arm.evaluation.evaluate_persistent_pick_place",
    "refine-obstacle-policy": "robotic_arm.training.refine_obstacle_policy",
    "benchmark-learned-pick-place": "robotic_arm.experiments.benchmark_learned_pick_place",
    "learn-pick-place": "robotic_arm.training.learn_pick_place",
    "train-pick-place": "robotic_arm.training.train_pick_place",
    "evaluate-pick-place": "robotic_arm.evaluation.evaluate_pick_place",
    "evaluate": "robotic_arm.evaluation.evaluate_policy",
    "pick-place": "robotic_arm.controllers.pick_place_controller",
    "train": "robotic_arm.training.train_ppo",
    "record": "robotic_arm.evaluation.record_policy",
    "compare-rewards": "robotic_arm.experiments.compare_rewards",
    "benchmark-pick-place": "robotic_arm.experiments.benchmark_pick_place",
    "analyze-failures": "robotic_arm.experiments.analyze_failures",
}


def main():
    parser=argparse.ArgumentParser(prog="python -m robotic_arm",description="Robotic arm reaching and pick-and-place simulation")
    parser.add_argument("command",choices=COMMANDS)
    parser.add_argument("arguments",nargs=argparse.REMAINDER,help="Options for the selected command")
    args=parser.parse_args()
    original_arguments=sys.argv
    try:
        sys.argv=[f"{parser.prog} {args.command}",*args.arguments]
        import_module(COMMANDS[args.command]).main()
    finally:
        sys.argv=original_arguments


if __name__ == "__main__":
    main()
