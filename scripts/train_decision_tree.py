from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auscult_ml.models.common import TASKS
from auscult_ml.models.decision_tree import run_decision_tree


def parse_args():
    parser = ArgumentParser(description="Train and evaluate the decision tree model.")
    parser.add_argument(
        "--task",
        choices=["all", *sorted(TASKS)],
        default="all",
        help="Task preset to run. Defaults to all tasks.",
    )
    parser.add_argument("--include-location", action="store_true", help="Include Location metadata.")
    parser.add_argument("--include-gender", action="store_true", help="Include Gender metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Let one script handle either a single preset or the full task list.
    task_names = sorted(TASKS) if args.task == "all" else [args.task]

    print(f"Decision tree run starting for {len(task_names)} task(s)")

    for index, task_name in enumerate(task_names, start=1):
        print(f"\nTask {index}/{len(task_names)}: {task_name}")
        # Each task writes its own metrics, predictions, and confusion matrix.
        run_decision_tree(
            task_name=task_name,
            include_location=args.include_location,
            include_gender=args.include_gender,
        )

    print("\nDecision tree run complete")


if __name__ == "__main__":
    main()
