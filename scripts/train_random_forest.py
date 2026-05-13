from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auscult_ml.models.common import TASKS
from auscult_ml.models.random_forest import run_random_forest


def parse_args():
    parser = ArgumentParser(description="Train and evaluate the random forest model.")
    parser.add_argument(
        "--task",
        choices=["all", *sorted(TASKS)],
        default="all",
        help="Task preset to run. Defaults to all tasks.",
    )
    parser.add_argument(
        "--include-location", action="store_true", help="Include Location metadata."
    )
    parser.add_argument(
        "--include-gender", action="store_true", help="Include Gender metadata."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_names = sorted(TASKS) if args.task == "all" else [args.task]

    print(f"Random forest run starting for {len(task_names)} task(s)")

    for index, task_name in enumerate(task_names, start=1):
        print(f"\nTask {index}/{len(task_names)}: {task_name}")
        run_random_forest(
            task_name=task_name,
            include_location=args.include_location,
            include_gender=args.include_gender,
        )

    print("\nRandom forest run complete")


if __name__ == "__main__":
    main()
