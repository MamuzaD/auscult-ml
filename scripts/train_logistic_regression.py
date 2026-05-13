from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli_helpers import add_metadata_feature_arguments, add_task_argument, make_parser
from auscult_ml.models.common import TASKS
from auscult_ml.models.logistic_regression import run_logistic_regression


def parse_args():
    parser = make_parser("Train and evaluate the logistic regression model.")
    add_task_argument(parser, TASKS)
    add_metadata_feature_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    task_names = sorted(TASKS) if args.task == "all" else [args.task]

    print(f"logistic regression run starting for {len(task_names)} task(s)")

    for index, task_name in enumerate(task_names, start=1):
        print(f"\nTask {index}/{len(task_names)}: {task_name}")
        run_logistic_regression(
            task_name=task_name,
            include_location=args.include_location,
            include_gender=args.include_gender,
        )

    print("\nlogistic regression run complete")


if __name__ == "__main__":
    main()
