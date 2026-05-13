from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

METADATA_VARIANTS = {
    "none": [],
    "location": ["--include-location"],
    "gender": ["--include-gender"],
    "both": ["--include-location", "--include-gender"],
}


def parse_args():
    parser = ArgumentParser(
        description="Run every training script across selected metadata variants."
    )
    parser.add_argument(
        "--task",
        default="all",
        help="Task preset to pass through to each training script. Defaults to all.",
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        choices=list(METADATA_VARIANTS),
        default=["none", "location", "gender", "both"],
        help="Metadata variants to run. Defaults to all variants.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Optional model script names without the train_ prefix, such as "
            "decision_tree or random_forest. Defaults to all training scripts."
        ),
    )
    return parser.parse_args()


def discover_training_scripts(selected_models=None):
    scripts = []

    for path in sorted(SCRIPTS_DIR.glob("train_*.py")):
        if path.name == Path(__file__).name:
            continue

        model_name = path.stem.removeprefix("train_")
        if selected_models and model_name not in selected_models:
            continue

        scripts.append((model_name, path))

    if not scripts:
        requested = ", ".join(selected_models or []) or "all"
        raise ValueError(f"No training scripts matched selection: {requested}")

    return scripts


def build_command(script_path, task_name, metadata_variant):
    command = [sys.executable, str(script_path), "--task", task_name]
    command.extend(METADATA_VARIANTS[metadata_variant])
    return command


def main():
    args = parse_args()
    training_scripts = discover_training_scripts(args.models)
    total_runs = len(training_scripts) * len(args.metadata)

    print(f"Running {len(training_scripts)} model(s) across {len(args.metadata)} metadata variant(s)")

    run_index = 0
    for model_name, script_path in training_scripts:
        for metadata_variant in args.metadata:
            run_index += 1
            command = build_command(script_path, args.task, metadata_variant)
            print(
                f"\nRun {run_index}/{total_runs}: {model_name} [{metadata_variant}]"
            )
            subprocess.run(command, check=True, cwd=ROOT)

    print("\nAll model runs complete")


if __name__ == "__main__":
    main()
