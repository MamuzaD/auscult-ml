from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli_helpers import add_compare_arguments, make_parser
from auscult_ml.results import compare_results


# arguments for choosing which saved experiment results to summarize
def parse_args():
    parser = make_parser("Compare saved model results and generate report figures.")
    add_compare_arguments(parser)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write comparison CSVs; skip PNG figures.",
    )
    return parser.parse_args()


# print compact metric table
def print_metric_table(results_df):
    # keep only the columns needed for the comparison table
    table = results_df[
        [
            "task",
            "model_display",
            "metadata_variant",
            "accuracy_mean",
            "precision_macro_mean",
            "recall_macro_mean",
            "f1_macro_mean",
        ]
    ].copy()

    table = table.sort_values(
        ["task", "f1_macro_mean", "accuracy_mean"], ascending=[True, False, False]
    )

    # higher macro F1 is better for our multi-class tasks, especially with uneven classes
    print("\n--- Model Comparison Results ---")

    for _, row in table.iterrows():
        print(
            f"{row['task']} | {row['model_display']} ({row['metadata_variant']}): "
            f"Accuracy={row['accuracy_mean']:.4f}, "
            f"Precision={row['precision_macro_mean']:.4f}, "
            f"Recall={row['recall_macro_mean']:.4f}, "
            f"F1={row['f1_macro_mean']:.4f}"
        )


def main():
    args = parse_args()

    try:
        # compare_results reads saved summary CSVs and writes tables/figures
        results_df, table_paths, plot_paths = compare_results(
            task=args.task,
            metric=args.metric,
            make_figures=not args.no_plots,
        )
    except ValueError as exc:
        print(exc)
        return

    print("\n--- Comparison Summary ---")
    print(f"Runs compared: {len(results_df)}")
    print_metric_table(results_df)

    # these CSV files are the tables for our evaluation
    for name, path in table_paths.items():
        print(f"Saved {name}: {path.relative_to(ROOT)}")

    # these PNG files are the figures for our evaluation
    for path in plot_paths:
        print(f"Saved figure: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
