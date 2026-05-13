import json
import os
import tempfile
from pathlib import Path

import pandas as pd

# paths and metrics
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"

METRIC_COLUMNS = [
    "accuracy_mean",
    "precision_macro_mean",
    "recall_macro_mean",
    "f1_macro_mean",
]


# load saved model results
def display_model_name(model_name):
    # make model folder names easier to read in tables and figures
    names = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
    }
    return names.get(model_name, model_name.replace("_", " ").title())


def collect_summary_files(results_dir=RESULTS_DIR):
    # each training script saves one summary.csv for a model/task/metadata run
    files = []

    if not results_dir.exists():
        return files

    for path in sorted(results_dir.glob("*/*/*/summary.csv")):
        if "comparison" not in path.parts:
            files.append(path)

    return files


def collect_run_dirs(results_dir=RESULTS_DIR):
    # run folder also contains fold metrics, predictions, and confusion matrices
    run_dirs = []

    for summary_path in collect_summary_files(results_dir):
        run_dirs.append(summary_path.parent)

    return run_dirs


def load_results(results_dir=RESULTS_DIR):
    # combine all model summary files into one dataframe for comparison
    rows = []

    for path in collect_summary_files(results_dir):
        model_name = path.relative_to(results_dir).parts[0]
        summary = pd.read_csv(path)
        summary["model"] = model_name
        summary["model_display"] = display_model_name(model_name)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    # put the most useful report columns first
    df = pd.concat(rows, ignore_index=True)
    ordered_columns = [
        "model",
        "model_display",
        "task",
        "dataset",
        "target",
        "metadata_variant",
        "grouped_cv",
        "file_level_evaluation",
        *METRIC_COLUMNS,
        "accuracy_std",
        "precision_macro_std",
        "recall_macro_std",
        "f1_macro_std",
        "n_splits",
    ]
    existing_columns = [column for column in ordered_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in existing_columns]

    return df[existing_columns + remaining_columns]


# comparison tables
def best_rows_by_task(results_df, metric):
    # for each task, keep the best run according to the selected metric
    if results_df.empty:
        return results_df

    sorted_df = results_df.sort_values(
        ["task", metric, "accuracy_mean"], ascending=[True, False, False]
    )
    return sorted_df.groupby("task", as_index=False).head(1)


def save_comparison_tables(
    results_df, output_dir=COMPARISON_DIR, metric="f1_macro_mean", task=None
):
    # save full tables plus smaller tables (for report)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results_path = output_dir / "all_results.csv"
    best_path = output_dir / "best_by_task.csv"
    report_table_path = output_dir / "report_table.csv"
    parameter_path = output_dir / "parameter_summary.csv"

    sorted_results = results_df.sort_values(
        ["task", metric, "accuracy_mean"], ascending=[True, False, False]
    )
    sorted_results.to_csv(all_results_path, index=False)

    best_df = best_rows_by_task(results_df, metric)
    best_df.to_csv(best_path, index=False)

    report_columns = [
        "task",
        "model_display",
        "metadata_variant",
        "accuracy_mean",
        "precision_macro_mean",
        "recall_macro_mean",
        "f1_macro_mean",
    ]
    report_df = sorted_results[report_columns].copy()

    # round metrics so the report table is easy to read
    for column in METRIC_COLUMNS:
        report_df[column] = report_df[column].round(3)

    report_df.to_csv(report_table_path, index=False)
    parameter_df = summarize_parameters(task=task)
    parameter_df.to_csv(parameter_path, index=False)

    return {
        "all_results": all_results_path,
        "best_by_task": best_path,
        "report_table": report_table_path,
        "parameter_summary": parameter_path,
    }


# helper function for turning saved GridSearchCV parameters into report text
def parse_params(params_text):
    try:
        return json.loads(params_text)
    except (TypeError, json.JSONDecodeError):
        return {}


def summarize_parameters(results_dir=RESULTS_DIR, task=None):
    # count how often each best parameter setting was selected across CV folds
    rows = []

    for run_dir in collect_run_dirs(results_dir):
        metrics_path = run_dir / "fold_metrics.csv"
        if not metrics_path.exists():
            continue

        model_name, task_name, metadata_variant = run_dir.relative_to(results_dir).parts
        if task and task_name != task:
            continue

        metrics_df = pd.read_csv(metrics_path)

        if "best_params" not in metrics_df.columns:
            continue

        for params_text, count in metrics_df["best_params"].value_counts().items():
            rows.append(
                {
                    "model": model_name,
                    "model_display": display_model_name(model_name),
                    "task": task_name,
                    "metadata_variant": metadata_variant,
                    "fold_count": int(count),
                    "best_params": params_text,
                    "best_params_readable": format_params(parse_params(params_text)),
                }
            )

    return pd.DataFrame(rows)


def format_params(params):
    # remove the model__ prefix that comes from scikit-learn pipelines
    pieces = []

    for key, value in sorted(params.items()):
        clean_key = key.replace("model__", "")
        pieces.append(f"{clean_key}={value}")

    return ", ".join(pieces)


# figures
def load_matplotlib():
    # use a writable cache directory so matplotlib works cleanly on this machine
    cache_dir = Path(tempfile.gettempdir()) / "auscult_ml_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plots. Install dependencies, then rerun."
        ) from exc

    return plt


def plot_best_models(best_df, metric, output_dir):
    # bar chart showing the single best model/metadata setting for each task
    plt = load_matplotlib()
    labels = best_df["task"]
    values = best_df[metric]
    colors = ["steelblue", "seagreen", "darkorange", "slateblue", "indianred", "teal"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=colors[: len(best_df)])
    plt.ylim(0, 1)
    plt.ylabel(metric.replace("_mean", "").replace("_", " ").title())
    plt.title("Best Model by Task")
    plt.xticks(rotation=35, ha="right")

    for bar, model_name in zip(bars, best_df["model_display"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            model_name,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()
    output_path = output_dir / "best_model_by_task.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# bar chart with plotted evaluation output
def plot_metric_comparison(results_df, task_name, metric, output_dir):
    # compare every model/metadata setting for one task
    plt = load_matplotlib()
    task_df = results_df[results_df["task"] == task_name].sort_values(
        [metric, "accuracy_mean"], ascending=False
    )

    labels = task_df["model_display"] + "\n" + task_df["metadata_variant"]
    values = task_df[metric]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, values, color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel(metric.replace("_mean", "").replace("_", " ").title())
    plt.title(task_name + " Model Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = output_dir / f"{task_name}_{metric}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_confusion_matrix(matrix_path, output_dir):
    # recreate a saved confusion matrix as a PNG for report/slides
    plt = load_matplotlib()
    from sklearn.metrics import ConfusionMatrixDisplay

    matrix_df = pd.read_csv(matrix_path, index_col=0)
    run_parts = matrix_path.relative_to(RESULTS_DIR).parts
    model_name, task_name, metadata_variant = run_parts[:3]

    display_labels = matrix_df.index.tolist()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix_df.values,
        display_labels=display_labels,
    )
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(
        display_model_name(model_name)
        + " "
        + task_name
        + " Confusion Matrix"
        + "\n"
        + metadata_variant
    )
    plt.tight_layout()
    output_path = output_dir / (
        model_name + "_" + task_name + "_" + metadata_variant + "_confusion_matrix.png"
    )
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_confusion_matrices(output_dir, task=None):
    # make confusion matrix figures for every saved run, or one selected task
    plot_paths = []

    for run_dir in collect_run_dirs():
        model_name, task_name, metadata_variant = run_dir.relative_to(RESULTS_DIR).parts
        if task and task_name != task:
            continue

        matrix_path = run_dir / "confusion_matrix.csv"
        if matrix_path.exists():
            plot_paths.append(plot_confusion_matrix(matrix_path, output_dir))

    return plot_paths


def make_plots(results_df, output_dir=COMPARISON_DIR, metric="f1_macro_mean", task=None):
    # generate all report figures from the saved experiment outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    best_df = best_rows_by_task(results_df, metric)
    if not best_df.empty:
        plot_paths.append(plot_best_models(best_df, metric, output_dir))

    task_names = [task] if task else sorted(results_df["task"].unique())
    for task_name in task_names:
        plot_paths.append(plot_metric_comparison(results_df, task_name, metric, output_dir))

    plot_paths.extend(plot_confusion_matrices(output_dir, task=task))

    return plot_paths


# main comparison helper
def compare_results(task=None, metric="f1_macro_mean", make_figures=True):
    # called by scripts/compare_results.py
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unknown metric '{metric}'. Choices: {METRIC_COLUMNS}")

    results_df = load_results()

    if task:
        results_df = results_df[results_df["task"] == task]

    if results_df.empty:
        raise ValueError(
            "No saved model summaries found. Run training scripts before comparing results."
        )

    table_paths = save_comparison_tables(results_df, metric=metric, task=task)
    plot_paths = []

    # figures are optional so tables can still be created in environments without matplotlib
    if make_figures:
        try:
            plot_paths = make_plots(results_df, metric=metric, task=task)
        except RuntimeError as exc:
            print(exc)

    return results_df, table_paths, plot_paths
