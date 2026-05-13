import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from auscult_ml.models.common import (
    ID_COL,
    LABEL_COLUMNS,
    RANDOM_STATE,
    TASKS,
    build_metadata_lookup,
    determine_n_splits,
    load_dataset,
    make_inner_splitter,
    make_preprocessor,
    make_splitter,
    metadata_variant_name,
    score_predictions,
)

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "svm"


def make_param_grid():
    return [
        {
            "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],
            "model__loss": ["squared_hinge"],
        }
    ]


def ensure_2d_scores(scores):
    if scores.ndim == 1:
        return np.column_stack([-scores, scores])
    return scores


def aggregate_group_outputs(ids, y_true, outputs, classes):
    fold_df = pd.DataFrame(outputs, columns=classes)
    fold_df[ID_COL] = ids.to_numpy()
    fold_df["y_true"] = y_true.to_numpy()

    grouped_outputs = fold_df.groupby(ID_COL, sort=False)[list(classes)].mean()
    grouped_truth = fold_df.groupby(ID_COL, sort=False)["y_true"].first()
    y_pred = grouped_outputs.idxmax(axis=1)

    return pd.DataFrame(
        {
            ID_COL: grouped_truth.index,
            "y_true": grouped_truth.values,
            "y_pred": y_pred.values,
        }
    )


def run_svm(task_name, include_location=False, include_gender=False):
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Choices: {sorted(TASKS)}")

    task = TASKS[task_name]
    metadata_lookup = build_metadata_lookup()
    df = load_dataset(task.dataset, metadata_lookup)
    variant_name = metadata_variant_name(include_location, include_gender)
    output_dir = RESULTS_DIR / task.name / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = make_preprocessor(df, include_location, include_gender)
    n_splits = determine_n_splits(df, task.target, task.grouped)
    splitter = make_splitter(n_splits, task.grouped)

    # use every non-label column as input features.
    feature_columns = [column for column in df.columns if column not in LABEL_COLUMNS]
    X = df[feature_columns]
    y = df[task.target]
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
    groups = df[ID_COL] if task.grouped else None

    fold_metrics = []
    fold_predictions = []

    print(f"Running svm for {task.name} [{variant_name}]")

    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(X, y_encoded, groups), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_encoded.iloc[train_idx]
        y_test = y_encoded.iloc[test_idx]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LinearSVC(
                        random_state=RANDOM_STATE,
                        dual="auto",
                        tol=1e-3,
                        max_iter=200000,
                    ),
                ),
            ]
        )

        # tune the model inside each training fold before testing on the holdout fold.
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=make_param_grid(),
            scoring="f1_macro",
            cv=make_inner_splitter(X_train, y_train, task.grouped),
            n_jobs=-1,
            refit=True,
        )

        fit_kwargs = {}
        if task.grouped:
            fit_kwargs["groups"] = X_train[ID_COL].to_numpy()

        print(f"  Fold {fold_index}/{n_splits}")
        search.fit(X_train, y_train, **fit_kwargs)
        best_pipeline = search.best_estimator_
        model = best_pipeline.named_steps["model"]
        classes = model.classes_
        scores = ensure_2d_scores(best_pipeline.decision_function(X_test))

        # grouped tasks average per-window class scores back to one file-level prediction.
        if task.aggregate_by_id:
            predictions_df = aggregate_group_outputs(
                X_test[ID_COL], y_test, scores, classes
            )
        else:
            y_pred = classes[scores.argmax(axis=1)]
            predictions_df = pd.DataFrame(
                {
                    ID_COL: X_test[ID_COL].to_numpy(),
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred,
                }
            )

        predictions_df["y_true"] = label_encoder.inverse_transform(
            predictions_df["y_true"].astype(int)
        )
        predictions_df["y_pred"] = label_encoder.inverse_transform(
            predictions_df["y_pred"].astype(int)
        )
        predictions_df["fold"] = fold_index

        metrics = score_predictions(predictions_df["y_true"], predictions_df["y_pred"])
        metrics["fold"] = fold_index
        metrics["n_test_examples"] = int(len(predictions_df))
        metrics["best_params"] = json.dumps(search.best_params_, sort_keys=True)
        fold_metrics.append(metrics)
        fold_predictions.append(predictions_df)

        print(
            f"    macro_f1={metrics['f1_macro']:.3f} accuracy={metrics['accuracy']:.3f} "
            f"params={search.best_params_}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    labels = sorted(predictions_df["y_true"].unique().tolist())
    matrix = confusion_matrix(
        predictions_df["y_true"], predictions_df["y_pred"], labels=labels
    )

    # save a small summary so results are easy to compare later.
    summary = {
        "task": task.name,
        "dataset": task.dataset,
        "target": task.target,
        "metadata_variant": variant_name,
        "grouped_cv": task.grouped,
        "file_level_evaluation": task.aggregate_by_id,
        "accuracy_mean": metrics_df["accuracy"].mean(),
        "accuracy_std": metrics_df["accuracy"].std(ddof=0),
        "precision_macro_mean": metrics_df["precision_macro"].mean(),
        "precision_macro_std": metrics_df["precision_macro"].std(ddof=0),
        "recall_macro_mean": metrics_df["recall_macro"].mean(),
        "recall_macro_std": metrics_df["recall_macro"].std(ddof=0),
        "f1_macro_mean": metrics_df["f1_macro"].mean(),
        "f1_macro_std": metrics_df["f1_macro"].std(ddof=0),
        "n_splits": n_splits,
    }

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    relative_output_dir = output_dir.relative_to(ROOT)
    print(f"Saved svm results to {relative_output_dir}")
    return summary
