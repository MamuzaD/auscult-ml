import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from auscult_ml.models.common import (
    ID_COL,
    LABEL_COLUMNS,
    RANDOM_STATE,
    TASKS,
    aggregate_group_predictions,
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
RESULTS_DIR = ROOT / "results" / "random_forest"


def make_param_grid():
    return {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 8, 12, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", 0.5, 0.75],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
        "model__bootstrap": [True],
    }


def determine_search_iterations(task):
    # harder to tune grouped, so slightly larger budget
    return 24 if task.grouped else 18


def run_random_forest(task_name, include_location=False, include_gender=False):
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

    print(f"Running random forest for {task.name} [{variant_name}]")

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
                    RandomForestClassifier(random_state=RANDOM_STATE),
                ),
            ]
        )

        # tune the model inside each training fold before testing on the holdout fold.
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=make_param_grid(),
            n_iter=determine_search_iterations(task),
            scoring="f1_macro",
            cv=make_inner_splitter(X_train, y_train, task.grouped),
            n_jobs=-1,
            refit=True,
            random_state=RANDOM_STATE + fold_index,
        )

        fit_kwargs = {}
        if task.grouped:
            fit_kwargs["groups"] = X_train[ID_COL].to_numpy()

        print(f"  Fold {fold_index}/{n_splits}")
        search.fit(X_train, y_train, **fit_kwargs)
        best_pipeline = search.best_estimator_
        probabilities = best_pipeline.predict_proba(X_test)
        classes = best_pipeline.named_steps["model"].classes_

        # grouped tasks average window predictions back to one prediction per file.
        if task.aggregate_by_id:
            predictions_df = aggregate_group_predictions(
                X_test[ID_COL], y_test, probabilities, classes
            )
        else:
            y_pred = classes[probabilities.argmax(axis=1)]
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

    # save a small summary
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
    print(f"Saved random forest results to {relative_output_dir}")
    return summary
