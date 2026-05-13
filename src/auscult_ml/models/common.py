from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MIX_METADATA_PATH = RAW_DATA_DIR / "Mix.csv"
HEART_METADATA_PATH = RAW_DATA_DIR / "HS.csv"
LUNG_METADATA_PATH = RAW_DATA_DIR / "LS.csv"

ID_COL = "id"
HEART_LABEL = "Heart Sound Type"
LUNG_LABEL = "Lung Sound Type"
LABEL_COLUMNS = [HEART_LABEL, LUNG_LABEL]
DEFAULT_SPLITS = 5
RANDOM_STATE = 1111


@dataclass(frozen=True)
class ClassificationTask:
    name: str
    dataset: str
    target: str
    grouped: bool
    aggregate_by_id: bool


TASKS = {
    "heart_only__heart": ClassificationTask(
        name="heart_only__heart",
        dataset="heart_only.csv",
        target=HEART_LABEL,
        grouped=False,
        aggregate_by_id=False,
    ),
    "lungs_only__lung": ClassificationTask(
        name="lungs_only__lung",
        dataset="lungs_only.csv",
        target=LUNG_LABEL,
        grouped=True,
        aggregate_by_id=True,
    ),
    "mixed_full__heart": ClassificationTask(
        name="mixed_full__heart",
        dataset="mixed_without_sliding_window.csv",
        target=HEART_LABEL,
        grouped=False,
        aggregate_by_id=False,
    ),
    "mixed_full__lung": ClassificationTask(
        name="mixed_full__lung",
        dataset="mixed_without_sliding_window.csv",
        target=LUNG_LABEL,
        grouped=False,
        aggregate_by_id=False,
    ),
    "mixed_windowed__heart": ClassificationTask(
        name="mixed_windowed__heart",
        dataset="mixed_with_sliding_window.csv",
        target=HEART_LABEL,
        grouped=True,
        aggregate_by_id=True,
    ),
    "mixed_windowed__lung": ClassificationTask(
        name="mixed_windowed__lung",
        dataset="mixed_with_sliding_window.csv",
        target=LUNG_LABEL,
        grouped=True,
        aggregate_by_id=True,
    ),
}


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_string_columns(df):
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].str.strip()

    return df


def normalize_lung_file_id(file_id):
    return file_id.replace("_C_", "_FC_").replace("_G_", "_CC_")


def metadata_frame(path, id_column, normalize_id=None):
    metadata = clean_string_columns(pd.read_csv(path))
    frame = metadata[[id_column, "Location", "Gender"]].rename(
        columns={id_column: ID_COL}
    )

    if normalize_id is not None:
        frame[ID_COL] = frame[ID_COL].map(normalize_id)

    return frame


def build_metadata_lookup():
    mix_metadata = clean_string_columns(pd.read_csv(MIX_METADATA_PATH))

    frames = [
        mix_metadata[["Heart Sound ID", "Location", "Gender"]].rename(
            columns={"Heart Sound ID": ID_COL}
        ),
        mix_metadata[["Lung Sound ID", "Location", "Gender"]].rename(
            columns={"Lung Sound ID": ID_COL}
        ),
        mix_metadata[["Mixed Sound ID", "Location", "Gender"]].rename(
            columns={"Mixed Sound ID": ID_COL}
        ),
    ]

    if HEART_METADATA_PATH.exists():
        frames.append(metadata_frame(HEART_METADATA_PATH, "Heart Sound ID"))

    if LUNG_METADATA_PATH.exists():
        frames.append(
            metadata_frame(
                LUNG_METADATA_PATH, "Lung Sound ID", normalize_id=normalize_lung_file_id
            )
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=[ID_COL])

def load_dataset(dataset_name, metadata_lookup):
    dataset_path = PROCESSED_DATA_DIR / dataset_name
    df = pd.read_csv(dataset_path)
    merged = df.merge(metadata_lookup, on=ID_COL, how="left", validate="many_to_one")

    if merged[["Location", "Gender"]].isna().any().any():
        missing_ids = merged.loc[
            merged[["Location", "Gender"]].isna().any(axis=1), ID_COL
        ].unique()
        raise ValueError(
            f"Missing metadata for ids: {sorted(missing_ids.tolist())[:10]}"
        )

    return merged


def metadata_variant_name(include_location, include_gender):
    columns = []
    if include_location:
        columns.append("location")
    if include_gender:
        columns.append("gender")
    return "audio_only" if not columns else "audio_plus_" + "_".join(columns)


def make_preprocessor(df, include_location, include_gender):
    metadata_columns = []
    if include_location:
        metadata_columns.append("Location")
    if include_gender:
        metadata_columns.append("Gender")

    metadata_pool = {"Location", "Gender"}
    excluded = {ID_COL, *LABEL_COLUMNS, *metadata_pool}
    numeric_columns = [column for column in df.columns if column not in excluded]

    transformers = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_columns,
        )
    ]

    if metadata_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                metadata_columns,
            )
        )

    return ColumnTransformer(transformers=transformers)


def determine_n_splits(df, target, grouped):
    if grouped:
        counts = df.groupby(ID_COL)[target].first().value_counts()
    else:
        counts = df[target].value_counts()

    if counts.empty or int(counts.min()) < 2:
        raise ValueError(f"Need at least two examples per class for {target}.")

    return min(DEFAULT_SPLITS, int(counts.min()))


def make_splitter(n_splits, grouped):
    if grouped:
        try:
            return StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
            )
        except TypeError:
            return GroupKFold(n_splits=n_splits)

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def make_inner_splitter(X_train, y_train, grouped):
    train_df = X_train.copy()
    train_df["__target__"] = y_train.to_numpy()
    n_splits = determine_n_splits(
        train_df.rename(columns={"__target__": "target"}), "target", grouped
    )
    return make_splitter(min(3, n_splits), grouped)


def aggregate_group_predictions(ids, y_true, probabilities, classes):
    fold_df = pd.DataFrame(probabilities, columns=classes)
    fold_df[ID_COL] = ids.to_numpy()
    fold_df["y_true"] = y_true.to_numpy()

    grouped_proba = fold_df.groupby(ID_COL, sort=False)[list(classes)].mean()
    grouped_truth = fold_df.groupby(ID_COL, sort=False)["y_true"].first()
    y_pred = grouped_proba.idxmax(axis=1)

    return pd.DataFrame(
        {
            ID_COL: grouped_truth.index,
            "y_true": grouped_truth.values,
            "y_pred": y_pred.values,
        }
    )


def score_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
