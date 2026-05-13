# auscult-ml

### Auscultation-Based Heart and Lung Disorder Classification

`auscult-ml` is a CS 422 machine learning project for classifying heart rhythm classes and lung sound disorders from 15-second digital stethoscope recordings. The pipeline turns raw auscultation audio into fixed-length feature tables, optionally adds metadata like chest location and gender, and compares baseline classifiers across heart-only, lung-only, and mixed-audio tasks.

### Built With

![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge)
![uv Badge](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=fff&style=for-the-badge)
![librosa Badge](https://img.shields.io/badge/librosa-C14D8C?style=for-the-badge)
![NumPy Badge](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge)
![pandas Badge](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=fff&style=for-the-badge)
![scikit-learn Badge](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=fff&style=for-the-badge)

## Features

### Audio Preprocessing

- **Feature Extraction** - Converts each recording into fixed-length numeric features using MFCCs, MFCC deltas, RMS, zero-crossing rate, and spectral features.
- **Tabular Datasets** - Builds four processed CSV datasets from the raw recordings for downstream training.
- **Spike-Based Windowing** - Uses sliding windows for lung-heavy tasks to focus on the most active parts of a recording.

### Multiple Classification Tasks

- **Heart and Lung Targets** - Supports heart-only, lung-only, mixed heart-label, and mixed lung-label prediction tasks.
- **Grouped Evaluation** - Keeps windows from the same recording in the same fold when needed.
- **File-Level Aggregation** - Averages window predictions back to the recording level for windowed tasks.

### Baseline Model Comparison

- **Multiple Baselines** - Trains decision tree, random forest, logistic regression, and SVM models.
- **Metadata Variants** - Compares audio-only runs against audio plus location and gender variants.
- **Cross-Validation Outputs** - Saves fold metrics, predictions, confusion matrices, and summaries for each run.
- **Report Figures** - Combines saved model summaries into comparison tables and bar charts for the final report.

## Dataset

This project uses the **[HLS-CMDS](https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope)** auscultation dataset from the **[UCI Machine Learning Repository](https://archive.ics.uci.edu/)**.

<details>
<summary>Dataset details</summary>

- **Source** - UCI dataset id `1202`
- **DOI** - <https://doi.org/10.1109/IEEEDATA.2025.3566012>
- **Dataset page** - <https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope>
- **License** - CC BY 4.0
- **535 total recordings**
- **15-second `.wav` clips** sampled at **22,050 Hz**
- Includes **heart-only**, **lung-only**, and **mixed cardiopulmonary** recordings
- Metadata includes **sound type**, **auscultation location**, **gender**, and **sound ID**
- **Heart Labels** - `NH`, `LDM`, `MSM`, `LSM`, `AF`, `S4`, `ESM`, `S3`, `T`, `AVB`
- **Lung Labels** - `NL`, `W`, `FC`, `R`, `PR`, `CC`
- **Sound ID** is only used to link metadata rows to audio files, not as a training feature
- In this repo, the raw dataset is expected locally as `data/raw/Mix.csv` plus the corresponding `.wav` files under `data/raw/mixed/`

</details>

## Prerequisites

- **Python 3.11+**
- **uv** installed locally: <https://docs.astral.sh/uv/>
- Raw data placed at `data/raw/Mix.csv` and `data/raw/mixed/*.wav`

The preprocessing pipeline expects file IDs in `Mix.csv` to match the `.wav` filenames in `data/raw/mixed/`.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Preprocess the raw audio

This reads `data/raw/Mix.csv` and the `.wav` files in `data/raw/mixed/`, then writes processed feature tables to `data/processed/`.

```bash
uv run python scripts/preprocess_data.py
```

### 3. Train one baseline model

```bash
uv run scripts/train_decision_tree.py
uv run scripts/train_random_forest.py
uv run scripts/train_logistic_regression.py
uv run scripts/train_svm.py
```

### 4. Run all baseline models

```bash
uv run scripts/train_all_models.py
```

### 5. Compare models and generate figures

After training at least one model, collect the saved summaries and create tables/bar charts.

```bash
uv run scripts/compare_results.py
```

## Usage

### Run a single task

```bash
uv run scripts/train_decision_tree.py --task lungs_only__lung
uv run scripts/train_random_forest.py --task mixed_full__heart
uv run scripts/train_svm.py --task mixed_windowed__lung
```

### Include metadata features

Metadata is joined from `Mix.csv` and currently supports `Location` and `Gender`.

```bash
uv run scripts/train_decision_tree.py --include-location
uv run scripts/train_decision_tree.py --include-gender
uv run scripts/train_decision_tree.py --include-location --include-gender
```

### Run all models across selected metadata variants

```bash
uv run scripts/train_all_models.py --task lungs_only__lung --metadata none location both
uv run scripts/train_all_models.py --models decision_tree random_forest --metadata both
```

Supported `--metadata` variants:

- `none`
- `location`
- `gender`
- `both`

### Compare saved results

```bash
uv run scripts/compare_results.py --metric f1_macro_mean
uv run scripts/compare_results.py --task mixed_windowed__lung --metric accuracy_mean
uv run scripts/compare_results.py --no-plots
```

The comparison script writes:

- `results/comparison/all_results.csv`
- `results/comparison/best_by_task.csv`
- `results/comparison/report_table.csv`
- `results/comparison/parameter_summary.csv`
- `results/comparison/*.png` bar charts
- `results/comparison/*_confusion_matrix.png` confusion matrices

## Roadmap

- [x] Preprocess raw auscultation audio into feature tables
- [x] Train decision tree, random forest, logistic regression, and SVM baselines
- [x] Support metadata variants for location and gender
- [x] Save per-run metrics, predictions, confusion matrices, and summaries
- [x] Add a top-level comparison script for summarizing all model results
- [x] Add charts for model comparison
