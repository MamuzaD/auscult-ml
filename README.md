# auscult-ml

## Quick Start

Generate processed feature tables:

```bash
uv sync
uv run python scripts/preprocess_data.py
```

Train the decision tree and random forest baselines:

```bash
uv run scripts/train_decision_tree.py
uv run scripts/train_random_forest.py
```

Run a single task or include metadata features:

```bash
uv run scripts/train_decision_tree.py --task lungs_only__lung
uv run scripts/train_decision_tree.py --include-location --include-gender
uv run scripts/train_random_forest.py --task lungs_only__lung
```

## Installation

```bash
uv sync
```

<details>
<summary>General UV Usage</summary>

- Sync dependencies:

  ```bash
  uv sync
  ```

- Run a script without activating a virtual environment:

  ```bash
  uv run python scripts/script.py
  ```

- Run a package module directly:

  ```bash
  uv run python -m package.module
  ```

- Activate the local virtual environment manually:

  ```bash
  source .venv/bin/activate
  ```

  Then run commands normally:

  ```bash
  python scripts/script.py
  ```

- Typical pattern:

  ```bash
  uv sync
  uv run python scripts/script.py
  ```

</details>
