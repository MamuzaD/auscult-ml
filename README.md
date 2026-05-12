# auscult-ml

## Quick Start

Generate processed feature tables:

```bash
uv sync
uv run python scripts/preprocess_data.py
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
