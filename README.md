# arbCube

GNN trained on CFOP 3x3 solutions applied to arbitrary cubes.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
the following from the repository root:

```sh
uv sync --locked
```

This creates a local `.venv` using Python 3.12 (specified in `.python-version`)
and installs the dependencies from `uv.lock`. uv downloads Python if needed.
The project currently has no dependencies.

Run Python or a project script inside the managed environment:

```sh
uv run python
# Once you have a script: uv run python path/to/script.py
```

Environment activation is optional; `uv run` uses `.venv` automatically.

## Managing dependencies

Use uv to update the shared dependency files:

```sh
uv add <package>
uv add --dev <package>
uv remove <package>
```

To intentionally upgrade all dependencies allowed by `pyproject.toml`:

```sh
uv lock --upgrade
uv sync --locked
```

Commit `pyproject.toml` and `uv.lock` together whenever dependencies change.
Commit `.python-version` when changing the team's Python version. After pulling
changes, run `uv sync --locked` to update your environment. `.venv` is ignored
by Git. Let uv generate `uv.lock` rather than editing it by hand.
