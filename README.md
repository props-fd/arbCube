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

Run Python or a project script inside the managed environment:

```sh
uv run python
# Once you have a script: uv run python path/to/script.py
```

Environment activation is optional; `uv run` uses `.venv` automatically.

## CFOP dataset

Download the [Rubik's cube CFOP solutions dataset](https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions)
into `data/raw/cfop-v1/`:

```sh
uv run python -m arbcube.data
```

The downloader uses KaggleHub and pins dataset version 1 (the latest when added)
so collaborators analyze the same data. Repeating the command reuses the local
download. Raw data and generated reports are ignored by Git; share the download
command instead of committing the dataset. The download is about 137 MiB and
extracts to about 756 MiB.

For Python scripts or an interactive session from the repo root:

```python
from arbcube.data import load_solutions, load_transitions

solutions = load_solutions()
transitions = load_transitions(limit_per_shard=1_000)
```

`load_solutions()` reads all complete solutions; pass `limit_per_shard=1_000`
for a smaller starting sample. `load_transitions()` reads a bounded prefix of
each sequence shard, rather than loading the much larger sequence files whole.
These prefixes are convenience samples, not random samples.

The `training.N` files alternate a 54-color state with a complete solution;
`training.seq.N` alternates a 54-integer state with one action or the terminal
marker `#`. Blank separator lines are skipped. Shard names and record indices
are preserved; the role of `training.99` is unverified, so check the intended
train/test split before training. Face ordering and the relationship between
the color and integer encodings also need verification before building graphs.

If Kaggle requests authentication, follow the
[KaggleHub authentication instructions](https://github.com/Kaggle/kagglehub#authenticate).

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
