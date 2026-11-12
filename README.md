# arbCube

GNN trained on CFOP 3x3 solutions applied to arbitrary cubes.

## Results so far

Three pieces of work on one dataset, the Kaggle CFOP solutions (110,000 human solves of
the 3x3):

| Work | Where | Key result |
|---|---|---|
| Dataset, loaders, move-block (BPE) analysis | `arbcube/data.py`, `arbcube/bpe.py`, `notebooks/01_cfop_basics.py` | Human solutions average about 62 moves; a learned library of about 1,000 move blocks shortens them to about 15 blocks |
| Transfer of a puzzle-agnostic GNN across cubes | `cubetransfer.zip` | Pretraining on the 3x3x3 cuts the training data needed on the 4x4x4 by about 2.7x and on the 3x3x3x3 by 2.8x; trained on random walks, it stops solving 3x3 scrambles deeper than about 10 quarter turns |
| CFOP imitation + beam search; NxN reduction | `nxn_reduction/` | A policy trained on the human solves, with beam search, solves 99.9% of 1,000 held-out full 3x3 scrambles in 60 face turns (human: 62) |

### 1. The dataset and its move blocks

![Solution lengths across all complete solutions](docs/figures/cfop_solution_lengths.png)

The human CFOP solutions are about 62 moves long on average, almost all between 45 and
75.

![Solution length as the move library grows](docs/figures/bpe_length_curve.png)

BPE learns move blocks by repeatedly merging the most frequent adjacent pair. As the
library grows from the 18 single moves, solutions shrink from 61.6 blocks on average to
about 24 blocks with 218 distinct blocks, about 19 with 518 and about 15 with roughly
1,000: about a 4x compression. The steepest drop comes from the first 50 merges. These
numbers are compression of the training corpus, not held-out results.

<p>
<img src="docs/figures/bpe_block_lengths_218.png" width="49%" alt="Move block lengths, 218 distinct blocks">
<img src="docs/figures/bpe_block_lengths_518.png" width="49%" alt="Move block lengths, 518 distinct blocks">
</p>

Most learned blocks are 2 or 4 moves long, the length of common CFOP move pairs and
triggers. With 218 blocks the longest is 6 moves. With 518 blocks a tail of 8 to 17-move
blocks appears, the length of whole last-layer algorithms. A move library like this is a
candidate action space for the search below: about 4x fewer decisions per solve.

### 2. Transfer across cubes (cubetransfer)

A GNN with puzzle-agnostic inputs, trained on random scramble walks and searched with a
width-32 beam, is pretrained on the 3x3x3 and then fine-tuned on bigger puzzles.

- **Sample efficiency.** On the 4x4x4, fine-tuning reaches the 90% solve bar after a
  median of 150k training states, against 400k from scratch (8 seeds). On the 3x3x3x3 it
  needs 100k, against 280k on average (5 seeds). Without any fine-tuning, the 3x3x3
  weights already solve 48% of the 4x4x4 suite and 87% of the 3x3x3x3 suite. The 4D
  result varies between checkpoints, from 38% to 100% on the same scrambles.
- **The head start fades.** By about 1M 4x4x4 states, fine-tuned and scratch networks
  are level; pretraining buys speed, not a higher ceiling. After 3M states both GNNs
  solve about 70% of a depth 1-16 suite, against 49% for a parameter-matched MLP,
  matched in training states rather than compute.
- **Depth limit.** On the 3x3x3 the piece-slot GNN solves everything up to 10 quarter
  turns and little beyond 12; the sticker-graph variant hits the same wall, and a 16x
  wider beam buys about one move.

### 3. Imitation on the human solves (nxn_reduction)

![3x3 scrambles solved by scramble depth](nxn_reduction/results/figures/solve_by_depth.png)

Training on full human solves removes the depth limit. A one-hot MLP that predicts the
human's next move, searched with a width-256 beam, solves every random-walk scramble
from 1 to 20 quarter turns deep. The cubetransfer GNN curve above is beam 32.

![Full scrambles solved and solution length against beam width](nxn_reduction/results/figures/solve_vs_beam_width.png)

On full scrambles from the held-out test solves, the solve rate climbs with beam width:
81.5% at 32, 97.5% at 64, 99.5% at 128 and 99.9% of 1,000 at 256, in 60 face turns on
average (human: 62), at 0.2 s per scramble on a laptop CPU.

- **Next-move accuracy.** Top-1: MLP 79.8%; sticker GNN 63.2%, up from 54.7% before
  adding a feature for each sticker's home face; always guessing `U` 15%. The GNN trails
  most on the last layer (OLL 61% against 90%). In the search the gap widens: at beam 32
  the GNN solves 13% of full scrambles against the MLP's 81.5%.
- **NxN.** The reduction to a 3x3, move lifting and the even-N parity fixes are done and
  tested; center solving and edge pairing aren't written yet, so NxN cubes can't be
  solved end to end yet.

### Caveats

- `nxn_reduction/` uses its own cube convention, not SPEC.md's; states and move
  permutations can't be passed between it and cubetransfer without a converter. See
  `nxn_reduction/README.md`.
- The metrics differ. cubetransfer counts quarter turns and requires solutions within
  K + 4 moves. `nxn_reduction` counts face turns, has no length cap, and solves short
  scrambles the long, CFOP way (61% of depth 1-10 scrambles within 14 quarter turns).
- The BPE numbers are read off the notebook's plots; run `notebooks/01_cfop_basics.py`
  for exact values. The cubetransfer numbers come from the results inside
  `cubetransfer.zip`.

Details: `nxn_reduction/FINDINGS.md` (one page) and `nxn_reduction/results/RESULTS.md`
(all figures).

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

## Basic exploration with marimo

From the repo root, run:

```sh
uv sync --locked
uv run python -m arbcube.data
uv run marimo edit notebooks/01_cfop_basics.py
```

The notebook loads all 110,000 complete solutions and shows solution lengths,
move counts, and BPE compression. Set `rows_per_shard` to
an integer in the loading cell if you want to limit the rows read per shard.

The BPE section learns move blocks by repeatedly merging frequent adjacent
blocks, starting from single moves. A move block is a single move or a learned
sequence of moves. Adjust the merge budget to explore larger move libraries.
Merges never cross solution boundaries.

The solution-length curve plots mean move blocks per solution against the
number of distinct move blocks on a linear x axis, with 10–90% and 25–75%
quantile bands. It starts at the unmerged baseline and follows the selected
merge budget. These lengths measure compression on the loaded training corpus,
not held-out performance. A histogram below shows moves per block for the
selected move library, counting each distinct block once, including single moves.

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
