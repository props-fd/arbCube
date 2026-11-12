# NxN cube solving by reduction + a learned 3x3 policy (CS224W)

A solver for NxNxN Rubik's cubes using the reduction method: reduce the NxN to a 3x3,
then solve the 3x3 with a beam search guided by a move policy trained by imitation on
human CFOP solves (Kaggle `antbob/rubiks-cube-cfop-solutions`, v1). The policy is a
one-hot MLP baseline or a sticker-graph GNN (PyTorch Geometric).

**Status:** the 3x3 solver works end to end (below). For NxN, the projection to a 3x3,
the move lifting and the even-N parity fixes are done and tested. Center solving and
edge pairing are not written yet, so NxN cubes can't be solved end to end yet.

## Results

Short version: [`FINDINGS.md`](FINDINGS.md). Figures and details: [`results/RESULTS.md`](results/RESULTS.md).

### 3x3 solving: MLP policy + beam search

Every solution is replayed in the simulator before it is counted. Full scrambles are the
start states of solves in the held-out test split (split by solve).

| Beam width | Full scrambles solved | Mean length, face turns (human CFOP) | Time per scramble |
|---|---|---|---|
| 32 | 81.5% of 200 | 85.7 (62.1) | 0.04 s |
| 64 | 97.5% of 200 | 74.5 (62.0) | 0.10 s |
| 128 | 99.5% of 200 | 65.7 (62.1) | 0.16 s |
| 256 | **99.9% of 1,000** | **60.1** (61.7) | 0.20 s |

Depth suite (cubetransfer-style random quarter-turn walks, 10 per depth 1..20): 88.5%
solved at width 32 and 100% at width 256. Solutions are CFOP-like, so short scrambles
get long solutions: only 50% (width 32) to 61% (width 256) of depth 1..10 walks are
solved within 14 quarter turns, cubetransfer's 3x3 pass test.

### Next-move prediction (20,000 held-out test rows, 100,000 training rows)

| Model | Test top-1 | Cross | F2L | OLL | PLL | Params | CPU train time |
|---|---|---|---|---|---|---|---|
| Most common move (U) | 15.0% | | | | | 0 | 0 |
| One-hot MLP | **79.8%** | 61.0% | 72.3% | 90.3% | 99.5% | 438k | 42 s |
| GNN v2 (home-face feature, hidden 128) | 63.2% | 57.7% | 57.1% | 61.0% | 82.3% | 86k | 50 min |
| GNN v1 (hidden 64) | 54.7% | 37.3% | 49.6% | 51.7% | 78.7% | 22k | 48 min |

The GNN still underfits (final training loss 1.05, against 0.27 for the MLP) and
trails most on OLL / PLL. In the search it does much worse than its accuracy suggests:
GNN v2 at width 32 solves 13% of 200 full scrambles (MLP: 81.5%) and 39% of the depth
suite. The full numbers, with per-epoch histories, are in `results/`.

### Next to cubetransfer (in this repository)

cubetransfer trains on random walks from solved and reports a horizon of about 10
quarter turns on the 3x3 (no full scrambles solved). Imitation on full human solves
removes that horizon: the CFOP-trained policy solves essentially every full scramble,
but it is not aiming for short solutions. The two are complementary rather than
competing results, and they measure different things (see Conventions).

## Conventions: different from SPEC.md / cubetransfer

This folder was built separately and uses its own cube convention. It is self-contained:
nothing here imports from the rest of arbCube, and nothing outside it imports from here.

| | this folder (`cube/sim.py`) | cubetransfer `cube.py` / SPEC.md |
|---|---|---|
| State | flat `uint8`, length 6N², faces U R F D L B, each row-major in the Kociemba facelet layout | stickers sorted by position |
| Sticker value | index of the face whose color it is (U = 0 ... B = 5) | facet-id colors |
| Move permutation | `new = state[perm]`: `perm[i]` is where position i's sticker comes **from** | `perm[i]` is where sticker i goes **to** |
| Moves | any NxN notation (`R`, `Rw`, `3Rw`, SiGN slices `2R`, `M E S`, `x y z`) parsed to `Move(axis, lo, hi, turns)`; the 3x3 policy uses the 18 face turns | quarter turns in n^d (12 / 18 / 48 for 3x3x3 / 4x4x4 / 3x3x3x3) |
| Length metric | face turns (HTM) unless marked QTM | quarter turns (QTM) |
| Dimensions | 3D cubes, any N >= 2 | n^d, d = 3 or 4 |

What this means in practice:

- **States, permutations and checkpoints are not interchangeable.** Feeding one engine's
  states or permutations to the other gives wrong states silently, not an error (the
  array shapes match on the 3x3). A converter is a fixed sticker reordering, a color
  relabeling and a permutation inversion, checked by applying every move in both
  engines; none is written yet.
- **Metrics differ.** The depth suite here uses quarter-turn walks to match
  cubetransfer, but solve lengths above are face turns unless marked QTM, and "solved"
  here has no length cap (cubetransfer's bar caps it at 14 QTM).
- **Python packages.** This folder has its own top-level packages named `cube`, `data`,
  `models` and `search` and its own `pyproject.toml`, outside the root uv project. Run
  every command from inside this folder so these names don't collide with anything at
  the repository root (e.g. `arbcube`).
- **Data.** The parser reads the same Kaggle shards as `arbcube/data.py`, from this
  folder's `data/raw/` (copy or link pmn's `data/raw/cfop-v1/` files there). It settles
  the face order and the color-to-integer mapping that the root README lists as open
  questions, and checks them by replaying every solve to solved
  (`python -m data.parse_cfop --check-seq`; formats documented in `data/parse_cfop.py`).

## Layout

| Path | Owns |
|---|---|
| `cube/sim.py` | state format, move parsing, cached move permutations, solved checks |
| `cube/reduce.py`, `cube/parity.py` | NxN <-> 3x3 projection, move lifting, OLL/PLL parity detection and fixes |
| `data/parse_cfop.py`, `data/build_dataset.py` | Kaggle CFOP records -> verified move lists -> (state, next move, stage) splits |
| `models/graph.py`, `models/gnn.py`, `models/train.py` | sticker graphs, GNN + MLP, training and next-move evaluation |
| `search/beam.py`, `search/evaluate.py` | batched beam search, scramble suites, solve-rate report |
| `results/` | `RESULTS.md`, `figures/` (drawn by `plot.py`), `train_report.json`, `search_report.json`, the trained MLP (`models/mlp.pt`) |
| `tests/` | pytest suite (280 tests) |

`CLAUDE.md` documents every convention in detail.

## Run (Windows; use `.venv/bin/python` elsewhere)

```
python -m venv .venv
.venv\Scripts\python -m pip install "numpy>=1.26" "pytest>=8"
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install torch_geometric
.venv\Scripts\python -m pytest
.venv\Scripts\python -m data.parse_cfop --check-seq   # needs the shards in data/raw/
.venv\Scripts\python -m data.build_dataset            # writes artifacts/cfop_*.npz
.venv\Scripts\python -m models.train --model mlp      # or copy results/ into artifacts/
.venv\Scripts\python -m search.evaluate --model mlp --width 256
```

Generated data, checkpoints and logs go to `artifacts/` (gitignored); `results/` is a
copy of the reports and the MLP checkpoint behind the numbers above.
