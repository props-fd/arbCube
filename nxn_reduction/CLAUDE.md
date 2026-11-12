# CLAUDE.md

CS224W final project: solve NxNxN Rubik's cubes with graph neural networks using the
reduction method.

## Pipeline

1. **Centers** (`cube/reduce.py`): solve each face's center block (N >= 4).
2. **Edges** (`cube/reduce.py`, `cube/parity.py`): pair edge pieces into 3x3-style edges;
   fix OLL (edge flip) and PLL (edge swap) parity on even N.
3. **Project** the reduced cube to a 3x3 state, using the same state format with N = 3.
4. **3x3 solve** (`search/beam.py`): beam search guided by a move policy (`models/`)
   trained by imitation on CFOP solves (`data/`).

N = 3 skips steps 1-3.

## Layout

| Path | Owns |
|---|---|
| `cube/sim.py` | state format, move parsing, move permutations, solved checks |
| `cube/reduce.py` | NxN <-> 3x3 projection (`is_reduced`, `reduce`, `lift_state`, `lift_move`); center solving and edge pairing still to come |
| `cube/parity.py` | 3x3 cubie decoding, OLL/PLL parity detection and fixes, center-arrangement check |
| `data/parse_cfop.py` | CFOP reconstruction text -> move lists |
| `data/build_dataset.py` | replay solves into (state, next move) training pairs |
| `models/graph.py` | NxN state -> PyG sticker graph (size-free features, typed edges); move descriptions for the policy head |
| `models/gnn.py` | `StickerGNN` (shared GINE layer x T, value head, move-scoring policy head) and the `OneHotMLP` baseline |
| `models/train.py` | train / evaluate next-move prediction on the CFOP rows (CPU) |
| `search/beam.py` | batched beam search over the 18 face turns, ranked by policy log-prob |
| `search/evaluate.py` | depth and CFOP scramble suites, model loading, `search_report.json` |
| `tests/` | pytest suite |
| `notebooks/` | exploration only; nothing imports from here |

## State convention

- A state is a flat `numpy.uint8` array of length `6*N*N`.
- Face order is **U R F D L B** = face indices 0-5. Sticker index:
  `idx = face*N*N + row*N + col`.
- A sticker's value is the index of the face whose color it is (U=0 ... B=5), so the solved
  state is `np.repeat(np.arange(6, dtype=np.uint8), N*N)`.
- Each face is row-major as seen from outside, in the **Kociemba facelet layout** (the
  cross net below, folded into a cube). Shown for N = 3, numbered within each face:

  ```
              U0 U1 U2
              U3 U4 U5
              U6 U7 U8
     L0 L1 L2 F0 F1 F2 R0 R1 R2 B0 B1 B2
     L3 L4 L5 F3 F4 F5 R3 R4 R5 B3 B4 B5
     L6 L7 L8 F6 F7 F8 R6 R7 R8 B6 B7 B8
              D0 D1 D2
              D3 D4 D5
              D6 D7 D8
  ```

  | Face | row 0 is next to | col 0 is next to |
  |---|---|---|
  | U | B | L |
  | R | U | F |
  | F | U | L |
  | D | F | L |
  | L | U | B |
  | B | U | R |

- A 3x3 state, including the projection of a reduced cube, is the same format with N = 3,
  so a 3x3 is interchangeable with Kociemba facelet strings (`"UUU...BBB"`).
- **Solved** = every face is one color. On even N there are no fixed centers, so any
  whole-cube orientation counts as solved (`sim.is_solved`).
- Functions take a single state or a batch (`[..., 6*N*N]`) and return new arrays; they
  never mutate their inputs.

## Move notation

Every turn is clockwise as seen looking at the named face. `R'` = counterclockwise,
`R2` = half turn (`R2'` is accepted and equals `R2`). Curly apostrophes are accepted.

| Notation | Meaning |
|---|---|
| `R` | outer layer only |
| `Rw`, `r` | the 2 outer layers |
| `nRw`, `nr` (2 <= n <= N) | the n outer layers; `NRw` turns the whole cube |
| `nR` (1 <= n <= N) | only layer n counted from R (SiGN slice style); `NR` == `L'` |
| `M` `E` `S` | odd N only: the middle layer, turning like L / D / F |
| `x` `y` `z` | whole-cube rotations, turning like R / U / F |

Lowercase `r` always means `Rw` (as in 3x3 CFOP reconstructions), never an inner slice.
Single inner slices are written only as `nR`.

## Simulator (`cube/sim.py`)

- Every move is a precomputed, cached permutation index array: `new = state[..., perm]`.
  `perm[i]` is the index of the sticker that lands on position `i`. Applying `m1` then
  `m2` is the single permutation `p1[p2]`.
- Every notation parses to a canonical `Move(axis, lo, hi, turns)`:
  - `axis` 0/1/2 = x/y/z, pointing toward R/U/F;
  - layers `lo..hi` inclusive, 0 = outermost on the R/U/F side;
  - `turns` = clockwise quarter turns (1-3) as seen from the R/U/F side.

  Equivalent notations give the same `Move` (on a 4x4, `4R` == `L'` and `4Rw` == `x`),
  which is also the action space the GNN sees.
- Supports any N >= 2.

## Reduction (`cube/reduce.py`, `cube/parity.py`)

- Every NxN sticker merges into one 3x3 sticker (`reduce.projection(n)`): corners to
  corners, the N-2 wings of an edge to that edge's sticker on the same face, all center
  stickers to the face center. Reduced = every merged group is one color.
- `lift_move(m, n)` maps 3x3 layers onto NxN layers: outer stays outer, the 3x3 middle
  layer becomes all inner layers. With `base=4` (even N) each 4x4 inner layer becomes a
  block of N/2 - 1 layers; that is how the 4x4 parity algorithms run on bigger even cubes.
- Parity lives on the 3x3 level of a reduced even cube: OLL parity = odd edge flip sum,
  PLL parity = corner and edge permutation parities differ. `parity.fix_parity` applies
  `OLL_PARITY` (flips UF only) and/or `PLL_PARITY` (swaps UF and UB only); both are
  verified to keep the cube reduced and to change nothing else at the 3x3 level.
- Even cubes have no fixed centers: `parity.center_arrangement_ok` checks the center
  blocks are a rotation of the color scheme (a mirrored arrangement can't be solved).
- Edge orientation: 0 when the edge's U/D color (F/B color for middle-layer edges) is on
  the slot's U/D (F/B) sticker. Corner twist counts clockwise from the slot's U/D sticker.

## Data

- Raw data: the Kaggle dataset `antbob/rubiks-cube-cfop-solutions` v1 in `data/raw/`
  (gitignored, about 755 MB; 11 shards `training.N` / `training.seq.N`, N = 0-9 and 99,
  10,000 solves each). The file formats are documented at the top of `data/parse_cfop.py`.
- Generated datasets, reports, checkpoints and logs go in `artifacts/` (gitignored). Code
  under `data/` never writes into `data/`.
- `parse_cfop` converts states to our convention (yellow U, orange R, green F, white D,
  red L, blue B) and normalizes solutions to the 18 face turns `FACE_TURNS`, removing
  x/y/z rotations by relabeling the moves that follow them, so training states always have
  the U center on top and the F center in front. The v1 data contains only face turns.
- `build_dataset` writes `artifacts/cfop_{train,val,test}.npz` (90/5/5 by `solve_id`,
  seed 224) with `state`, `next_move`, `moves_remaining`, `cfop_stage`, `solve_id`, plus
  `move_names` and `stage_names`. Stages are relative to the white cross on D.

## Models (`models/`)

- PyTorch Geometric, CPU only. Graph nodes are stickers in state order. Node features are
  size-free: face one-hot, (u, v) in [-1, 1], piece size one-hot (`STATIC_DIM`, the same
  for every state of a given N), plus the home face (`HOME_DIM`): the share of the
  sticker's color's center stickers on each face, one-hot on a 3x3. Colors themselves are
  never features, so the GNN is invariant to relabeling colors. Edge types (one-hot
  `edge_attr`): surface, same piece, same color; `color="center"` (default) keeps only
  same-color pairs that include a center sticker, `"all"` keeps every pair.
- Graph options (`home`, `color`) are passed to `batch_states` and must match between
  training and prediction; `StickerGNN(node_dim=graph.node_dim(home))`. `gnn_v1` in the
  report is the first version (`--no-home --color-edges all --hidden 64`).
- `graph.batch_states` builds one `Data` for a batch of same-N states directly (no
  per-graph objects); `nodes_per_graph` lets the model reshape to `(B, nodes, hidden)`.
- The policy head scores any list of moves via `graph.move_tensors(moves, n)`: the mean
  embedding of the stickers each move turns, plus its turn one-hot as seen from the
  layer's own side. Train targets index `data.parse_cfop.FACE_TURNS`.
- `models.train` trains on a seeded random subset of `cfop_train.npz` and scores fixed
  seeded subsets of val and test; results go to `artifacts/train_report.json`.

## Search (`search/`)

- `beam.beam_search(states, policy, width, max_moves)` searches many 3x3 scrambles at
  once. `policy` maps `(L, 54)` uint8 states to `(L, 18)` log-probs over `FACE_TURNS`, so
  the search needs no torch. Children are ranked by cumulative log-prob (the cubetransfer
  protocol); a move never turns the previous move's face, and states already in that
  scramble's beam are dropped. It stops at the first depth with a solved child.
- `beam.verify` replays every solution in the simulator; `evaluate` calls it before
  counting anything.
- `evaluate` suites: `depth` = seeded quarter-turn random walks, depth 1..K, plus `bar`
  (depth 1..10 solved within 14 QTM, cubetransfer's 3x3 pass test); `cfop` = start states
  of seeded test-split solves, against the human solution length. Lengths are reported in
  face turns (HTM) and quarter turns (QTM).

## Commands (Windows, from the repo root)

```
python -m venv .venv
.venv\Scripts\python -m pip install "numpy>=1.26" "pytest>=8"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m data.parse_cfop --check-seq   # verify every solve, ~70 s
.venv\Scripts\python -m data.build_dataset            # write the .npz splits
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install torch_geometric
.venv\Scripts\python -m models.train                  # MLP + GNN, 100k rows, CPU
.venv\Scripts\python -m search.evaluate --model mlp --width 128   # solve rates
```

## Notes

- `data` and `models` are generic top-level package names. Run from the repo root (pytest
  adds it via `pythonpath`); don't pip-install this project.
- The GNN framework is PyTorch Geometric (`ml` extra in pyproject). Tests that need torch
  skip when it isn't installed.
