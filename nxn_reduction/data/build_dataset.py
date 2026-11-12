"""Build next-move training rows from the verified CFOP solves, split 90/5/5 by solve_id.

One row per move of every solve that verifies in data.parse_cfop:

    state            (54,) uint8   the 3x3 before the move, in our sticker convention
    next_move        uint8         index into FACE_TURNS (the 18 face turns)
    moves_remaining  uint16        moves left in the solve, counting this one (>= 1)
    cfop_stage       uint8         index into STAGES: the stage this move belongs to
    solve_id         int64         data.parse_cfop.solve_id(shard, record index)

Solves are shuffled once with a fixed seed and split 90/5/5 into train / val / test by
solve_id, so all rows of a solve land in the same split. Writes artifacts/cfop_{split}.npz
(each also holds move_names and stage_names) and artifacts/cfop_dataset_summary.json.

Stages follow the white cross on D, which is where the solver builds it. A move belongs to
the first milestone not yet reached, where each milestone counts only once the previous
one has been reached. The solver merges same-face moves across stage boundaries (F' + F'
is written F2), so milestones are also checked partway through each move; a merged move
counts toward the stage it finishes. Where the two moves cancelled out entirely, the
boundary is recovered from the state one face turn away from the milestone (see
stage_labels).
    cross  D cross (4 D edges and the D center) solved
    f2l    first two layers solved (all of D plus rows 1-2 of F, R, B, L)
    oll    first two layers solved and U all one color
    pll    everything after OLL, including the final U turn

Run from the repo root:
    .venv\\Scripts\\python -m data.build_dataset [--exclude-shard 99]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from cube import sim
from data.parse_cfop import (
    ARTIFACTS,
    FACE_TURN_INDEX,
    FACE_TURNS,
    RAW_DIR,
    SOLVED,
    iter_records,
    iter_solves,
    shard_number,
    shard_paths,
    solve_id,
    verify,
)

STAGES = ("cross", "f2l", "oll", "pll")
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.90, 0.05, 0.05)
SEED = 224

_D, _SIDES = 3 * 9, [9 * sim.FACES.index(f) for f in "RFLB"]
CROSS_STICKERS = np.array([_D + 4, _D + 1, _D + 3, _D + 5, _D + 7] + [s + 7 for s in _SIDES])
F2L_STICKERS = np.array(list(range(_D, _D + 9)) + [s + k for s in _SIDES for k in range(3, 9)])
U_STICKERS = np.arange(9)
_MOVE_PERMS = np.stack([sim.move_perm(m, 3) for m in FACE_TURNS])


def replay(scramble: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """States before each move and after the last: (len(moves) + 1, 54) uint8."""
    states = np.empty((len(moves) + 1, scramble.size), dtype=np.uint8)
    states[0] = scramble
    for t, m in enumerate(moves):
        states[t + 1] = states[t][_MOVE_PERMS[m]]
    return states


def _milestones(states: np.ndarray) -> np.ndarray:
    """(3, len(states)) booleans: cross, f2l and oll reached at each state."""
    cross = (states[..., CROSS_STICKERS] == SOLVED[CROSS_STICKERS]).all(axis=-1)
    f2l = (states[..., F2L_STICKERS] == SOLVED[F2L_STICKERS]).all(axis=-1)
    oll = f2l & (states[..., U_STICKERS] == SOLVED[U_STICKERS]).all(axis=-1)
    return np.stack([cross, f2l, oll])


def _inside_moves(states: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """(len(moves), 2, 54): the states partway through each move.

    The solver merges same-face moves across stage boundaries (the last F2L move F' and
    the first OLL move F' are written as one F2), so a stage can end partway through a
    token. A token of k quarter turns passes through the states after a quarter turns
    for the two values of a in 1..3 other than k.
    """
    quarter = _MOVE_PERMS[(moves // 3) * 3]  # the clockwise quarter turn of each move's face
    k = np.array([1, 3, 2])[moves % 3]  # quarter turns in each move: "", "'", "2"
    out = np.empty((len(moves), 2, states.shape[1]), dtype=np.uint8)
    for t in range(len(moves)):
        partial = states[t]
        slot = 0
        for a in (1, 2, 3):
            partial = partial[quarter[t]]
            if a != k[t]:
                out[t, slot] = partial
                slot += 1
    return out


HIDDEN_GAP = 3


def stage_labels(states: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """CFOP stage of each move, given the states before every move and after the last.

    Milestone m is reached at row t when it holds at state t, or partway through move
    t - 1 (that merged move then counts toward the earlier stage). Each milestone counts
    only from the row where the previous one was reached.

    When the last move of a stage and the first of the next cancel out (X then X'), the
    milestone state never appears. That shows up as a state HIDDEN_GAP or more moves
    before the detected boundary that is already one face turn from the milestone; the
    boundary moves back to that state. A gap of 1 is the stage's own last move.
    """
    at_state = _milestones(states)
    inside = _milestones(_inside_moves(states, moves)).any(axis=-1)  # (3, len(moves))
    near = _milestones(states[:, _MOVE_PERMS]).any(axis=-1)  # (3, len(states)): one turn away
    if not at_state[2, -1]:
        raise ValueError("the final state does not complete OLL; is the solve verified?")
    reached = at_state.copy()
    reached[:, 1:] |= inside
    t_prev, bounds = 0, []
    for m in range(3):
        t_m = t_prev + int(np.argmax(reached[m, t_prev:]))
        earlier = np.flatnonzero(near[m, t_prev:t_m])
        if len(earlier) and t_m - (t_prev + int(earlier[0])) >= HIDDEN_GAP:
            t_m = t_prev + int(earlier[0])
        bounds.append(t_m)
        t_prev = t_m
    t = np.arange(len(moves))
    return sum((t >= b).astype(np.uint8) for b in bounds).astype(np.uint8)


def assign_splits(ids: Iterable[int], seed: int = SEED) -> dict[int, str]:
    """Shuffle solve ids once and cut them 90/5/5 into train / val / test."""
    ids = np.array(sorted(ids), dtype=np.int64)
    order = np.random.default_rng(seed).permutation(len(ids))
    n_val = round(SPLIT_FRACTIONS[1] * len(ids))
    n_test = round(SPLIT_FRACTIONS[2] * len(ids))
    split = {}
    for rank, i in enumerate(order):
        split[int(ids[i])] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return split


def all_solve_ids(raw_dir: Path, shards: set[int]) -> list[int]:
    ids = []
    for path in shard_paths(raw_dir):
        shard = shard_number(path)
        if shard in shards:
            ids += [solve_id(shard, i) for i, _ in enumerate(iter_records(path))]
    return ids


def build(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = ARTIFACTS,
    *,
    exclude_shards: Iterable[int] = (),
    limit_per_shard: int | None = None,
    seed: int = SEED,
) -> dict:
    shards = {shard_number(p) for p in shard_paths(raw_dir)} - set(exclude_shards)
    split_of = assign_splits(all_solve_ids(raw_dir, shards), seed)
    parts = {s: {"state": [], "next_move": [], "moves_remaining": [], "cfop_stage": [], "solve_id": []} for s in SPLITS}
    solves, skipped = Counter(), []
    stage_moves = {s: [] for s in STAGES}
    for solve in iter_solves(raw_dir, shards=shards, limit_per_shard=limit_per_shard):
        if not all(verify(solve)):
            skipped.append(solve.solve_id)
            continue
        moves = np.array([FACE_TURN_INDEX[m] for m in solve.moves], dtype=np.uint8)
        states = replay(solve.scramble, moves)
        stages = stage_labels(states, moves)
        n = len(moves)
        split = split_of[solve.solve_id]
        part = parts[split]
        part["state"].append(states[:-1])
        part["next_move"].append(moves)
        part["moves_remaining"].append(np.arange(n, 0, -1, dtype=np.uint16))
        part["cfop_stage"].append(stages)
        part["solve_id"].append(np.full(n, solve.solve_id, dtype=np.int64))
        solves[split] += 1
        counts = np.bincount(stages, minlength=len(STAGES))
        for k, name in enumerate(STAGES):
            stage_moves[name].append(int(counts[k]))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "shards": sorted(shards), "skipped_unverified": skipped, "splits": {}}
    for split in SPLITS:
        arrays = {k: np.concatenate(v) if v else np.empty(0) for k, v in parts[split].items()}
        if not parts[split]["state"]:
            arrays["state"] = np.empty((0, 54), dtype=np.uint8)
        path = out_dir / f"cfop_{split}.npz"
        np.savez_compressed(path, **arrays, move_names=np.array(FACE_TURNS), stage_names=np.array(STAGES))
        stage_counts = np.bincount(arrays["cfop_stage"].astype(np.int64), minlength=len(STAGES))
        summary["splits"][split] = {
            "file": path.name,
            "solves": solves[split],
            "rows": int(len(arrays["next_move"])),
            "rows_per_stage": dict(zip(STAGES, map(int, stage_counts))),
            "megabytes": round(path.stat().st_size / 2**20, 1),
        }
        parts[split] = None  # free memory before the next split
    summary["moves_per_stage"] = {
        name: {"mean": round(float(np.mean(v)), 2), "min": int(np.min(v)), "max": int(np.max(v)),
               "solves_with_none": int(np.sum(np.array(v) == 0))}
        for name, v in stage_moves.items() if v
    }  # fmt: skip
    (out_dir / "cfop_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--exclude-shard", type=int, action="append", default=[], help="e.g. 99; repeatable")
    parser.add_argument("--limit-per-shard", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    summary = build(
        args.raw_dir, args.out_dir, exclude_shards=args.exclude_shard,
        limit_per_shard=args.limit_per_shard, seed=args.seed,
    )  # fmt: skip
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped_unverified"}, indent=2))
    print(f"skipped (did not verify): {len(summary['skipped_unverified'])}")


if __name__ == "__main__":
    main()
