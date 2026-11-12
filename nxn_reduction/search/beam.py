"""Batched beam search over the 18 3x3 face turns, guided by a move policy.

Each step scores every child of every beam state by its path's cumulative policy
log-probability, the same ranking as the team's cubetransfer protocol, and keeps the
best ``width`` children per scramble. Many scrambles are searched together, so every
step makes one batched policy call for all of their beam states.

    policy(states (L, 54) uint8) -> (L, 18) log-probabilities over FACE_TURNS

Pruning: a move never turns the face the previous move turned (R R' and R R2 are never
useful), and a child is dropped if its state was already in that scramble's beam at any
depth, or appears twice in the same step (the higher-scoring copy is kept). A scramble
stops at the first depth where some child is solved; that child's path is returned.
States are 3x3 in the project convention with U on top and F in front; face turns never
move the centers, so solved means equal to sim.solved(3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cube import sim
from data.parse_cfop import FACE_TURNS

Policy = Callable[[np.ndarray], np.ndarray]

MOVE_PERMS = np.stack([sim.move_perm(m, 3) for m in FACE_TURNS])  # (18, 54)
FACE_OF = np.arange(len(FACE_TURNS)) // 3
SOLVED = sim.solved(3)
_HASH = np.random.default_rng(224).integers(1, 2**63, size=54, dtype=np.uint64)


@dataclass
class SearchResult:
    solutions: list[list[int] | None]  # FACE_TURNS indices per scramble, None = not found
    nodes: np.ndarray  # policy evaluations spent on each scramble
    seconds: float

    @property
    def solved(self) -> np.ndarray:
        return np.array([s is not None for s in self.solutions])


def state_hash(states: np.ndarray) -> np.ndarray:
    """64-bit hash of each 3x3 state (..., 54) -> (...) uint64."""
    return ((states.astype(np.uint64) + np.uint64(1)) * _HASH).sum(axis=-1, dtype=np.uint64)


def tokens(solution: list[int]) -> list[str]:
    return [FACE_TURNS[m] for m in solution]


def qtm(solution: list[int]) -> int:
    """Length in quarter turns (a half turn counts 2)."""
    return sum(2 if m % 3 == 2 else 1 for m in solution)


def verify(states: np.ndarray, solutions: list[list[int] | None]) -> None:
    """Raise unless every found solution solves its scramble when replayed in the simulator."""
    for k, (state, sol) in enumerate(zip(states, solutions)):
        if sol is not None and not sim.is_solved(sim.apply_alg(state, tokens(sol))):
            raise AssertionError(f"solution {k} does not solve its scramble: {' '.join(tokens(sol))}")


def beam_search(
    states: np.ndarray, policy: Policy, width: int = 32, max_moves: int = 150, overfetch: int = 3
) -> SearchResult:
    """Search every scramble in ``states`` (S, 54) at once; see the module docstring.

    ``overfetch`` * ``width`` best children per scramble are generated each step, so that
    ``width`` usually survive deduplication.
    """
    start = time.perf_counter()
    states = np.atleast_2d(np.asarray(states, dtype=np.uint8))
    s_count, moves = len(states), len(FACE_TURNS)
    rows = np.arange(s_count)[:, None]
    beam = np.zeros((s_count, width, 54), dtype=np.uint8)
    beam[:, 0] = states
    logp = np.full((s_count, width), -np.inf)
    logp[:, 0] = 0.0
    last = np.full((s_count, width), -1)  # last move of each beam path
    history: list[tuple[np.ndarray, np.ndarray]] = []  # (parent slot, move) of each beam, per depth
    solutions: list[list[int] | None] = [[] if sim.is_solved(s) else None for s in states]
    active = np.array([sol is None for sol in solutions])
    visited = [{int(h)} for h in state_hash(states)]
    nodes = np.zeros(s_count, dtype=np.int64)
    fetch = min(moves * width, overfetch * width)

    for _ in range(max_moves):
        if not active.any():
            break
        live = active[:, None] & np.isfinite(logp)
        where = np.nonzero(live)
        scores = np.full((s_count, width, moves), -np.inf)
        scores[where] = logp[where][:, None] + policy(beam[where])
        np.add.at(nodes, where[0], 1)
        scores[(FACE_OF == FACE_OF[np.maximum(last, 0)][..., None]) & (last >= 0)[..., None]] = -np.inf
        scores = scores.reshape(s_count, width * moves)

        top = np.argpartition(-scores, fetch - 1, axis=1)[:, :fetch]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), axis=1, kind="stable"), 1)
        child_logp = np.take_along_axis(scores, top, 1)  # (S, fetch), best first
        parent, move = top // moves, top % moves
        child = np.take_along_axis(beam[rows, parent], MOVE_PERMS[move], axis=2)
        valid = np.isfinite(child_logp) & active[:, None]
        hashes = state_hash(child)
        done = (child == SOLVED).all(-1) & valid

        new_beam = np.zeros_like(beam)
        new_logp = np.full_like(logp, -np.inf)
        new_last = np.full_like(last, -1)
        new_parent = np.full_like(last, -1)
        for s in np.nonzero(active)[0]:
            if done[s].any():
                k = int(np.argmax(done[s]))  # best-scoring solved child
                path, slot = [int(move[s, k])], int(parent[s, k])
                for parents, moved in reversed(history):
                    path.append(int(moved[s, slot]))
                    slot = int(parents[s, slot])
                solutions[s] = path[::-1]
                active[s] = False
                continue
            kept, seen = 0, visited[s]
            for k in np.nonzero(valid[s])[0]:
                h = int(hashes[s, k])
                if h in seen:
                    continue
                seen.add(h)
                new_beam[s, kept], new_logp[s, kept] = child[s, k], child_logp[s, k]
                new_last[s, kept], new_parent[s, kept] = move[s, k], parent[s, k]
                kept += 1
                if kept == width:
                    break
            if kept == 0:
                active[s] = False  # nothing new left to explore
        history.append((new_parent, new_last))
        beam, logp, last = new_beam, new_logp, new_last

    return SearchResult(solutions, nodes, time.perf_counter() - start)
