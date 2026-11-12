"""Parity on reduced even cubes, plus the center-arrangement check.

A reduced cube maps to a 3x3 state (cube.reduce.reduce). On even N that 3x3 can be one no
real 3x3 could reach:

    OLL parity  the edge flip sum is odd (one edge looks flipped)
    PLL parity  edge and corner permutation parities differ (two edges look swapped)

Each is fixed by an algorithm that keeps the cube reduced and changes exactly one thing at
the 3x3 level. tests/test_parity.py checks both exactly, from their permutations:

    OLL_PARITY  2R2 B2 U2 2L U2 2R' U2 2R U2 F2 2R F2 2L' B2 2R2   flips the UF edge
    PLL_PARITY  2R2 U2 2R2 Uw2 2R2 2U2                              swaps the UF and UB edges

Because they're pure, they work wherever the cube is: flipping or swapping whichever edges
sit at UF / UB is enough to make the 3x3 reachable. They're written for the 4x4 and carried
to bigger even cubes with cube.reduce.lift_alg(..., base=4), which turns each 4x4 inner
layer as a block of N/2 - 1 layers. On odd cubes the middle wings behave like 3x3 edges,
so neither parity can happen there.

Even cubes have no fixed centers either. The center blocks must be in an arrangement some
whole-cube rotation turns into the color scheme (U/D, R/L and F/B opposite, with the same
handedness as the corners). A mirrored arrangement, such as red and orange swapped, can't
be solved as a 3x3; center_arrangement_ok reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from cube import reduce as rd
from cube import sim
from cube.sim import Move

OLL_PARITY = "2R2 B2 U2 2L U2 2R' U2 2R U2 F2 2R F2 2L' B2 2R2"
PLL_PARITY = "2R2 U2 2R2 Uw2 2R2 2U2"

_U, _R, _F, _D, _L, _B = range(6)
_SOLVED3 = sim.solved(3)
_ROTATIONS = tuple(
    sim.alg_perm([t for t in (a, b) if t], 3)
    for a in ("", "x", "x2", "x'", "z", "z'")
    for b in ("", "y", "y2", "y'")
)  # all 24 whole-cube rotations of the 3x3, as sticker permutations
VALID_ARRANGEMENTS = frozenset(tuple(_SOLVED3[p][4::9].tolist()) for p in _ROTATIONS)


@dataclass(frozen=True)
class Cubies:
    """Cubie-level 3x3 state; requires the centers in their solved positions.

    Slots follow cube.parity's corner and edge slot order. ``*_perm[slot]`` is the piece
    (by home slot) sitting there. Edge flip is 0 when the edge's U/D color (or F/B color,
    for an edge with neither U nor D) is on the slot's U/D (or F/B) sticker. Corner twist
    is how far clockwise the corner's U/D color sits from the slot's U/D sticker.
    """

    corner_perm: tuple[int, ...]
    corner_twist: tuple[int, ...]
    edge_perm: tuple[int, ...]
    edge_flip: tuple[int, ...]


@dataclass(frozen=True)
class ParityReport:
    centers_ok: bool
    oll_parity: bool = False
    pll_parity: bool = False
    edge_flip_sum: int | None = None
    corner_twist_sum: int | None = None  # mod 3; nonzero only for an impossible state
    corner_parity: int | None = None
    edge_parity: int | None = None

    @property
    def solvable_as_3x3(self) -> bool:
        return self.centers_ok and not self.oll_parity and not self.pll_parity and self.corner_twist_sum == 0


@lru_cache(maxsize=None)
def slots() -> tuple[np.ndarray, np.ndarray]:
    """3x3 sticker indices of the 8 corner slots (U/D sticker first, then clockwise) and the
    12 edge slots (U/D sticker first, or F/B for the middle-layer edges)."""
    pos, nrm = sim.sticker_positions(3)
    zeros = (pos == 0).sum(axis=1)
    cubies = {}
    for i in np.flatnonzero(zeros < 2):  # corners have no zero coordinate, edges one
        cubies.setdefault(tuple(pos[i]), []).append(int(i))
    corners, edges = [], []
    for stickers in cubies.values():
        if len(stickers) == 3:
            ud = next(i for i in stickers if i // 9 in (_U, _D))
            a, b = [i for i in stickers if i != ud]
            if np.linalg.det(np.stack([nrm[ud], nrm[a], nrm[b]])) > 0:  # make it clockwise
                a, b = b, a
            corners.append((ud, a, b))
        else:
            rank = {_U: 0, _D: 0, _F: 1, _B: 1, _R: 2, _L: 2}
            edges.append(tuple(sorted(stickers, key=lambda i: rank[i // 9])))
    corner_slots, edge_slots = np.array(corners), np.array(edges)
    corner_slots.flags.writeable = edge_slots.flags.writeable = False
    return corner_slots, edge_slots


@lru_cache(maxsize=None)
def _piece_index() -> tuple[dict, dict]:
    corner_slots, edge_slots = slots()
    corners = {tuple(sorted(_SOLVED3[s].tolist())): k for k, s in enumerate(corner_slots)}
    edges = {tuple(sorted(_SOLVED3[s].tolist())): k for k, s in enumerate(edge_slots)}
    return corners, edges


def cubies(state3: np.ndarray) -> Cubies:
    """Decode a 3x3 sticker state with solved-position centers into cubies."""
    if sim.cube_size(state3) != 3 or not np.array_equal(state3[4::9], np.arange(6)):
        raise ValueError("cubies() needs a 3x3 state with its centers in solved positions")
    corner_slots, edge_slots = slots()
    corner_ids, edge_ids = _piece_index()
    try:
        cc = state3[corner_slots]
        corner_perm = tuple(corner_ids[tuple(sorted(c.tolist()))] for c in cc)
        corner_twist = tuple(int(np.flatnonzero((c == _U) | (c == _D))[0]) for c in cc)
        ec = state3[edge_slots]
        edge_perm = tuple(edge_ids[tuple(sorted(e.tolist()))] for e in ec)
    except (KeyError, IndexError):
        raise ValueError("stickers don't form real 3x3 corners and edges") from None
    if len(set(corner_perm)) != 8 or len(set(edge_perm)) != 12:
        raise ValueError("a corner or edge appears twice")
    primary = _SOLVED3[edge_slots[:, 0]]  # each piece's reference color, by home slot
    edge_flip = tuple(int(e[0] != primary[p]) for e, p in zip(ec, edge_perm))
    return Cubies(corner_perm, corner_twist, edge_perm, edge_flip)


def _parity(perm: tuple[int, ...]) -> int:
    seen, cycles = set(), 0
    for start in range(len(perm)):
        if start not in seen:
            cycles += 1
            i = start
            while i not in seen:
                seen.add(i)
                i = perm[i]
    return (len(perm) - cycles) % 2


def center_colors(state: np.ndarray) -> np.ndarray:
    """Color of each face's center block (U R F D L B). Raises if a block is mixed."""
    n = sim.cube_size(state)
    proj = rd.projection(n)
    colors = []
    for face in range(6):
        block = state[proj == 9 * face + 4]
        if not (block == block[0]).all():
            raise ValueError(f"the {sim.FACES[face]} center block has more than one color")
        colors.append(int(block[0]))
    return np.array(colors)


def center_arrangement_ok(state: np.ndarray) -> bool:
    """True when the solved center blocks match the color scheme up to a rotation."""
    return tuple(center_colors(state).tolist()) in VALID_ARRANGEMENTS


def _with_centers_home(state3: np.ndarray) -> np.ndarray | None:
    """The 3x3 state rotated so each center is on its own face, or None if impossible."""
    for perm in _ROTATIONS:
        rotated = state3[perm]
        if np.array_equal(rotated[4::9], np.arange(6)):
            return rotated
    return None


def detect(state: np.ndarray) -> ParityReport:
    """Parity of a reduced NxN state (N >= 3). Raises if the state isn't reduced."""
    state3 = rd.reduce(state)
    rotated = _with_centers_home(state3)
    if rotated is None:
        return ParityReport(centers_ok=False)
    c = cubies(rotated)
    corner_parity, edge_parity = _parity(c.corner_perm), _parity(c.edge_perm)
    flips = sum(c.edge_flip)
    return ParityReport(
        centers_ok=True,
        oll_parity=flips % 2 == 1,
        pll_parity=corner_parity != edge_parity,
        edge_flip_sum=flips,
        corner_twist_sum=sum(c.corner_twist) % 3,
        corner_parity=corner_parity,
        edge_parity=edge_parity,
    )


@lru_cache(maxsize=None)
def parity_alg(kind: str, n: int) -> tuple[Move, ...]:
    """OLL_PARITY ('oll') or PLL_PARITY ('pll') as moves for an even NxN cube."""
    if n % 2 or n < 4:
        raise ValueError(f"parity algorithms are for even N >= 4, got N={n}")
    alg = {"oll": OLL_PARITY, "pll": PLL_PARITY}[kind]
    return tuple(rd.lift_alg(alg, n, base=4))


def fix_parity(state: np.ndarray) -> tuple[np.ndarray, list[Move]]:
    """Fix OLL and PLL parity on a reduced cube; returns (new state, moves applied).

    The result is still reduced and maps to a 3x3 state a real 3x3 can reach. Raises
    ValueError if the center arrangement is wrong or the state can't occur on a real cube.
    """
    n = sim.cube_size(state)
    report = detect(state)
    if not report.centers_ok:
        raise ValueError("the center blocks are in a mirrored or impossible arrangement")
    if report.corner_twist_sum:
        raise ValueError("a corner is twisted on its own, which no real cube allows")
    moves: list[Move] = []
    if report.oll_parity or report.pll_parity:
        if n % 2:
            raise ValueError(f"odd cubes can't have parity, so this {n}x{n} state can't occur")
        if report.oll_parity:
            moves += parity_alg("oll", n)
        if report.pll_parity:
            moves += parity_alg("pll", n)
    fixed = sim.apply_alg(state, moves)
    after = detect(fixed)
    if not after.solvable_as_3x3:
        raise RuntimeError(f"parity fix left {after}")
    return fixed, moves
