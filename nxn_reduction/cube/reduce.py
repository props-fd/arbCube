"""Reduction: treat an NxN cube (N >= 3) as a big 3x3.

Every NxN sticker merges into one 3x3 sticker: corners into corners, the N-2 wings along
an edge into that edge's sticker on the same face, and every center sticker into its
face's center. A state is reduced when all NxN stickers that merge into the same 3x3
sticker share one color, which is the same as every center block being one color and
every edge group being matched. ``reduce`` then reads one sticker per group, and
``lift_state`` copies each 3x3 sticker back out to its whole group.

``lift_move`` maps a move of a smaller "base" cube onto the NxN layers: outer layers stay
outer, and the inner layers are split evenly among the base cube's inner layers. With a
3x3 base, a 3x3 slice turn becomes all inner NxN layers on that axis turning together.
With a 4x4 base (even N only) it carries 4x4 algorithms, such as the parity algorithms,
to any even cube. Lifted moves keep a reduced cube reduced, so
``reduce(apply(lift_state(s), lift_move(m))) == apply(s, m)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Union

import numpy as np

from cube import sim
from cube.sim import Move

MoveLike = Union[Move, str]


def _check_size(n: int) -> None:
    if n < 3:
        raise ValueError(f"reduction needs N >= 3, got N={n}")


@lru_cache(maxsize=None)
def projection(n: int) -> np.ndarray:
    """Read-only: for every NxN sticker, the index of the 3x3 sticker it merges into."""
    _check_size(n)
    pos, nrm = sim.sticker_positions(n)
    m = n - 1
    merged = np.where(pos == m, 2, np.where(pos == -m, -2, 0))  # outer layers stay outer
    small_pos, small_nrm = sim.sticker_positions(3)
    index = {tuple(p): i for i, p in enumerate((small_pos + small_nrm).tolist())}
    proj = np.array([index[tuple(p)] for p in (merged + nrm).tolist()], dtype=np.intp)
    proj.flags.writeable = False
    return proj


@lru_cache(maxsize=None)
def _representatives(n: int) -> np.ndarray:
    """One NxN sticker for each of the 54 3x3 stickers."""
    _, first = np.unique(projection(n), return_index=True)
    first.flags.writeable = False
    return first


def is_reduced(state: np.ndarray) -> bool | np.ndarray:
    """True where every center block is one color and every edge group is matched."""
    n = sim.cube_size(state)
    _check_size(n)
    merged = state[..., _representatives(n)][..., projection(n)]
    result = (state == merged).all(axis=-1)
    return bool(result) if result.ndim == 0 else result


def reduce(state: np.ndarray, *, check: bool = True) -> np.ndarray:
    """The 3x3 state of a reduced NxN state (or batch). Raises if a state isn't reduced."""
    n = sim.cube_size(state)
    _check_size(n)
    if check and not np.all(is_reduced(state)):
        raise ValueError("state is not reduced: a center block or edge group has mixed colors")
    return state[..., _representatives(n)]


def lift_state(state3: np.ndarray, n: int) -> np.ndarray:
    """The reduced NxN state (or batch) whose groups carry the colors of a 3x3 state."""
    if sim.cube_size(state3) != 3:
        raise ValueError("lift_state takes a 3x3 state")
    return state3[..., projection(n)]


@lru_cache(maxsize=None)
def layer_groups(base: int, n: int) -> tuple[tuple[int, int], ...]:
    """NxN layer range (first, last) for each layer of the base cube, 0 = R/U/F side."""
    if base < 3 or n < base or (n - 2) % (base - 2):
        raise ValueError(
            f"can't lift {base}x{base} moves to N={n}: its {n - 2} inner layers "
            f"don't split evenly into {base - 2} groups"
        )
    size = (n - 2) // (base - 2)
    inner = tuple((1 + k * size, (k + 1) * size) for k in range(base - 2))
    return ((0, 0), *inner, (n - 1, n - 1))


def lift_move(move: MoveLike, n: int, *, base: int = 3) -> Move:
    """The NxN move that turns the layers a base-cube move turns.

    With base 3: an outer turn stays an outer-layer turn, a slice turn becomes all inner
    layers on that axis together, a wide turn becomes everything but the far outer layer.
    """
    if isinstance(move, str):
        move = sim.parse_move(move, base)
    groups = layer_groups(base, n)
    return Move(move.axis, groups[move.lo][0], groups[move.hi][1], move.turns)


def lift_alg(alg: str | Iterable[MoveLike], n: int, *, base: int = 3) -> list[Move]:
    """lift_move for every move of an algorithm written for the base cube."""
    return [lift_move(m, n, base=base) for m in sim.parse_alg(alg, base)]
