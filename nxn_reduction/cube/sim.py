"""Vectorized NxNxN cube simulator (any N >= 2).

State: a flat uint8 array of 6*N*N stickers, faces U R F D L B, each face row-major as seen
from outside in the Kociemba facelet layout; a sticker's value is the index of the face
whose color it is. See CLAUDE.md for the full convention and notation table.

Every move is a precomputed permutation index array, applied as ``state[..., perm]``.
``perm[i]`` is the index of the sticker that lands on position ``i``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from math import isqrt
from typing import Iterable, Union

import numpy as np

__all__ = [
    "FACES",
    "Move",
    "solved",
    "cube_size",
    "is_solved",
    "parse_move",
    "parse_alg",
    "move_perm",
    "alg_perm",
    "apply_move",
    "apply_alg",
    "invert_token",
    "invert_alg",
    "move_tokens",
    "sticker_positions",
]

FACES = "URFDLB"


@dataclass(frozen=True)
class Move:
    """Canonical move: turn layers ``lo..hi`` (inclusive) of ``axis`` by ``turns``.

    ``axis`` 0/1/2 = x/y/z, pointing toward R/U/F. Layer 0 is the outermost layer on the
    R/U/F side. ``turns`` is 1-3 clockwise quarter turns as seen from the R/U/F side.
    """

    axis: int
    lo: int
    hi: int
    turns: int

    def inverse(self) -> Move:
        return Move(self.axis, self.lo, self.hi, 4 - self.turns)


MoveLike = Union[Move, str]
Alg = Union[str, Iterable[MoveLike]]

# face letter -> (axis, side); side +1 is the R/U/F side of the axis.
_FACE_AXIS = {"R": (0, 1), "L": (0, -1), "U": (1, 1), "D": (1, -1), "F": (2, 1), "B": (2, -1)}
_MIDDLE = {"M": "L", "E": "D", "S": "F"}  # middle slice turns like this face
_ROTATION = {"x": "R", "y": "U", "z": "F"}  # whole-cube rotation turns like this face

# One clockwise quarter turn seen from the + side of each axis (-90 degrees about +axis).
_QUARTER = (
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
)

_TOKEN = re.compile(r"(\d*)([URFDLBurfdlbMESxyz])(w?)(2'|2|'|)")
_TURNS = {"": 1, "'": 3, "2": 2, "2'": 2}


def _check_n(n: int) -> None:
    if n < 2:
        raise ValueError(f"cube size must be >= 2, got {n}")


def _normalize(token: str) -> str:
    return token.strip().replace("’", "'").replace("′", "'")


def solved(n: int) -> np.ndarray:
    """The solved state of an NxNxN cube."""
    _check_n(n)
    return np.repeat(np.arange(6, dtype=np.uint8), n * n)


def cube_size(state: np.ndarray) -> int:
    """N for a state (or batch of states) of length 6*N*N."""
    size = state.shape[-1]
    n = isqrt(size // 6)
    if n < 2 or 6 * n * n != size:
        raise ValueError(f"state length {size} is not 6*N*N for any N >= 2")
    return n


def is_solved(state: np.ndarray) -> bool | np.ndarray:
    """True where every face is a single color (any whole-cube orientation counts)."""
    n = cube_size(state)
    faces = state.reshape(*state.shape[:-1], 6, n * n)
    result = (faces == faces[..., :1]).all(axis=(-2, -1))
    return bool(result) if result.ndim == 0 else result


@lru_cache(maxsize=None)
def parse_move(token: str, n: int) -> Move:
    """Parse one move in standard / SiGN notation for an NxNxN cube into its canonical Move."""
    _check_n(n)
    match = _TOKEN.fullmatch(_normalize(token))
    if not match:
        raise ValueError(f"unrecognized move {token!r}")
    prefix, letter, wide, suffix = match.groups()
    count = int(prefix) if prefix else None
    turns = _TURNS[suffix]

    if letter in _ROTATION or letter in _MIDDLE:
        if prefix or wide:
            raise ValueError(f"{letter} takes no layer count or 'w': {token!r}")
        if letter in _ROTATION:
            axis, side = _FACE_AXIS[_ROTATION[letter]]
            lo, hi = 0, n - 1
        else:
            if n % 2 == 0:
                raise ValueError(f"{letter} needs an odd cube size, got N={n}")
            axis, side = _FACE_AXIS[_MIDDLE[letter]]
            lo = hi = (n - 1) // 2
    else:
        if letter.islower():  # r == Rw, 3r == 3Rw
            if wide:
                raise ValueError(f"lowercase moves are already wide: {token!r}")
            letter, wide = letter.upper(), "w"
        axis, side = _FACE_AXIS[letter]
        if wide:
            depth = 2 if count is None else count
            if not 2 <= depth <= n:
                raise ValueError(f"wide move {token!r} needs 2 <= layers <= {n}")
            first, last = 0, depth - 1
        else:
            layer = 1 if count is None else count
            if not 1 <= layer <= n:
                raise ValueError(f"slice move {token!r} needs 1 <= layer <= {n}")
            first = last = layer - 1
        # Layers counted from the L/D/B side map to the far end of the axis.
        lo, hi = (first, last) if side > 0 else (n - 1 - last, n - 1 - first)

    if side < 0:  # clockwise from L/D/B is counterclockwise from R/U/F
        turns = 4 - turns
    return Move(axis, lo, hi, turns)


def parse_alg(alg: Alg, n: int) -> list[Move]:
    """Parse a whitespace-separated algorithm (or a sequence of tokens / Moves)."""
    items = alg.split() if isinstance(alg, str) else alg
    return [m if isinstance(m, Move) else parse_move(m, n) for m in items]


@lru_cache(maxsize=None)
def _geometry(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sticker cubie position and face normal, plus a lookup from sticker point to index.

    Coordinates are doubled and centered: a cubie coordinate is one of -(N-1), ..., N-1 in
    steps of 2, and a sticker's point (cubie position + normal) is unique.
    """
    m = n - 1
    a = 2 * np.arange(n, dtype=np.int64) - m
    row = np.repeat(a, n)  # a[row] for each sticker of a face, row-major
    col = np.tile(a, n)  # a[col]
    edge = np.full(n * n, m, dtype=np.int64)
    layout = [  # (x, y, z) per face, then its outward normal
        ((col, edge, row), (0, 1, 0)),  # U: row 0 next to B, col 0 next to L
        ((edge, -row, -col), (1, 0, 0)),  # R: row 0 next to U, col 0 next to F
        ((col, -row, edge), (0, 0, 1)),  # F: row 0 next to U, col 0 next to L
        ((col, -edge, -row), (0, -1, 0)),  # D: row 0 next to F, col 0 next to L
        ((-edge, -row, col), (-1, 0, 0)),  # L: row 0 next to U, col 0 next to B
        ((-col, -row, -edge), (0, 0, -1)),  # B: row 0 next to U, col 0 next to R
    ]
    pos = np.concatenate([np.stack(xyz, axis=1) for xyz, _ in layout])
    nrm = np.repeat(np.array([normal for _, normal in layout], dtype=np.int64), n * n, axis=0)
    lut = np.full((2 * n + 1) ** 3, -1, dtype=np.intp)
    lut[_encode(pos + nrm, n)] = np.arange(6 * n * n)
    for arr in (pos, nrm, lut):
        arr.flags.writeable = False
    return pos, nrm, lut


def sticker_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only (6*N*N, 3) cubie position and outward normal of every sticker.

    x/y/z point toward R/U/F. Coordinates are doubled and centered: a cubie coordinate is
    one of -(N-1), ..., N-1 in steps of 2, so outer layers sit at +-(N-1).
    """
    _check_n(n)
    pos, nrm, _ = _geometry(n)
    return pos, nrm


def _encode(points: np.ndarray, n: int) -> np.ndarray:
    side = 2 * n + 1
    q = points + n
    return (q[:, 0] * side + q[:, 1]) * side + q[:, 2]


@lru_cache(maxsize=None)
def _move_perm(move: Move, n: int) -> np.ndarray:
    if not (0 <= move.axis <= 2 and 0 <= move.lo <= move.hi < n and move.turns in (1, 2, 3)):
        raise ValueError(f"invalid move {move} for N={n}")
    pos, nrm, lut = _geometry(n)
    layer = (n - 1 - pos[:, move.axis]) // 2
    selected = (layer >= move.lo) & (layer <= move.hi)
    rot = np.linalg.matrix_power(_QUARTER[move.axis], move.turns)
    points = pos + nrm
    points[selected] = points[selected] @ rot.T
    dest = lut[_encode(points, n)]  # sticker i moves to position dest[i]
    perm = np.empty_like(dest)
    perm[dest] = np.arange(dest.size)
    perm.flags.writeable = False
    return perm


def move_perm(move: MoveLike, n: int) -> np.ndarray:
    """Read-only permutation for one move: ``new_state = state[..., perm]``."""
    if isinstance(move, str):
        move = parse_move(move, n)
    return _move_perm(move, n)


def alg_perm(alg: Alg, n: int) -> np.ndarray:
    """Single permutation equal to applying every move of ``alg`` in order."""
    perm = np.arange(6 * n * n)
    for move in parse_alg(alg, n):
        perm = perm[_move_perm(move, n)]
    return perm


def apply_move(state: np.ndarray, move: MoveLike) -> np.ndarray:
    """New state (or batch of states) after one move."""
    return state[..., move_perm(move, cube_size(state))]


def apply_alg(state: np.ndarray, alg: Alg) -> np.ndarray:
    """New state (or batch of states) after every move of ``alg``."""
    return state[..., alg_perm(alg, cube_size(state))]


def invert_token(token: str) -> str:
    """Inverse of one move token: R -> R', R' -> R, R2 -> R2."""
    tok = _normalize(token)
    if not _TOKEN.fullmatch(tok):
        raise ValueError(f"unrecognized move {token!r}")
    if tok.endswith("2'"):
        return tok[:-1]
    if tok.endswith("2"):
        return tok
    if tok.endswith("'"):
        return tok[:-1]
    return tok + "'"


def invert_alg(alg: Alg) -> list[MoveLike]:
    """Inverse algorithm: reversed order, each move inverted."""
    items = alg.split() if isinstance(alg, str) else list(alg)
    return [m.inverse() if isinstance(m, Move) else invert_token(m) for m in reversed(items)]


def move_tokens(n: int) -> list[str]:
    """Every notation form for an NxNxN cube, each with '', ' and 2 suffixes.

    Outer turns, inner slices nR (2 <= n <= N-1), wide turns Rw / nRw (2 <= n <= N-1),
    M/E/S on odd N, and x/y/z. Not deduplicated: on a 3x3, ``2R`` and ``M'`` both appear.
    """
    _check_n(n)
    bases = []
    for face in FACES:
        bases.append(face)
        bases += [f"{k}{face}" for k in range(2, n)]
        bases += [f"{face}w" if k == 2 else f"{k}{face}w" for k in range(2, n)]
    if n % 2 == 1:
        bases += ["M", "E", "S"]
    bases += ["x", "y", "z"]
    return [base + suffix for base in bases for suffix in ("", "'", "2")]
