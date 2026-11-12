from math import lcm

import numpy as np
import pytest

from cube import sim

SIZES = range(2, 8)


def identity(n):
    return np.arange(6 * n * n)


def random_scramble(n, length, seed):
    tokens = sim.move_tokens(n)
    rng = np.random.default_rng(seed)
    return [tokens[i] for i in rng.integers(len(tokens), size=length)]


def state_order(alg, n, limit=5000):
    """Smallest k > 0 such that applying ``alg`` k times to solved gives solved."""
    perm = sim.alg_perm(alg, n)
    start = sim.solved(n)
    state = start[perm]
    for k in range(1, limit + 1):
        if np.array_equal(state, start):
            return k
        state = state[perm]
    raise AssertionError(f"order of {alg!r} exceeds {limit}")


def perm_order(perm):
    """Order of a permutation: lcm of its cycle lengths."""
    seen = np.zeros(perm.size, dtype=bool)
    order = 1
    for start in range(perm.size):
        length = 0
        i = start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
            length += 1
        if length:
            order = lcm(order, length)
    return order


# --- the required properties --------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_every_move_four_times_is_identity(n):
    ident = identity(n)
    trivial = [t for t in sim.move_tokens(n) if np.array_equal(sim.move_perm(t, n), ident)]
    bad = [t for t in sim.move_tokens(n) if not np.array_equal(sim.alg_perm([t] * 4, n), ident)]
    assert not trivial, f"moves that do nothing: {trivial}"
    assert not bad


@pytest.mark.parametrize("n", SIZES)
def test_every_move_then_inverse_is_identity(n):
    ident = identity(n)
    bad = [
        t
        for t in sim.move_tokens(n)
        if not np.array_equal(sim.alg_perm([t, sim.invert_token(t)], n), ident)
    ]
    assert not bad


@pytest.mark.parametrize("n", SIZES)
def test_color_counts_preserved(n):
    expected = np.full(6, n * n)
    state = sim.solved(n)
    assert np.array_equal(np.bincount(state, minlength=6), expected)
    for token in random_scramble(n, 300, seed=n):
        state = sim.apply_move(state, token)
        assert state.dtype == np.uint8 and state.shape == (6 * n * n,)
        assert np.array_equal(np.bincount(state, minlength=6), expected), token


@pytest.mark.parametrize("alg, order", [("R U", 105), ("R U R' U'", 6)])
def test_known_orders_on_3x3(alg, order):
    assert state_order(alg, 3) == order
    assert perm_order(sim.alg_perm(alg, 3)) == order


@pytest.mark.parametrize("n", [4, 5])
def test_inner_slices_four_times_is_identity(n):
    ident = identity(n)
    slices = [f"{k}{face}" for face in sim.FACES for k in range(2, n)]
    if n % 2:
        slices += ["M", "E", "S"]
    for token in slices:
        perm = sim.move_perm(token, n)
        assert not np.array_equal(perm, ident), token
        assert np.array_equal(sim.alg_perm([token] * 4, n), ident), token
        # An inner slice never touches the two outer faces perpendicular to its axis.
        axis = sim.parse_move(token, n).axis
        for face in ("RL", "UD", "FB")[axis]:
            f = sim.FACES.index(face) * n * n
            assert np.array_equal(perm[f : f + n * n], ident[f : f + n * n]), (token, face)


@pytest.mark.parametrize("n", SIZES)
def test_scramble_then_inverse_is_solved(n):
    scramble = random_scramble(n, 60, seed=1000 + n)
    scrambled = sim.apply_alg(sim.solved(n), " ".join(scramble))
    assert not sim.is_solved(scrambled)
    restored = sim.apply_alg(scrambled, sim.invert_alg(scramble))
    assert np.array_equal(restored, sim.solved(n))


# --- convention checks --------------------------------------------------------
# The properties above also hold for a mirrored or mis-oriented simulator, so these pin
# the actual face layout and turn directions from CLAUDE.md.


def facelet(name):
    """Kociemba facelet name (e.g. 'U3', numbered 1-9) -> 3x3 sticker index."""
    return sim.FACES.index(name[0]) * 9 + int(name[1]) - 1


# For each move: destination facelet <- source facelet, for stickers on the four side faces.
KNOWN_3X3 = {
    "R": {
        "U3": "F3", "U6": "F6", "U9": "F9",
        "F3": "D3", "F6": "D6", "F9": "D9",
        "D3": "B7", "D6": "B4", "D9": "B1",
        "B1": "U9", "B4": "U6", "B7": "U3",
    },
    "U": {
        "F1": "R1", "F2": "R2", "F3": "R3",
        "R1": "B1", "R2": "B2", "R3": "B3",
        "B1": "L1", "B2": "L2", "B3": "L3",
        "L1": "F1", "L2": "F2", "L3": "F3",
    },
    "F": {
        "R1": "U7", "R4": "U8", "R7": "U9",
        "D3": "R1", "D2": "R4", "D1": "R7",
        "L3": "D1", "L6": "D2", "L9": "D3",
        "U9": "L3", "U8": "L6", "U7": "L9",
    },
}  # fmt: skip


@pytest.mark.parametrize("move", sorted(KNOWN_3X3))
def test_3x3_sticker_mapping(move):
    expected = identity(3)
    for dest, src in KNOWN_3X3[move].items():
        expected[facelet(dest)] = facelet(src)
    f = sim.FACES.index(move) * 9
    expected[f : f + 9] = np.rot90(np.arange(f, f + 9).reshape(3, 3), -1).ravel()  # clockwise
    assert np.array_equal(sim.move_perm(move, 3), expected)


def test_3x3_R_matches_kociemba_facelet_string():
    state = sim.apply_move(sim.solved(3), "R")
    facelets = "".join(sim.FACES[c] for c in state)
    assert facelets == "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"


EQUIVALENT = [
    (3, "x", "R M' L'"),
    (3, "y", "U E' D'"),
    (3, "z", "F S B'"),
    (3, "M", "2L"),
    (2, "Rw", "x"),
    (4, "Rw", "R 2R"),
    (4, "r", "Rw"),
    (4, "3Rw", "R 2R 3R"),
    (4, "3r", "3Rw"),
    (4, "4Rw", "x"),
    (4, "4R", "L'"),
    (5, "M", "3L"),
    (5, "E", "3D"),
    (5, "S", "3F"),
    (6, "2R", "5L'"),
    (6, "3Uw'", "3U' 2U' U'"),
    (7, "x", "R 2R 3R 4R 5R 6R 7R"),
    (7, "R2'", "R2"),
    (5, "R’", "R'"),
]


@pytest.mark.parametrize("n, lhs, rhs", EQUIVALENT)
def test_notation_equivalences(n, lhs, rhs):
    assert np.array_equal(sim.alg_perm(lhs, n), sim.alg_perm(rhs, n))


@pytest.mark.parametrize(
    "n, token",
    [(4, "M"), (2, "E"), (3, "4R"), (3, "0R"), (3, "4Rw"), (3, "1Rw"), (3, "rw"),
     (3, "2x"), (3, "Mw"), (3, "R3"), (3, "Q")],
)  # fmt: skip
def test_invalid_moves_raise(n, token):
    with pytest.raises(ValueError):
        sim.parse_move(token, n)


@pytest.mark.parametrize("n", SIZES)
def test_is_solved_ignores_whole_cube_orientation(n):
    assert sim.is_solved(sim.apply_alg(sim.solved(n), "x y2 z'"))
    assert not sim.is_solved(sim.apply_move(sim.solved(n), "R"))


def test_batched_states():
    n = 4
    single = sim.apply_alg(sim.solved(n), "R U")
    batch = np.stack([sim.solved(n), single])
    out = sim.apply_move(batch, "Fw")
    assert np.array_equal(out[1], sim.apply_move(single, "Fw"))
    assert list(sim.is_solved(batch)) == [True, False]
    assert np.array_equal(batch[1], single)  # inputs are not mutated
