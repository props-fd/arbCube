import numpy as np
import pytest

from cube import reduce as rd
from cube import sim
from cube.sim import Move

TOKENS3 = sim.move_tokens(3)  # outer, slice, wide and rotation moves of the 3x3
PERMS3 = np.stack([sim.move_perm(t, 3) for t in TOKENS3])


def random_3x3_states(count, seed, length=30):
    rng = np.random.default_rng(seed)
    states = np.tile(sim.solved(3), (count, 1))
    for _ in range(length):
        states = np.take_along_axis(states, PERMS3[rng.integers(len(TOKENS3), size=count)], axis=1)
    return states


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_lifted_moves_commute_with_reduction(n):
    """reduce(apply_N(lift_state(s), lift_move(m))) == apply_3(s, m) for 10k random (s, m)."""
    count = 10_000
    states3 = random_3x3_states(count, seed=n)
    which = np.random.default_rng(100 + n).integers(len(TOKENS3), size=count)
    lifted = np.stack([sim.move_perm(rd.lift_move(t, n), n) for t in TOKENS3])
    big = np.take_along_axis(rd.lift_state(states3, n), lifted[which], axis=1)
    small = np.take_along_axis(states3, PERMS3[which], axis=1)
    assert np.array_equal(rd.reduce(big), small)


@pytest.mark.parametrize("n", range(3, 8))
def test_group_sizes(n):
    sizes = np.bincount(rd.projection(n), minlength=54).reshape(6, 9)
    expected = np.array([1, n - 2, 1, n - 2, (n - 2) ** 2, n - 2, 1, n - 2, 1])
    assert (sizes == expected).all()


@pytest.mark.parametrize("n", range(3, 8))
def test_lift_then_reduce_round_trips(n):
    states3 = random_3x3_states(50, seed=7)
    big = rd.lift_state(states3, n)
    assert rd.is_reduced(big).all()
    assert np.array_equal(rd.reduce(big), states3)
    assert np.array_equal(rd.lift_state(sim.solved(3), n), sim.solved(n))


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_inner_layer_turns_break_reduction(n):
    solved = sim.solved(n)
    assert rd.is_reduced(solved)
    assert not rd.is_reduced(sim.apply_move(solved, "2R"))
    with pytest.raises(ValueError):
        rd.reduce(sim.apply_alg(solved, "2R U"))
    # Outer turns, and all inner layers turning together, keep it reduced.
    assert rd.is_reduced(sim.apply_alg(solved, ["R", "U'", rd.lift_move("M", n), "F2"]))


def test_lift_move_examples():
    assert rd.lift_move("R", 5) == sim.parse_move("R", 5)
    assert rd.lift_move("L'", 6) == sim.parse_move("L'", 6)
    assert rd.lift_move("M", 6) == Move(0, 1, 4, 3)  # all inner layers, turning like L
    assert rd.lift_move("Rw", 5) == sim.parse_move("4Rw", 5)
    assert rd.lift_move("y'", 7) == sim.parse_move("y'", 7)
    assert rd.lift_move("2R", 6, base=4) == Move(0, 1, 2, 1)
    assert rd.lift_move("3R", 6, base=4) == Move(0, 3, 4, 1)
    assert rd.lift_move("R", 3) == sim.parse_move("R", 3)


def test_layer_groups_reject_uneven_splits():
    with pytest.raises(ValueError):
        rd.layer_groups(4, 5)
    with pytest.raises(ValueError):
        rd.lift_state(sim.solved(4), 6)
