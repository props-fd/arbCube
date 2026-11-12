import numpy as np
import pytest

from cube import parity as pa
from cube import reduce as rd
from cube import sim

FACE_TURNS = [f + s for f in sim.FACES for s in ("", "'", "2")]
CORNER_SLOTS, EDGE_SLOTS = pa.slots()


def legal_3x3(seed, length=40):
    rng = np.random.default_rng(seed)
    return sim.apply_alg(sim.solved(3), [FACE_TURNS[i] for i in rng.integers(18, size=length)])


def name(i):
    return f"{sim.FACES[i // 9]}{i % 9}"


def induced(moves, n):
    """(keeps every reduction group together, 3x3-level effect as {dest: src} sticker names)."""
    perm = sim.alg_perm(list(moves), n)
    proj = rd.projection(n)
    source_group = proj[perm]
    keeps = all(len(set(source_group[proj == g].tolist())) == 1 for g in range(54))
    rep = np.unique(proj, return_index=True)[1]
    q = proj[perm[rep]]
    return keeps, {name(i): name(q[i]) for i in range(54) if q[i] != i}


@pytest.mark.parametrize("n", [4, 6])
def test_oll_parity_alg_only_flips_uf(n):
    assert induced(pa.parity_alg("oll", n), n) == (True, {"U7": "F1", "F1": "U7"})


@pytest.mark.parametrize("n", [4, 6])
def test_pll_parity_alg_only_swaps_uf_and_ub(n):
    assert induced(pa.parity_alg("pll", n), n) == (True, {"U1": "U7", "U7": "U1", "F1": "B1", "B1": "F1"})


def test_cubie_definitions():
    solved = pa.cubies(sim.solved(3))
    assert solved.corner_perm == tuple(range(8)) and solved.edge_perm == tuple(range(12))
    assert not any(solved.corner_twist) and not any(solved.edge_flip)
    after_r = pa.cubies(sim.apply_move(sim.solved(3), "R"))
    assert sum(after_r.corner_twist) % 3 == 0 and sorted(after_r.corner_twist) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert not any(after_r.edge_flip)
    after_f = pa.cubies(sim.apply_move(sim.solved(3), "F"))
    assert sum(after_f.edge_flip) == 4  # F and B quarter turns flip four edges


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_lifted_legal_states_have_no_parity(n):
    for seed in range(25):
        report = pa.detect(rd.lift_state(legal_3x3(seed), n))
        assert report.solvable_as_3x3, report


def flip_edge(state3, k):
    s = state3.copy()
    a, b = EDGE_SLOTS[k]
    s[[a, b]] = s[[b, a]]
    return s


def swap_edges(state3, j, k):
    s = state3.copy()
    s[EDGE_SLOTS[j]], s[EDGE_SLOTS[k]] = state3[EDGE_SLOTS[k]], state3[EDGE_SLOTS[j]]
    return s


def swap_corners(state3, j, k):
    s = state3.copy()
    s[CORNER_SLOTS[j]], s[CORNER_SLOTS[k]] = state3[CORNER_SLOTS[k]], state3[CORNER_SLOTS[j]]
    return s


def parity_cases(seed):
    """(name, 3x3-level state, expected OLL parity, expected PLL parity) from one legal state."""
    rng = np.random.default_rng(seed)
    base = legal_3x3(seed)
    e1, e2 = (int(x) for x in rng.choice(12, size=2, replace=False))
    c1, c2 = (int(x) for x in rng.choice(8, size=2, replace=False))
    return [
        ("flipped edge", flip_edge(base, e1), True, False),
        ("swapped edges", swap_edges(base, e1, e2), False, True),
        ("swapped corners", swap_corners(base, c1, c2), False, True),
        ("both", flip_edge(swap_edges(base, e1, e2), e1), True, True),
    ]


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("seed", range(40))
def test_lifted_parity_case_is_fixed_and_stays_reduced(n, seed):
    for label, state3, oll, pll in parity_cases(seed):
        big = rd.lift_state(state3, n)
        report = pa.detect(big)
        assert (report.oll_parity, report.pll_parity) == (oll, pll), label
        fixed, moves = pa.fix_parity(big)
        assert rd.is_reduced(fixed), label
        assert pa.detect(fixed).solvable_as_3x3, label
        assert len(moves) == (15 if oll else 0) + (6 if pll else 0), label
        assert np.array_equal(big, rd.lift_state(state3, n)), "fix_parity must not change its input"


def test_fix_parity_leaves_good_states_alone():
    big = rd.lift_state(legal_3x3(3), 4)
    fixed, moves = pa.fix_parity(big)
    assert moves == [] and np.array_equal(fixed, big)


@pytest.mark.parametrize("n", [4, 6])
def test_center_arrangement(n):
    solved = sim.solved(n)
    assert pa.center_arrangement_ok(solved)
    rotated = rd.lift_state(sim.apply_alg(sim.solved(3), "x y'"), n)
    assert pa.center_arrangement_ok(rotated) and pa.detect(rotated).solvable_as_3x3
    centers = rd.projection(n) % 9 == 4
    for a, b in [("R", "L"), ("U", "F")]:  # mirrored, and two neighbours swapped
        bad = solved.copy()
        fa, fb = sim.FACES.index(a), sim.FACES.index(b)
        bad[centers & (np.arange(6 * n * n) // (n * n) == fa)] = fb
        bad[centers & (np.arange(6 * n * n) // (n * n) == fb)] = fa
        assert rd.is_reduced(bad)
        assert not pa.center_arrangement_ok(bad)
        assert not pa.detect(bad).centers_ok
        with pytest.raises(ValueError):
            pa.fix_parity(bad)


def test_center_blocks_must_be_solved_to_check_them():
    with pytest.raises(ValueError):
        pa.center_colors(sim.apply_move(sim.solved(4), "2R"))


@pytest.mark.parametrize("n", [5, 7])
def test_odd_cubes_reject_parity(n):
    assert pa.center_arrangement_ok(rd.lift_state(legal_3x3(1), n))
    with pytest.raises(ValueError):
        pa.fix_parity(rd.lift_state(flip_edge(legal_3x3(1), 0), n))


def test_detect_requires_a_reduced_state():
    with pytest.raises(ValueError):
        pa.detect(sim.apply_move(sim.solved(4), "2R"))
