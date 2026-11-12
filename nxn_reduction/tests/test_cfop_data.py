import numpy as np
import pytest

from cube import sim
from data import build_dataset as bd
from data import parse_cfop as pc

# Our face -> the color letter the dataset uses on that face (yellow up, green front).
LETTER = dict(zip(sim.FACES, "yogwrb"))


def to_dataset_letters(state, face_order=pc.SOLUTION_FACES):
    """Our state -> the 54 color letters of a training.N state line."""
    blocks = {f: state[9 * k : 9 * k + 9] for k, f in enumerate(sim.FACES)}
    return [LETTER[sim.FACES[c]] for f in face_order for c in blocks[f]]


def scrambled(seed=0, length=30):
    rng = np.random.default_rng(seed)
    return sim.apply_alg(sim.solved(3), [pc.FACE_TURNS[i] for i in rng.integers(18, size=length)])


def test_parse_state_round_trips_solution_format():
    state = scrambled()
    assert np.array_equal(pc.parse_state(to_dataset_letters(state)), state)


def test_parse_state_round_trips_seq_format():
    state = scrambled(1)
    letters = to_dataset_letters(state, pc.SEQ_FACES)
    digits = [str(pc.SEQ_DIGITS.index(x)) for x in letters]
    assert np.array_equal(pc.parse_state(digits, pc.SEQ_FACES), state)


def test_parse_state_rejects_bad_states():
    letters = to_dataset_letters(sim.solved(3))
    with pytest.raises(ValueError):
        pc.parse_state(letters[:53])
    with pytest.raises(ValueError):
        pc.parse_state(letters[:18] + ["r"] + letters[19:])  # a U sticker turned red: ten reds


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["R", "U'", "F2"], ["R", "U'", "F2"]),
        (["y", "R"], ["B"]),  # after y the old back face is on the right
        (["x", "U"], ["F"]),  # after x the old front face is on top
        (["y'", "F2"], ["L2"]),  # after y' the old left face is in front
        (["r", "U"], ["L", "F"]),  # r = L + x
        (["M"], ["R", "L'"]),  # M = R L' x'
        (["x", "y", "R", "U"], ["U", "F"]),
    ],
)
def test_normalize_known_cases(tokens, expected):
    assert pc.normalize_moves(tokens) == expected


ROTATIONS = [
    sim.alg_perm(seq, 3)
    for seq in [f"{a} {b}".split() for a in ("", "x", "x2", "x'", "z", "z'") for b in ("", "y", "y2", "y'")]
]


@pytest.mark.parametrize("seed", range(20))
def test_normalized_solution_matches_raw_up_to_a_rotation(seed):
    rng = np.random.default_rng(seed)
    pool = ["x", "y", "z", "M", "E", "S", "r", "u", "f", "l", "d", "b"] + list(pc.FACE_TURNS)
    raw = [pool[i] + ["", "'", "2"][j] if pool[i][-1] not in "'2" else pool[i]
           for i, j in zip(rng.integers(len(pool), size=25), rng.integers(3, size=25))]  # fmt: skip
    normalized = pc.normalize_moves(raw)
    assert set(normalized) <= set(pc.FACE_TURNS)
    start = scrambled(seed)
    raw_end = sim.apply_alg(start, raw)
    norm_end = sim.apply_alg(start, normalized)
    assert any(np.array_equal(norm_end[r], raw_end) for r in ROTATIONS)


def test_token_kinds():
    assert [pc.token_kind(t) for t in ["R2", "x'", "Rw", "r", "M", "2R", "Q"]] == [
        "face turn", "rotation", "wide", "wide", "slice", "slice", "unrecognized",
    ]  # fmt: skip


def _stages(scramble_alg, solution):
    moves = np.array([pc.FACE_TURN_INDEX[m] for m in pc.normalize_moves(solution.split())], dtype=np.uint8)
    states = bd.replay(sim.apply_alg(sim.solved(3), scramble_alg), moves)
    return "".join("cfop"[s] for s in bd.stage_labels(states, moves))


def test_stage_labels_simple_cases():
    assert _stages("U", "U'") == "p"  # only the last layer is turned: AUF counts as PLL
    assert _stages("R", "R'") == "c"  # R breaks the cross
    sune = "R U R' U R U2 R'"
    assert _stages(sim.invert_alg(sune), sune) == "ooooooo"


def _before_last_f2l_move(oll_alg, last_f2l_move):
    """Scramble whose solution is: last_f2l_move (finishing F2L), then oll_alg (solving)."""
    return sim.invert_alg(oll_alg.split()) + [sim.invert_token(last_f2l_move)]


def test_stage_boundary_inside_a_merged_move():
    # Real path: F' finishes F2L, then the OLL "F' L' U' L U F". Written: F' F' -> F2.
    scramble = _before_last_f2l_move("F' L' U' L U F", "F'")
    # Without the partial-move check every row would be "c": no recorded state has the cross.
    # The merged F2 finishes the cross and F2L here, since nothing came before it.
    assert _stages(scramble, "F2 L' U' L U F") == "c" + "o" * 5


def test_stage_boundary_hidden_by_cancelled_moves():
    # Real path: F' finishes F2L, then the OLL "F R U R' U' F'". Written: F' F cancel.
    scramble = _before_last_f2l_move("F R U R' U' F'", "F'")
    assert _stages(scramble, "R U R' U' F'") == "o" * 5


def test_assign_splits_partitions_90_5_5():
    ids = list(range(1000, 3000))
    split = bd.assign_splits(ids, seed=7)
    counts = {s: sum(v == s for v in split.values()) for s in bd.SPLITS}
    assert counts == {"train": 1800, "val": 100, "test": 100}
    assert bd.assign_splits(ids, seed=7) == split  # deterministic


needs_data = pytest.mark.skipif(
    not pc.RAW_DIR.is_dir() or not any(pc.RAW_DIR.rglob("training.0")), reason="data/raw not downloaded"
)


@needs_data
def test_first_solves_of_training_0_verify():
    solves = list(pc.iter_solves(shards=[0], limit_per_shard=200))
    assert len(solves) == 200
    assert all(all(pc.verify(s)) for s in solves)


@needs_data
def test_seq_file_matches_solutions():
    result = pc.check_sequences(limit_per_shard=20)
    assert result["solves"] > 0 and result["mismatched_solves"] == 0, result["examples"]


@needs_data
def test_dataset_rows_replay(tmp_path):
    bd.build(out_dir=tmp_path, limit_per_shard=15)
    data = np.load(tmp_path / "cfop_train.npz")
    s, m, r, st, sid = (data[k] for k in ("state", "next_move", "moves_remaining", "cfop_stage", "solve_id"))
    perms = np.stack([sim.move_perm(x, 3) for x in data["move_names"]])
    after = np.take_along_axis(s, perms[m], axis=1)
    last = r == 1
    assert (after[last] == sim.solved(3)).all()
    cont = ~last[:-1]
    assert np.array_equal(after[:-1][cont], s[1:][cont])
    assert (sid[1:][cont] == sid[:-1][cont]).all() and (np.diff(st.astype(int))[cont] >= 0).all()
