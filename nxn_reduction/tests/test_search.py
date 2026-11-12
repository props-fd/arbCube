import numpy as np
import pytest

from cube import sim
from data.parse_cfop import FACE_TURNS
from search.beam import FACE_OF, MOVE_PERMS, beam_search, qtm, state_hash, tokens, verify
from search.evaluate import QUARTER_TURNS, inverse, random_walks


def uniform(states):
    return np.full((len(states), 18), -np.log(18.0))


def face_walk(length, rng):
    """A face-turn walk from solved that never turns the same face twice in a row or revisits a state."""
    while True:
        state, moves, seen = sim.solved(3), [], {sim.solved(3).tobytes()}
        for _ in range(length):
            m = int(rng.choice([m for m in range(18) if not moves or FACE_OF[m] != FACE_OF[moves[-1]]]))
            state = state[MOVE_PERMS[m]]
            moves.append(m)
            if state.tobytes() in seen:
                break
            seen.add(state.tobytes())
        else:
            return state, moves


def oracle(walks):
    """A policy that puts 0.9 on undoing the scramble's last unundone move, uniform elsewhere."""
    best = {}
    for _, moves in walks:
        state = sim.solved(3)
        for m in moves:
            state = state[MOVE_PERMS[m]]
            best[state.tobytes()] = inverse(m)

    def policy(states):
        out = np.full((len(states), 18), np.log(0.1 / 17))
        for row, state in zip(out, states):
            if state.tobytes() in best:
                row[best[state.tobytes()]] = np.log(0.9)
        return out

    return policy


def test_solved_scramble_needs_no_moves():
    result = beam_search(sim.solved(3)[None], uniform)
    assert result.solutions == [[]] and result.nodes.tolist() == [0]


def test_uniform_policy_solves_every_two_move_scramble():
    rng = np.random.default_rng(0)
    walks = [face_walk(2, rng) for _ in range(20)] + [face_walk(1, rng) for _ in range(5)]
    states = np.stack([w[0] for w in walks])
    result = beam_search(states, uniform, width=270)  # 270 = every 2-move path without a repeated face
    verify(states, result.solutions)
    assert result.solved.all()
    assert [len(s) for s in result.solutions] == [len(w[1]) for w in walks]


def test_oracle_policy_is_followed_along_long_paths():
    rng = np.random.default_rng(1)
    walks = [face_walk(25, rng) for _ in range(6)]
    states = np.stack([w[0] for w in walks])
    for width in (1, 4):
        result = beam_search(states, oracle(walks), width=width)
        verify(states, result.solutions)
        assert result.solved.all()
        assert [s == [inverse(m) for m in reversed(w[1])] for s, w in zip(result.solutions, walks)] == [True] * 6
        assert (result.nodes == 25 * width).all() if width == 1 else (result.nodes <= 25 * width).all()


def test_gives_up_at_max_moves():
    state, _ = face_walk(12, np.random.default_rng(2))
    result = beam_search(state[None], uniform, width=4, max_moves=3)
    assert result.solutions == [None] and result.nodes[0] <= 1 + 4 + 4


def test_solutions_never_turn_a_face_twice_in_a_row():
    rng = np.random.default_rng(3)
    states = np.stack([face_walk(3, rng)[0] for _ in range(10)])
    result = beam_search(states, uniform, width=64, max_moves=6)
    for sol in result.solutions:
        if sol:
            assert all(FACE_OF[a] != FACE_OF[b] for a, b in zip(sol, sol[1:]))


def test_verify_rejects_a_wrong_solution():
    state = sim.apply_alg(sim.solved(3), "R U")
    verify(state[None], [[FACE_TURNS.index("U'"), FACE_TURNS.index("R'")]])
    with pytest.raises(AssertionError):
        verify(state[None], [[FACE_TURNS.index("R'"), FACE_TURNS.index("U'")]])


def test_helpers():
    assert tokens([0, 4, 8]) == ["U", "R'", "F2"] and qtm([0, 4, 8]) == 4
    assert [FACE_TURNS[inverse(FACE_TURNS.index(t))] for t in ("R", "R'", "R2")] == ["R'", "R", "R2"]
    states = random_walks(np.array([0, 1, 5]), seed=0)
    assert len(set(state_hash(states).tolist())) == 3 and sim.is_solved(states[0])


def test_random_walks_are_quarter_turns_without_undo():
    depths = np.array([1, 2, 6])
    states = random_walks(depths, seed=5)
    assert set(QUARTER_TURNS.tolist()) == {m for m in range(18) if not FACE_TURNS[m].endswith("2")}
    one = {sim.apply_alg(sim.solved(3), FACE_TURNS[m]).tobytes() for m in QUARTER_TURNS}
    assert states[0].tobytes() in one
    assert not sim.is_solved(states[1])  # no immediate undo, so 2 quarter turns can't cancel


def test_torch_policy_wrapper(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import json

    from models import graph
    from models.gnn import OneHotMLP, StickerGNN
    from search.evaluate import load_policy

    (tmp_path / "models").mkdir()
    torch.save(OneHotMLP().state_dict(), tmp_path / "models" / "mlp.pt")
    torch.save(StickerGNN(hidden=8, steps=2, node_dim=graph.node_dim(True)).state_dict(), tmp_path / "models" / "g.pt")
    report = {"mlp": {}, "g": {"hidden": 8, "steps": 2, "home": True, "color_edges": "center"}}
    (tmp_path / "train_report.json").write_text(json.dumps(report))
    states = random_walks(np.array([3, 7, 9]), seed=1)
    for name in ("mlp", "g"):
        out = load_policy(name, tmp_path, batch_size=2)(states)
        assert out.shape == (3, 18) and np.allclose(np.exp(out).sum(1), 1, atol=1e-5)
