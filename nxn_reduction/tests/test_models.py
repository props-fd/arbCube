import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from cube import sim  # noqa: E402
from data.parse_cfop import FACE_TURNS  # noqa: E402
from models import graph  # noqa: E402
from models.gnn import OneHotMLP, StickerGNN  # noqa: E402


def scrambles(n, count, seed=0, length=30):
    rng = np.random.default_rng(seed)
    tokens = sim.move_tokens(n)
    return np.stack([
        sim.apply_alg(sim.solved(n), [tokens[i] for i in rng.integers(len(tokens), size=length)])
        for _ in range(count)
    ])  # fmt: skip


@pytest.mark.parametrize("n", range(2, 8))
def test_node_features(n):
    x = graph.node_features(n).numpy()
    assert x.shape == (6 * n * n, graph.STATIC_DIM)
    assert (x[:, :6].sum(1) == 1).all() and (x[:, 8:].sum(1) == 1).all()
    assert x[:, 6:8].min() == -1 and x[:, 6:8].max() == 1
    assert len(np.unique(x[:, :8], axis=0)) == 6 * n * n  # face + (u, v) names every sticker
    sizes = x[:, 8:].argmax(1) + 1
    assert (sizes == 3).sum() == 24 and (sizes == 2).sum() == 24 * (n - 2)
    assert (sizes == 1).sum() == 6 * (n - 2) ** 2


@pytest.mark.parametrize("n", range(2, 8))
def test_static_edges(n):
    index, types = graph.static_edges(n)
    src, dst = index.numpy()
    surface = types.numpy() == graph.SURFACE
    assert (np.bincount(src[surface], minlength=6 * n * n) == 4).all()  # 4 neighbours each
    assert (types == graph.PIECE).sum() == 48 + 24 * (n - 2)
    pairs = set(zip(src.tolist(), dst.tolist(), types.tolist()))
    assert all((d, s, t) in pairs for s, d, t in pairs)  # symmetric
    assert not (src == dst).any()


@pytest.mark.parametrize("n", range(2, 8))
def test_home_faces(n):
    home = graph.home_faces(scrambles(n, 3, seed=n))
    assert home.shape == (3, 6 * n * n, graph.HOME_DIM)
    assert np.allclose(home.sum(-1), 1.0 if n > 2 else 0.0)
    solved = graph.home_faces(sim.solved(n))[0]
    if n > 2:  # solved: every sticker's home is its own face
        assert (solved.argmax(-1) == np.arange(6 * n * n) // (n * n)).all() and (solved.max(-1) == 1).all()


def test_home_faces_on_3x3_is_the_matching_center():
    states = scrambles(3, 5, seed=7)
    home = graph.home_faces(states)
    assert set(np.unique(home)) == {0.0, 1.0}
    centers = states[:, 4::9]  # U R F D L B center colors
    for state, h, c in zip(states, home, centers):
        assert (c[h.argmax(-1)] == state).all()


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", graph.COLOR_MODES)
def test_color_edges(n, mode):
    states = scrambles(n, 4, seed=n)
    edges = graph.color_edges(states, mode)
    g, k = n * n, (n - 2) ** 2
    expected = {"all": g * (g - 1), "center": g * (g - 1) - (g - k) * (g - k - 1), "none": 0}[mode]
    assert edges.shape == (4, 2, 6 * expected)
    center = np.zeros(6 * g, dtype=bool)
    center[graph.node_features(n)[:, 8].numpy() == 1] = True
    for state, (src, dst) in zip(states, edges):
        assert (state[src] == state[dst]).all() and not (src == dst).any()
        assert len(set(zip(src.tolist(), dst.tolist()))) == len(src)
        if mode == "center":
            assert (center[src] | center[dst]).all()


def test_color_edges_reject_bad_counts():
    state = sim.solved(3)
    with pytest.raises(ValueError, match="exactly N"):
        graph.color_edges(np.where(np.arange(54) == 0, 1, state).astype(np.uint8))  # 8 U, 10 R
    swapped = state.copy()
    swapped[[4, 13]] = swapped[[13, 4]]  # the U and R centers trade places: still one center each
    assert graph.color_edges(swapped).shape == (1, 2, 96)
    bad = state.copy()
    bad[[4, 10]] = bad[[10, 4]]  # an R edge sticker onto the U center: U has no center, R has two
    with pytest.raises(ValueError, match="center"):
        graph.color_edges(bad)


def test_batch_matches_single_graphs():
    states = scrambles(3, 3, seed=1)
    batch = graph.batch_states(states)
    nodes = batch.nodes_per_graph
    per_graph = batch.edge_index.shape[1] // 3
    for k, state in enumerate(states):
        single = graph.state_to_data(state)
        assert torch.equal(single.x, batch.x[k * nodes : (k + 1) * nodes])
        sl = slice(k * per_graph, (k + 1) * per_graph)
        assert torch.equal(single.edge_index, batch.edge_index[:, sl] - k * nodes)
        assert torch.equal(single.edge_attr, batch.edge_attr[sl])
    assert batch.edge_index.shape[1] == 3 * (216 + 72 + 96)
    assert batch.x.shape == (3 * 54, graph.NODE_DIM)
    old = graph.batch_states(states, home=False, color="all")
    assert old.x.shape == (3 * 54, graph.STATIC_DIM) and old.edge_index.shape[1] == 3 * (216 + 72 + 432)


def test_move_tensors():
    weights, turns = graph.move_tensors(["R", "L", "R'", "L'", "R2", "M"], 3)
    assert torch.allclose(weights.sum(1), torch.ones(6))
    assert ((weights > 0).sum(1) == torch.tensor([21, 21, 21, 21, 21, 12])).all()
    assert turns.argmax(1).tolist() == [0, 0, 1, 1, 2, 1]  # M turns like L: counterclockwise from R


def test_gnn_shapes_and_any_cube_size():
    model = StickerGNN(hidden=16, steps=2)
    for n, moves in [(3, FACE_TURNS), (4, ["R", "2R", "Rw'", "U2"])]:
        logits, value = model(graph.batch_states(scrambles(n, 5)), *graph.move_tensors(moves, n))
        assert logits.shape == (5, len(moves)) and value.shape == (5,)


@pytest.mark.parametrize("home, color", [(True, "center"), (False, "all"), (True, "none")])
def test_gnn_ignores_which_colors_are_which(home, color):
    torch.manual_seed(0)
    model = StickerGNN(hidden=16, steps=3, node_dim=graph.node_dim(home)).eval()
    states = scrambles(3, 4, seed=2)
    relabeled = np.array([3, 5, 0, 1, 4, 2], dtype=np.uint8)[states]
    moves = graph.move_tensors(FACE_TURNS, 3)
    with torch.no_grad():
        a = model(graph.batch_states(states, home, color), *moves)
        b = model(graph.batch_states(relabeled, home, color), *moves)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)


def test_mlp_shape():
    assert OneHotMLP()(torch.from_numpy(scrambles(3, 7))).shape == (7, 18)
