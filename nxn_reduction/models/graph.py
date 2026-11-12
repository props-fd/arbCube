"""Sticker graphs of NxN states for PyG, with features that don't depend on N.

Nodes are the 6*N*N stickers, in state order. Node features (NODE_DIM = 17):
    one-hot face normal (6)     the face the sticker is on, U R F D L B
    (u, v) in [-1, 1] (2)       column and row within the face, corners of the face at +-1
    one-hot piece size (3)      stickers on the sticker's piece: 1 center, 2 edge, 3 corner
    home face (6)               share of the center stickers of the sticker's color that
                                are on each face: on a 3x3, the one-hot face whose center
                                matches the sticker; all zeros on a 2x2 (no centers)

The first 11 (STATIC_DIM) are the same for every state of a given N. Colors themselves
are never features: the home face and the same-color edges depend only on which
stickers share a color, so relabeling the six colors changes nothing the model sees.
``home=False`` drops the home face (the first GNN version).

Edges are directed, one per ordered pair and type; edge_attr is the one-hot type
(EDGE_DIM = 3). A pair can be joined by more than one typed edge.
    0 surface   the stickers share a side on the cube's surface, within a face or across
                a cube edge (every sticker has exactly 4)
    1 piece     the stickers are on the same piece
    2 color     the stickers have the same color; ``color`` picks which pairs:
                "center" (default) only pairs with a center sticker in them, so on a 3x3
                each sticker is joined to its matching center; "all" every same-color
                pair (432 edges on a 3x3 against 96); "none" no color edges

``batch_states`` builds one PyG Data object holding a whole batch of same-N states, with
the usual ``batch`` vector; ``state_to_data`` builds a single graph. ``move_tensors``
describes moves for the policy head: which stickers each move turns, and its turn amount.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from cube import sim
from cube.sim import Move

STATIC_DIM = 11
HOME_DIM = 6
NODE_DIM = STATIC_DIM + HOME_DIM
EDGE_DIM = 3
SURFACE, PIECE, COLOR = range(3)
COLOR_MODES = ("center", "all", "none")


def node_dim(home: bool = True) -> int:
    """Node feature width of graphs built with ``home``."""
    return STATIC_DIM + HOME_DIM * home


@lru_cache(maxsize=None)
def _geometry(n: int) -> tuple[np.ndarray, np.ndarray]:
    pos, nrm = sim.sticker_positions(n)
    return pos.astype(np.int64), nrm.astype(np.int64)


@lru_cache(maxsize=None)
def node_features(n: int) -> torch.Tensor:
    """(6*N*N, STATIC_DIM) float features of every sticker that are the same for all states."""
    if n < 2:
        raise ValueError(f"cube size must be >= 2, got {n}")
    pos, _ = _geometry(n)
    idx = np.arange(6 * n * n)
    face, row, col = idx // (n * n), (idx % (n * n)) // n, idx % n
    _, piece_of, piece_size = np.unique(pos, axis=0, return_inverse=True, return_counts=True)
    size = piece_size[piece_of.ravel()]
    x = np.zeros((6 * n * n, STATIC_DIM), dtype=np.float32)
    x[idx, face] = 1.0
    x[:, 6] = 2 * col / (n - 1) - 1
    x[:, 7] = 2 * row / (n - 1) - 1
    x[idx, 7 + size] = 1.0  # size 1, 2, 3 -> columns 8, 9, 10
    out = torch.from_numpy(x)
    out.requires_grad_(False)
    return out


@lru_cache(maxsize=None)
def static_edges(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface and same-piece edges of one N graph: (edge_index (2, E), edge type (E,))."""
    pos, nrm = _geometry(n)
    points = pos + nrm  # sticker centers on the surface, doubled coordinates
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    same_piece = (pos[:, None, :] == pos[None, :, :]).all(-1)
    np.fill_diagonal(same_piece, False)
    surface = (d2 == 2) | (d2 == 4)  # across a cube edge, or next to each other on a face
    src_s, dst_s = np.nonzero(surface)
    src_p, dst_p = np.nonzero(same_piece)
    edge_index = np.stack([np.concatenate([src_s, src_p]), np.concatenate([dst_s, dst_p])])
    edge_type = np.concatenate([np.full(len(src_s), SURFACE), np.full(len(src_p), PIECE)])
    return torch.from_numpy(edge_index), torch.from_numpy(edge_type)


@lru_cache(maxsize=None)
def _centers(n: int) -> np.ndarray:
    """Indices of the center stickers ((N-2)**2 per face), in state order."""
    return np.nonzero(node_features(n)[:, 8].numpy() == 1)[0]


def _check_counts(states: np.ndarray) -> int:
    """N of a batch of states; each color must fill N*N stickers and (N-2)**2 centers."""
    n = sim.cube_size(states)
    colors = np.arange(6)
    if not ((states[:, :, None] == colors).sum(axis=1) == n * n).all():
        raise ValueError("each color must appear on exactly N*N stickers")
    if not ((states[:, _centers(n), None] == colors).sum(axis=1) == (n - 2) ** 2).all():
        raise ValueError("each color must appear on exactly (N-2)**2 center stickers")
    return n


def home_faces(states: np.ndarray) -> np.ndarray:
    """(B, 6*N*N, HOME_DIM) float32: per sticker, the share of its color's centers on each face."""
    states = np.atleast_2d(states)
    n = _check_counts(states)
    b, centers = len(states), _centers(n)
    counts = np.zeros((b, 6, 6), dtype=np.float32)  # [graph, color, face]
    if len(centers):
        graphs = np.broadcast_to(np.arange(b)[:, None], (b, len(centers)))
        np.add.at(counts, (graphs, states[:, centers], centers // (n * n)), 1.0)
        counts /= (n - 2) ** 2
    return counts[np.arange(b)[:, None], states]


@lru_cache(maxsize=None)
def _pairs(group: int, hubs: int) -> tuple[np.ndarray, np.ndarray]:
    """Ordered pairs (a, b), a != b, of positions within a group, keeping only the pairs
    where a or b is one of the first ``hubs`` positions (``hubs == group``: all pairs)."""
    hub = np.arange(group) < hubs
    return np.nonzero(~np.eye(group, dtype=bool) & (hub[:, None] | hub[None, :]))


def color_edges(states: np.ndarray, mode: str = "center") -> np.ndarray:
    """(B, 2, E) same-color edges of each state in local node indices; ``mode`` is a COLOR_MODES.

    E = 6*G*(G-1) for "all" (G = N*N stickers per color), 6*(G*(G-1) - (G-K)*(G-K-1))
    for "center" (K = (N-2)**2 centers per color) and 0 for "none".
    """
    if mode not in COLOR_MODES:
        raise ValueError(f"color edge mode must be one of {COLOR_MODES}, got {mode!r}")
    states = np.atleast_2d(states)
    n = _check_counts(states)
    g = n * n
    if mode == "all":
        key, hubs = states, g
    else:  # sort each color's centers first, so they are its group's hubs
        not_center = np.ones(6 * g, dtype=np.int64)
        not_center[_centers(n)] = 0
        key, hubs = 2 * states.astype(np.int64) + not_center, (n - 2) ** 2 if mode == "center" else 0
    order = np.argsort(key, axis=1, kind="stable").reshape(len(states), 6, g)  # stickers by color
    a, b = _pairs(g, hubs)
    src = order[:, :, a].reshape(len(states), -1)
    dst = order[:, :, b].reshape(len(states), -1)
    return np.stack([src, dst], axis=1)


def batch_states(states: np.ndarray, home: bool = True, color: str = "center") -> Data:
    """One Data object for a batch of same-N states (x, edge_index, edge_attr, batch).

    ``home`` adds the home-face features and ``color`` picks the same-color edges (see the
    module docstring). Also sets ``num_graphs`` and ``nodes_per_graph``; graph k owns
    nodes k*nodes_per_graph ... (k+1)*nodes_per_graph - 1, in state order.
    """
    states = np.atleast_2d(np.asarray(states))
    b = len(states)
    n = sim.cube_size(states)
    nodes = 6 * n * n
    static_index, static_type = static_edges(n)
    color_index = torch.from_numpy(color_edges(states, color))  # (B, 2, Ec)
    offsets = (torch.arange(b) * nodes).view(b, 1, 1)
    static = static_index.unsqueeze(0) + offsets  # (B, 2, Es)
    edge_index = torch.cat([static, color_index + offsets], dim=2).permute(1, 0, 2).reshape(2, -1)
    types = torch.cat([static_type, torch.full((color_index.shape[2],), COLOR)]).repeat(b)
    x = node_features(n).repeat(b, 1)
    if home:
        x = torch.cat([x, torch.from_numpy(home_faces(states)).reshape(b * nodes, HOME_DIM)], dim=1)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.nn.functional.one_hot(types, EDGE_DIM).float(),
        batch=torch.arange(b).repeat_interleave(nodes),
    )
    data.num_graphs = b
    data.nodes_per_graph = nodes
    data.n = n
    return data


def state_to_data(state: np.ndarray, home: bool = True, color: str = "center") -> Data:
    """The sticker graph of one NxN state."""
    if np.asarray(state).ndim != 1:
        raise ValueError("state_to_data takes a single state; use batch_states for a batch")
    data = batch_states(state[None], home, color)
    del data.batch
    return data


def move_tensors(moves: Iterable[Move | str], n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy-head description of moves on an NxN cube.

    Returns (layer_weights (M, 6*N*N), turn_features (M, 3)). Row m of layer_weights
    averages over the stickers move m turns (every sticker on a cubie in its layers).
    turn_features is the one-hot turn as seen from the layer's own side: clockwise,
    counterclockwise or half, so R and L both count as clockwise.
    """
    pos, _ = _geometry(n)
    weights, turns = [], []
    for move in moves:
        if isinstance(move, str):
            move = sim.parse_move(move, n)
        layer = (n - 1 - pos[:, move.axis]) // 2
        mask = (layer >= move.lo) & (layer <= move.hi)
        weights.append(mask / mask.sum())
        own_side_turns = move.turns if move.lo + move.hi <= n - 1 else 4 - move.turns
        turns.append({1: 0, 3: 1, 2: 2}[own_side_turns])
    layer_weights = torch.tensor(np.stack(weights), dtype=torch.float32)
    turn_features = torch.nn.functional.one_hot(torch.tensor(turns), 3).float()
    return layer_weights, turn_features
