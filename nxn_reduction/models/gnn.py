"""GINE encoder over sticker graphs, with a value head and a move-scoring policy head.

Encoder: node features (``node_dim`` = graph.node_dim(home) wide) -> hidden, then ONE
shared GINEConv layer applied T times (residual + LayerNorm after each step), so depth is
set by T without adding parameters.

Value head: mean and max readout over each graph's stickers -> MLP -> one number
(trained to predict moves_remaining).

Policy head: for each candidate move, average the embeddings of the stickers that move
turns (graph.move_tensors), append the move's turn one-hot (clockwise / counter / half),
and score it with an MLP. Nothing in it depends on N or on a fixed move list, so the same
weights can score any set of moves on any cube size.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv

from models.graph import EDGE_DIM, NODE_DIM


def mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers += [nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
    return nn.Sequential(*layers)


class StickerGNN(nn.Module):
    def __init__(
        self, hidden: int = 128, steps: int = 6, aggr: str = "add", dropout: float = 0.0, node_dim: int = NODE_DIM
    ):
        super().__init__()
        self.steps = steps
        self.embed = nn.Linear(node_dim, hidden)
        self.conv = GINEConv(mlp([hidden, hidden, hidden]), train_eps=True, edge_dim=EDGE_DIM, aggr=aggr)
        self.norm = nn.LayerNorm(hidden)
        self.value_head = mlp([2 * hidden, hidden, 1], dropout)
        self.policy_head = mlp([hidden + 3, hidden, 1], dropout)

    def encode(self, data: Data) -> torch.Tensor:
        """Sticker embeddings, (num_nodes, hidden)."""
        h = self.embed(data.x)
        for _ in range(self.steps):
            h = self.norm(h + self.conv(h, data.edge_index, data.edge_attr))
        return h

    def forward(
        self, data: Data, layer_weights: torch.Tensor, turn_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(move logits (B, M), value (B,)) for a batch of same-N graphs (graph.batch_states)."""
        h = self.encode(data)
        per_graph = h.view(data.num_graphs, data.nodes_per_graph, -1)
        readout = torch.cat([per_graph.mean(dim=1), per_graph.amax(dim=1)], dim=-1)
        value = self.value_head(readout).squeeze(-1)
        pooled = torch.einsum("mn,bnh->bmh", layer_weights, per_graph)
        turns = turn_features.unsqueeze(0).expand(per_graph.shape[0], -1, -1)
        logits = self.policy_head(torch.cat([pooled, turns], dim=-1)).squeeze(-1)
        return logits, value


class OneHotMLP(nn.Module):
    """Baseline: one-hot 3x3 sticker colors (54 x 6) -> MLP -> 18 move logits."""

    def __init__(self, stickers: int = 54, moves: int = 18, hidden: int = 512, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stickers = stickers
        self.net = mlp([stickers * 6] + [hidden] * depth + [moves], dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = nn.functional.one_hot(states.long(), 6).float().view(len(states), self.stickers * 6)
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
