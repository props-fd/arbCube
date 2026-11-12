"""Solve scrambles with a trained policy and beam search, and report solve rates.

Two suites, both seeded:
    depth   random quarter-turn walks from solved, ``--per-depth`` of each depth 1..K
            (QTM, no move followed by its inverse), like the team's cubetransfer suite;
            ``bar`` is its pass test: the share of depth 1..10 walks solved within 14 QTM
    cfop    full scrambles: the start states of ``--cfop`` solves from the test split,
            next to the length of the human CFOP solution

Every solution is replayed in the simulator before it is counted. Results go to
artifacts/search_report.json under "<model>/w<width>"; a later run with the same key
replaces only the suites it ran (``--per-depth 0`` or ``--cfop 0`` skips a suite).

    .venv\\Scripts\\python -m search.evaluate --model mlp
    .venv\\Scripts\\python -m search.evaluate --model gnn_v2 --width 64 --cfop 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cube import sim
from data.parse_cfop import ARTIFACTS, FACE_TURNS
from search.beam import Policy, SearchResult, beam_search, qtm, verify

QUARTER_TURNS = np.array([m for m in range(len(FACE_TURNS)) if m % 3 != 2])
BAR_DEPTH, BAR_SLACK = 10, 4  # cubetransfer's 3x3 suite: depth 1..10, solved within 10 + 4 QTM


def inverse(move: int) -> int:
    return move - move % 3 + {0: 1, 1: 0, 2: 2}[move % 3]


def random_walks(depths: np.ndarray, seed: int) -> np.ndarray:
    """(len(depths), 54) states: quarter-turn walks from solved, no move followed by its inverse."""
    rng = np.random.default_rng(seed)
    perms = np.stack([sim.move_perm(m, 3) for m in FACE_TURNS])
    out = np.empty((len(depths), 54), dtype=np.uint8)
    for k, depth in enumerate(depths):
        state, prev = sim.solved(3), -1
        for _ in range(depth):
            options = QUARTER_TURNS[QUARTER_TURNS != (inverse(prev) if prev >= 0 else -1)]
            prev = int(rng.choice(options))
            state = state[perms[prev]]
        out[k] = state
    return out


def cfop_scrambles(count: int, seed: int, data_dir: Path = ARTIFACTS) -> tuple[np.ndarray, np.ndarray]:
    """Start states of ``count`` random test-split solves, and their human solution lengths."""
    with np.load(data_dir / "cfop_test.npz") as d:
        solve_id, remaining, states = d["solve_id"], d["moves_remaining"], d["state"]
    order = np.lexsort((-remaining.astype(np.int64), solve_id))
    first = order[np.r_[True, solve_id[order][1:] != solve_id[order][:-1]]]  # longest-remaining row per solve
    pick = np.sort(np.random.default_rng(seed).choice(first, size=min(count, len(first)), replace=False))
    return states[pick], remaining[pick].astype(np.int64)


def load_policy(name: str, out: Path = ARTIFACTS, batch_size: int = 2048) -> Policy:
    """Log-probability policy of a model trained by models.train (artifacts/models/<name>.pt)."""
    import torch

    from models import graph
    from models.gnn import OneHotMLP, StickerGNN

    report = json.loads((out / "train_report.json").read_text())
    if name not in report:
        raise KeyError(f"no {name!r} entry in {out / 'train_report.json'}")
    if name == "mlp":
        model = OneHotMLP()

        def logits(states):
            return model(torch.from_numpy(states))
    else:
        cfg = report[name]
        home, color = cfg.get("home", False), cfg.get("color_edges", "all")
        model = StickerGNN(hidden=cfg["hidden"], steps=cfg["steps"], node_dim=graph.node_dim(home))
        moves = graph.move_tensors(FACE_TURNS, 3)

        def logits(states):
            return model(graph.batch_states(states, home, color), *moves)[0]

    model.load_state_dict(torch.load(out / "models" / f"{name}.pt"))
    model.eval()

    @torch.no_grad()
    def policy(states: np.ndarray) -> np.ndarray:
        chunks = [logits(states[i : i + batch_size]) for i in range(0, len(states), batch_size)]
        return torch.log_softmax(torch.cat(chunks), dim=-1).numpy()

    return policy


def _lengths(result: SearchResult) -> dict:
    found = [s for s in result.solutions if s is not None]
    if not found:
        return {"htm_mean": None, "htm_median": None, "qtm_mean": None}
    htm = np.array([len(s) for s in found])
    return {"htm_mean": round(float(htm.mean()), 1), "htm_median": float(np.median(htm)),
            "qtm_mean": round(float(np.mean([qtm(s) for s in found])), 1)}  # fmt: skip


def _summary(result: SearchResult) -> dict:
    return {"scrambles": len(result.solutions), "solve_rate": round(float(result.solved.mean()), 4),
            **_lengths(result), "nodes_mean": round(float(result.nodes.mean()), 1),
            "seconds": round(result.seconds, 1)}  # fmt: skip


def run_depth_suite(policy: Policy, max_depth: int, per_depth: int, seed: int, **search) -> dict:
    depths = np.repeat(np.arange(1, max_depth + 1), per_depth)
    states = random_walks(depths, seed)
    result = beam_search(states, policy, **search)
    verify(states, result.solutions)
    by_depth = {int(d): round(float(result.solved[depths == d].mean()), 3) for d in np.unique(depths)}
    shallow = depths <= BAR_DEPTH
    within = np.array([s is not None and qtm(s) <= BAR_DEPTH + BAR_SLACK for s in result.solutions])
    return {**_summary(result), "bar": round(float(within[shallow].mean()), 4), "by_depth": by_depth}


def run_cfop_suite(policy: Policy, count: int, seed: int, data_dir: Path = ARTIFACTS, **search) -> dict:
    states, human = cfop_scrambles(count, seed, data_dir)
    result = beam_search(states, policy, **search)
    verify(states, result.solutions)
    solved = result.solved
    return {**_summary(result), "human_htm_mean": round(float(human.mean()), 1),
            "human_htm_mean_on_solved": round(float(human[solved].mean()), 1) if solved.any() else None}  # fmt: skip


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mlp", help="a model name in artifacts/train_report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--per-depth", type=int, default=10)
    parser.add_argument("--cfop", type=int, default=200, help="test-split scrambles to solve (0 = skip)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    policy = load_policy(args.model, args.out)
    search = {"width": args.width, "max_moves": args.max_moves}
    entry = {"model": args.model, **search}
    if args.per_depth:
        entry["depth"] = run_depth_suite(policy, args.max_depth, args.per_depth, args.seed, **search)
        d = entry["depth"]
        print(f"depth suite: {d['solve_rate']:.1%} solved, bar {d['bar']:.1%}, by depth {d['by_depth']}")
    if args.cfop:
        entry["cfop"] = run_cfop_suite(policy, args.cfop, args.seed, args.out, **search)
        c = entry["cfop"]
        print(f"cfop suite: {c['solve_rate']:.1%} of {c['scrambles']} solved, {c['htm_mean']} moves "
              f"(human {c['human_htm_mean_on_solved']}), {c['seconds']:.0f} s")  # fmt: skip

    path = args.out / "search_report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    key = f"{args.model}/w{args.width}"
    report[key] = {**report.get(key, {}), **entry}  # a later run replaces only the suites it ran
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return entry


if __name__ == "__main__":
    main()
