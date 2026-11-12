"""Train the one-hot MLP baseline and the sticker GNN to predict the next CFOP move.

Both models train on the same ``--rows`` random rows of artifacts/cfop_train.npz and are
scored on the same ``--eval-rows`` random rows of the val and test splits: top-1
next-move accuracy overall and per CFOP stage, next to always guessing the most common
move. The GNN also trains its value head on moves_remaining. Everything runs on CPU.
Writes artifacts/train_report.json and the weights to artifacts/models/.

    .venv\\Scripts\\python -m models.train                        # both models, 100k rows
    .venv\\Scripts\\python -m models.train --model mlp --mlp-epochs 2 --rows 20000
    .venv\\Scripts\\python -m models.train --model gnn --name gnn_v1 --no-home --color-edges all --hidden 64

``--name`` keys the GNN's entry in the report and its checkpoint, so runs with different
graph options sit side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.parse_cfop import ARTIFACTS, FACE_TURNS
from models import graph
from models.gnn import OneHotMLP, StickerGNN, parameter_count

STAGES = ("cross", "f2l", "oll", "pll")
FIELDS = ("state", "next_move", "moves_remaining", "cfop_stage", "solve_id")
VALUE_SCALE = 20.0  # moves_remaining is divided by this for the value loss
VALUE_WEIGHT = 0.1


def load_rows(split: str, rows: int, seed: int, data_dir: Path = ARTIFACTS) -> dict[str, np.ndarray]:
    """``rows`` random rows (fixed by ``seed``) of artifacts/cfop_{split}.npz."""
    with np.load(data_dir / f"cfop_{split}.npz") as d:
        if tuple(d["move_names"].tolist()) != FACE_TURNS:
            raise ValueError("the dataset's move vocabulary is not data.parse_cfop.FACE_TURNS")
        total = len(d["next_move"])
        idx = np.sort(np.random.default_rng(seed).choice(total, size=min(rows, total), replace=False))
        return {k: d[k][idx] for k in FIELDS}


def _batches(count: int, size: int, rng: np.random.Generator | None = None):
    order = rng.permutation(count) if rng is not None else np.arange(count)
    for i in range(0, count, size):
        yield order[i : i + size]


def _scores(pred: np.ndarray, rows: dict[str, np.ndarray]) -> dict:
    correct = pred == rows["next_move"]
    stages = {s: round(float(correct[rows["cfop_stage"] == k].mean()), 4) for k, s in enumerate(STAGES)}
    return {"top1": round(float(correct.mean()), 4), "by_stage": stages, "rows": int(len(correct))}


class Trainer:
    """Shared loop: AdamW, cosine learning rate, per-epoch val accuracy."""

    def __init__(self, model: nn.Module, epochs: int, steps_per_epoch: int, lr: float):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1
        )

    def step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sched.step()


def train_mlp(train, evals, *, epochs, batch_size, lr, seed, log=print):
    torch.manual_seed(seed)
    model = OneHotMLP()
    states = torch.from_numpy(train["state"])
    target = torch.from_numpy(train["next_move"].astype(np.int64))
    steps = math.ceil(len(target) / batch_size)
    trainer, rng, history = Trainer(model, epochs, steps, lr), np.random.default_rng(seed), []
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in _batches(len(target), batch_size, rng):
            loss = loss_fn(model(states[idx]), target[idx])
            trainer.step(loss)
            total += float(loss) * len(idx)
        val = _scores(predict_mlp(model, evals["val"]), evals["val"])["top1"]
        history.append({"epoch": epoch + 1, "train_loss": round(total / len(target), 4), "val_top1": val})
        log(f"  mlp epoch {epoch + 1}/{epochs}: loss {total / len(target):.4f}, val top-1 {val:.4f}")
    return model, history, time.perf_counter() - start


@torch.no_grad()
def predict_mlp(model, rows, batch_size=4096):
    model.eval()
    states = torch.from_numpy(rows["state"])
    return np.concatenate([model(states[i]).argmax(-1).numpy() for i in _batches(len(states), batch_size)])


def train_gnn(train, evals, *, epochs, batch_size, lr, seed, hidden, steps, home, color, log=print):
    torch.manual_seed(seed)
    model = StickerGNN(hidden=hidden, steps=steps, node_dim=graph.node_dim(home))
    moves = graph.move_tensors(FACE_TURNS, 3)
    target = torch.from_numpy(train["next_move"].astype(np.int64))
    value_target = torch.from_numpy(train["moves_remaining"].astype(np.float32)) / VALUE_SCALE
    n_steps = math.ceil(len(target) / batch_size)
    trainer, rng, history = Trainer(model, epochs, n_steps, lr), np.random.default_rng(seed), []
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in _batches(len(target), batch_size, rng):
            logits, value = model(graph.batch_states(train["state"][idx], home, color), *moves)
            loss = ce(logits, target[idx]) + VALUE_WEIGHT * mse(value, value_target[idx])
            trainer.step(loss)
            total += float(loss) * len(idx)
        pred, _ = predict_gnn(model, evals["val"], home, color)
        val = _scores(pred, evals["val"])["top1"]
        history.append({"epoch": epoch + 1, "train_loss": round(total / len(target), 4), "val_top1": val})
        log(f"  gnn epoch {epoch + 1}/{epochs}: loss {total / len(target):.4f}, val top-1 {val:.4f} "
            f"({time.perf_counter() - start:.0f} s)")  # fmt: skip
    return model, history, time.perf_counter() - start


@torch.no_grad()
def predict_gnn(model, rows, home, color, batch_size=1024):
    model.eval()
    moves = graph.move_tensors(FACE_TURNS, 3)
    preds, values = [], []
    for idx in _batches(len(rows["next_move"]), batch_size):
        logits, value = model(graph.batch_states(rows["state"][idx], home, color), *moves)
        preds.append(logits.argmax(-1).numpy())
        values.append(value.numpy() * VALUE_SCALE)
    return np.concatenate(preds), np.concatenate(values)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("mlp", "gnn", "both"), default="both")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--eval-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--gnn-epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6, help="times the shared GINE layer is applied")
    parser.add_argument("--home", action=argparse.BooleanOptionalAction, default=True,
                        help="home-face node features (graph.home_faces)")
    parser.add_argument("--color-edges", choices=graph.COLOR_MODES, default="center")
    parser.add_argument("--name", default="gnn", help="report key and checkpoint name of the GNN run")
    parser.add_argument("--data-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    train = load_rows("train", args.rows, args.seed, args.data_dir)
    evals = {s: load_rows(s, args.eval_rows, args.seed + 1 + k, args.data_dir) for k, s in enumerate(("val", "test"))}
    common = int(np.bincount(train["next_move"], minlength=18).argmax())
    report = {
        "train_rows": len(train["next_move"]),
        "train_solves": int(len(np.unique(train["solve_id"]))),
        "eval_rows": {s: len(r["next_move"]) for s, r in evals.items()},
        "torch_threads": torch.get_num_threads(),
        "most_common_move": {
            "move": FACE_TURNS[common],
            **{s: _scores(np.full(len(r["next_move"]), common), r) for s, r in evals.items()},
        },
    }
    models_dir = args.out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("mlp", "both"):
        print(f"MLP baseline on {report['train_rows']:,} rows")
        model, history, seconds = train_mlp(train, evals, epochs=args.mlp_epochs, batch_size=512, lr=1e-3, seed=args.seed)
        report["mlp"] = {
            "params": parameter_count(model), "epochs": args.mlp_epochs, "train_seconds": round(seconds, 1),
            "history": history, **{s: _scores(predict_mlp(model, r), r) for s, r in evals.items()},
        }  # fmt: skip
        torch.save(model.state_dict(), models_dir / "mlp.pt")

    if args.model in ("gnn", "both"):
        options = {"home": args.home, "color": args.color_edges}
        print(f"GNN {args.name} (hidden {args.hidden}, {args.steps} steps, {options}) on {report['train_rows']:,} rows")
        model, history, seconds = train_gnn(
            train, evals, epochs=args.gnn_epochs, batch_size=256, lr=2e-3, seed=args.seed,
            hidden=args.hidden, steps=args.steps, **options,
        )  # fmt: skip
        entry = {"params": parameter_count(model), "epochs": args.gnn_epochs, "hidden": args.hidden,
                 "steps": args.steps, "home": args.home, "color_edges": args.color_edges,
                 "train_seconds": round(seconds, 1), "history": history}  # fmt: skip
        for s, r in evals.items():
            pred, value = predict_gnn(model, r, **options)
            entry[s] = {**_scores(pred, r), "value_mae_moves": round(float(np.abs(value - r["moves_remaining"]).mean()), 2)}
        report[args.name] = entry
        torch.save(model.state_dict(), models_dir / f"{args.name}.pt")

    path = args.out / "train_report.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(report)
    path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    for name in ("mlp", args.name):
        if name in report:
            r = report[name]
            print(f"{name}: val top-1 {r['val']['top1']:.4f}, test top-1 {r['test']['top1']:.4f}, "
                  f"{r['params']:,} params, {r['train_seconds']:.0f} s")  # fmt: skip
    print(f"most common move ({report['most_common_move']['move']}): test top-1 "
          f"{report['most_common_move']['test']['top1']:.4f}")  # fmt: skip
    print(f"wrote {path}")
    return report


if __name__ == "__main__":
    main()
