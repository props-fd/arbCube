"""Draw the result figures in results/figures/ from results/*.json.

    .venv\\Scripts\\python results\\plot.py

The cubetransfer curve is copied from cubetransfer.zip (web/diagnostics.json,
horizon_333, piece-slot graph, beam 32, 1M training states) in the arbCube repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
TRAIN = json.loads((HERE / "train_report.json").read_text())
SEARCH = json.loads((HERE / "search_report.json").read_text())

# Fixed categorical order (validated: dataviz palette, light surface); one color per model everywhere.
COLOR = {"mlp": "#2a78d6", "gnn_v2": "#eb6834", "gnn_v1": "#1baf7a", "cubetransfer": "#eda100"}
NAME = {"mlp": "MLP (one-hot)", "gnn_v2": "GNN v2 (home face)", "gnn_v1": "GNN v1", "cubetransfer": "cubetransfer GNN"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CUBETRANSFER_BEAM32 = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0,
                       11: 0.4, 12: 0.17, 13: 0.0, 14: 0.09}  # fmt: skip

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})  # fmt: skip


def pct_axis(ax, top=1.0):
    ax.set_ylim(0, top * 1.04)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def solve_vs_width():
    widths = sorted(int(k.split("/w")[1]) for k in SEARCH if k.startswith("mlp/"))
    rows = [SEARCH[f"mlp/w{w}"]["cfop"] for w in widths]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(widths, [r["solve_rate"] for r in rows], marker="o", color=COLOR["mlp"], label=NAME["mlp"])
    for w, r in zip(widths, rows):
        a.annotate(f"{r['solve_rate']:.1%}", (w, r["solve_rate"]), textcoords="offset points", xytext=(0, -15),
                   ha="center", color=INK2, fontsize=9)  # fmt: skip
    g = SEARCH.get("gnn_v2/w32", {}).get("cfop")
    if g:
        a.plot([32], [g["solve_rate"]], marker="o", linestyle="none", color=COLOR["gnn_v2"], label=NAME["gnn_v2"])
        a.annotate(f"{g['solve_rate']:.0%}", (32, g["solve_rate"]), textcoords="offset points", xytext=(10, 0),
                   va="center", color=INK2, fontsize=9)  # fmt: skip
    a.set_title("Full scrambles solved")
    a.set_xlabel("beam width")
    pct_axis(a)
    a.legend(loc="center right")
    b.plot(widths, [r["htm_mean"] for r in rows], marker="o", color=COLOR["mlp"], label=NAME["mlp"])
    human = sum(r["human_htm_mean"] for r in rows) / len(rows)
    b.axhline(human, color=MUTED, linestyle="--", linewidth=1.5)
    b.annotate(f"human CFOP solution ({human:.0f})", (widths[0], human), textcoords="offset points",
               xytext=(0, -14), ha="left", color=INK2, fontsize=9)  # fmt: skip
    b.set_title("Mean solution length (solved scrambles)")
    b.set_xlabel("beam width")
    b.set_ylabel("face turns")
    b.set_ylim(0, max(r["htm_mean"] for r in rows) * 1.15)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.minorticks_off()
    fig.tight_layout()
    save(fig, "solve_vs_beam_width.png")


def depth_suite():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    series = [("mlp/w256", "mlp", "-", "MLP, beam 256"), ("mlp/w32", "mlp", ":", "MLP, beam 32"),
              ("gnn_v2/w32", "gnn_v2", "-", "GNN v2, beam 32")]  # fmt: skip
    for key, model, style, label in series:
        if key not in SEARCH or "depth" not in SEARCH[key]:
            continue
        by = {int(d): v for d, v in SEARCH[key]["depth"]["by_depth"].items()}
        ax.plot(list(by), list(by.values()), style, marker="o", color=COLOR[model], label=label)
    ax.plot(list(CUBETRANSFER_BEAM32), list(CUBETRANSFER_BEAM32.values()), "--", marker="s",
            color=COLOR["cubetransfer"], label="cubetransfer GNN, beam 32 (random-walk training)")  # fmt: skip
    ax.set_title("3x3 scrambles solved, by scramble depth")
    ax.set_xlabel("scramble depth (random quarter turns from solved)")
    ax.set_xticks(range(1, 21))
    pct_axis(ax)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, "solve_by_depth.png")


def stage_accuracy():
    stages = ["cross", "f2l", "oll", "pll"]
    labels = ["Cross", "F2L", "OLL", "PLL", "Overall"]
    models = [m for m in ("mlp", "gnn_v2", "gnn_v1") if m in TRAIN]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bar = 0.26
    for k, m in enumerate(models):
        test = TRAIN[m]["test"]
        values = [test["by_stage"][s] for s in stages] + [test["top1"]]
        xs = [i + (k - (len(models) - 1) / 2) * (bar + 0.02) for i in range(len(labels))]
        ax.bar(xs, values, width=bar, color=COLOR[m], label=NAME[m], zorder=2)
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.0%}", (x, v), textcoords="offset points", xytext=(0, 2), ha="center",
                        color=INK2, fontsize=7.5)  # fmt: skip
    base = TRAIN["most_common_move"]["test"]["top1"]
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.2, zorder=1,
               label=f"always '{TRAIN['most_common_move']['move']}' ({base:.0%})")  # fmt: skip
    ax.set_xticks(range(len(labels)), labels)
    ax.grid(axis="x", visible=False)
    ax.set_title("Next-move accuracy (top-1) on held-out CFOP solves, by stage")
    pct_axis(ax)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    fig.tight_layout()
    save(fig, "accuracy_by_stage.png")


def training_curves():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for m in ("mlp", "gnn_v2", "gnn_v1"):
        if m not in TRAIN:
            continue
        hist = TRAIN[m]["history"]
        epochs, val = [h["epoch"] for h in hist], [h["val_top1"] for h in hist]
        minutes = TRAIN[m]["train_seconds"] / 60
        ax.plot(epochs, val, marker="o", markersize=4, color=COLOR[m], label=f"{NAME[m]} ({minutes:.0f} min CPU)")
        ax.annotate(f"{val[-1]:.1%}", (epochs[-1], val[-1]), textcoords="offset points", xytext=(6, 0),
                    va="center", color=INK2, fontsize=9)  # fmt: skip
    ax.set_title("Validation next-move accuracy during training")
    ax.set_xlabel("epoch (100,000 training rows each)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    pct_axis(ax)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "training_curves.png")


if __name__ == "__main__":
    solve_vs_width()
    depth_suite()
    stage_accuracy()
    training_curves()
