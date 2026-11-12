import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pandas as pd

    from arbcube.bpe import bpe_move_block_lengths, bpe_solution_length_curve
    from arbcube.data import load_solutions

    return (
        Counter,
        MaxNLocator,
        bpe_move_block_lengths,
        bpe_solution_length_curve,
        load_solutions,
        mo,
        pd,
        plt,
    )


@app.cell
def _(load_solutions):
    rows_per_shard = None
    solutions = load_solutions(limit_per_shard=rows_per_shard)
    return (solutions,)


@app.cell
def _(plt, solutions):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.hist(solutions.n_moves, bins=25, edgecolor="white")
    _ax.set(title="Solution lengths across all complete solutions", xlabel="Moves per solution", ylabel="Solutions")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(Counter, pd, plt, solutions):
    _counts = Counter(
        move for solution in solutions.solution for move in solution.split()
    )
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    pd.Series(_counts).sort_values(ascending=False).plot.bar(ax=_ax)
    _ax.set(title="Move counts across all complete solutions", xlabel="Move", ylabel="Occurrences")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Learn move blocks of different lengths with BPE

    A **move block** is a single move or a learned sequence of moves.
    Start with individual moves, then repeatedly merge the most common adjacent
    pair of move blocks. Those blocks can merge again into longer sequences.
    `R`, `R'`, and `R2` stay distinct, and merges stay within one solution.

    There is no fixed sequence length. More merges allow a larger vocabulary;
    this is a greedy discovery method, not an exhaustive list of all sequences.
    """)
    return


@app.cell
def _(mo):
    bpe_merges = mo.ui.slider(
        start=20, stop=1_000, step=20, value=100, show_value=True,
        label="Move block merges", debounce=True,
    )
    bpe_merges
    return (bpe_merges,)


@app.cell
def _(mo):
    mo.md("""
    ### Solution length as the move library grows

    How many move blocks does a solution need as we add learned sequences? The
    move library includes single moves plus learned blocks of multiple moves.
    The first point uses individual moves only; the curve ends at the merge
    budget selected above, or earlier if no more pairs meet the frequency limit.

    Length is measured in **move blocks per solution**, so the underlying move
    sequences stay the same. Bands show quantiles across loaded solutions,
    not uncertainty in the mean. These are training-corpus compression results.
    """)
    return


@app.cell
def _(bpe_merges, bpe_solution_length_curve, solutions):
    bpe_length_curve = bpe_solution_length_curve(
        solutions.solution, num_merges=bpe_merges.value,
        merge_step=max(1, bpe_merges.value // 20),
    )
    return (bpe_length_curve,)


@app.cell
def _(bpe_length_curve, mo, plt):
    mo.stop(bpe_length_curve.empty, mo.md("Load solutions to see the length curve."))
    _curve = bpe_length_curve
    _x = _curve["Vocabulary size"]
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.fill_between(_x, _curve["10%"], _curve["90%"], color="#2563eb", alpha=0.12, label="10–90% of solutions")
    _ax.fill_between(_x, _curve["25%"], _curve["75%"], color="#2563eb", alpha=0.24, label="25–75% of solutions")
    _ax.plot(_x, _curve.Mean, color="#2563eb", marker="o", markersize=3, label="Mean")
    _ax.set(
        title="Solution length as the move library grows",
        xlabel="Number of distinct move blocks",
        xscale="linear",
        ylabel="Move blocks per solution", ylim=(0, None),
    )
    _ax.grid(alpha=0.2)
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Move block lengths at the selected vocabulary size

    Use the merge slider above to change the move library. Each distinct block
    is counted once in this histogram, including single moves.
    """)
    return


@app.cell
def _(bpe_merges, bpe_move_block_lengths, solutions):
    move_block_lengths = bpe_move_block_lengths(
        solutions.solution, num_merges=bpe_merges.value,
    )
    return (move_block_lengths,)


@app.cell
def _(MaxNLocator, mo, move_block_lengths, plt):
    mo.stop(move_block_lengths.empty, mo.md("Load solutions to see move block lengths."))
    _longest = int(move_block_lengths.max())
    _bins = [length - 0.5 for length in range(1, _longest + 2)]
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.hist(move_block_lengths, bins=_bins, color="#2563eb", edgecolor="white")
    _ax.set(
        title=f"Move block lengths ({len(move_block_lengths):,} distinct blocks)",
        xlabel="Moves per block", ylabel="Distinct move blocks",
        xlim=(0.5, _longest + 0.5),
    )
    _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


if __name__ == "__main__":
    app.run()
