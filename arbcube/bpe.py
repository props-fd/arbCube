"""Learn variable-length move sequences with BPE, keeping whole moves intact."""

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import batched
import json
import re

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def find_sequence_occurrences(
    solution_strings: Iterable[str], sequence: str
) -> pd.DataFrame:
    """Locate exact, overlapping matches of whole moves within each solution.

    `solution_index` and `start_move` are zero-based. `start_fraction` locates
    the first move of the match on a 0-to-1 scale across the solution's moves.
    """
    phrase = sequence.split()
    if not phrase:
        raise ValueError("sequence must contain at least one move")
    pattern = re.compile(r"(?=(?<!\S)" + re.escape(" ".join(phrase)) + r"(?!\S))")
    matches = []
    for solution_index, solution in enumerate(solution_strings):
        moves = solution.split()
        text = " ".join(moves)
        for match in pattern.finditer(text):
            start_move = text.count(" ", 0, match.start())
            matches.append((
                solution_index, start_move, len(moves),
                start_move / max(1, len(moves) - 1),
            ))
    return pd.DataFrame(
        matches,
        columns=["solution_index", "start_move", "solution_moves", "start_fraction"],
    )


def _count_occurrences(corpus: Sequence[str], patterns: Sequence[str]) -> Counter:
    """Count all pattern matches, including overlaps, within each solution."""
    trie = {}
    for pattern in patterns:
        node = trie
        for symbol in pattern:
            node = node.setdefault(symbol, {})
        node[None] = pattern

    counts = Counter()
    if not trie:
        return counts
    for text in corpus:
        for start in range(len(text)):
            node = trie
            for end in range(start, len(text)):
                node = node.get(text[end])
                if node is None:
                    break
                pattern = node.get(None)
                if pattern is not None:
                    counts[pattern] += 1
    return counts


def _move_corpus(solution_strings: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    solutions = list(solution_strings)
    moves = sorted({move for solution in solutions for move in solution.split()})
    # Private-use characters are lossless stand-ins for individual moves.
    symbols = {move: chr(0xE000 + i) for i, move in enumerate(moves)}
    corpus = [
        "".join(symbols[move] for move in solution.split()) for solution in solutions
    ]
    return corpus, symbols


def _train_bpe(corpus: list[str], symbols: dict[str, str], num_merges: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=len(symbols) + num_merges,
        min_frequency=2,
        show_progress=False,
        initial_alphabet=list(symbols.values()),
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    return tokenizer


def bpe_move_block_lengths(
    solution_strings: Iterable[str], *, num_merges: int = 100
) -> pd.Series:
    """Count moves in each distinct vocabulary block, including single moves.

    Each available block contributes once, regardless of how often it occurs
    in the encoded solutions. The result's length is the actual vocabulary size.
    """
    if num_merges < 0:
        raise ValueError("num_merges must be nonnegative")
    corpus, symbols = _move_corpus(solution_strings)
    if not symbols or num_merges == 0:
        lengths = [1] * len(symbols)
    else:
        vocabulary = _train_bpe(corpus, symbols, num_merges).get_vocab()
        lengths = sorted(len(block) for block in vocabulary)
    return pd.Series(lengths, dtype="int64", name="Moves per block")


def bpe_solution_length_curve(
    solution_strings: Iterable[str], *, num_merges: int = 100, merge_step: int = 20
) -> pd.DataFrame:
    """Summarize encoded solution lengths as the BPE vocabulary grows.

    Train once, then encode with successive prefixes of the learned merge list.
    Include the unmerged baseline and the final learned vocabulary, even when
    training stops before the requested budget or between checkpoints. Vocabulary
    size includes individual moves. Lengths and quantiles count move blocks per
    solution, measured on the same corpus used for training (not held-out data).
    """
    if num_merges < 0:
        raise ValueError("num_merges must be nonnegative")
    if merge_step < 1:
        raise ValueError("merge_step must be positive")
    columns = ["Merges", "Vocabulary size", "Mean", "10%", "25%", "Median", "75%", "90%"]
    corpus, symbols = _move_corpus(solution_strings)
    if not corpus:
        return pd.DataFrame(columns=columns)

    def summarize(lengths, merges, vocabulary_size):
        lengths = pd.Series(lengths, dtype="int64")
        quantiles = lengths.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
        return [merges, vocabulary_size, lengths.mean(), *quantiles]

    rows = [summarize([len(text) for text in corpus], 0, len(symbols))]
    if symbols and num_merges:
        model = json.loads(_train_bpe(corpus, symbols, num_merges).to_str())["model"]
        merges = [tuple(pair) for pair in model["merges"]]
        checkpoints = sorted(set(range(merge_step, len(merges) + 1, merge_step)) | {len(merges)})
        for count in checkpoints:
            if count == 0:
                continue
            tokens = set(symbols.values()) | {left + right for left, right in merges[:count]}
            vocabulary = {token: model["vocab"][token] for token in tokens}
            tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=merges[:count]))
            lengths = []
            for batch in batched(corpus, 1_024):
                lengths.extend(len(encoding.ids) for encoding in tokenizer.encode_batch(list(batch)))
            rows.append(summarize(lengths, count, len(vocabulary)))
    return pd.DataFrame(rows, columns=columns)


def learn_move_bpe(
    solution_strings: Iterable[str], *, num_merges: int = 100
) -> pd.DataFrame:
    """Learn up to `num_merges` multi-move tokens across separate solutions.

    Each distinct move is mapped to one character before training; no text
    normalization or pre-tokenization is applied. Thus moves cannot be split,
    and each training string remains a separate solution. Token length is
    unconstrained; the merge budget controls how many patterns can be learned.

    `Occurrences` counts exact matches in the original solutions, including
    overlaps. `BPE uses` counts tokens in the final non-overlapping segmentation.
    BPE greedily discovers a vocabulary, not every possible frequent sequence.
    Requires the project's development dependencies (`uv sync`).
    """
    if num_merges < 0:
        raise ValueError("num_merges must be nonnegative")
    columns = ["Sequence", "Moves", "Occurrences", "BPE uses"]
    corpus, symbols = _move_corpus(solution_strings)
    if not symbols or num_merges == 0:
        return pd.DataFrame(columns=columns)

    reverse_symbols = {symbol: move for move, symbol in symbols.items()}
    tokenizer = _train_bpe(corpus, symbols, num_merges)
    patterns = [token for token in tokenizer.get_vocab() if len(token) > 1]
    occurrences = _count_occurrences(corpus, patterns)

    uses = Counter()
    for batch in batched(corpus, 1_024):
        for encoding in tokenizer.encode_batch(list(batch)):
            uses.update(encoding.tokens)

    return pd.DataFrame(
        [
            {
                "Sequence": " ".join(reverse_symbols[symbol] for symbol in token),
                "Moves": len(token),
                "Occurrences": occurrences[token],
                "BPE uses": uses[token],
            }
            for token in patterns
        ],
        columns=columns,
    ).sort_values(
        ["Occurrences", "Moves", "Sequence"],
        ascending=[False, False, True],
        ignore_index=True,
    )
