"""Parse the Kaggle CFOP solves into our sticker convention and verify every solve.

Dataset: antbob/rubiks-cube-cfop-solutions, version 1 (the version arbCube's downloader
pins), unpacked into data/raw/ (gitignored). The formats, as checked against the files:

training.N      Two lines per solve (blank lines between records are skipped):
                  1. the scrambled cube: 54 space-separated color letters (r y g w o b),
                     six 9-sticker face blocks in the order L R U D B F
                  2. the full CFOP solution: space-separated move tokens
training.seq.N  The same solves, one move per record:
                  1. the current cube: 54 space-separated digits, face blocks in the order
                     L U F D R B, colors r=0 y=1 g=2 w=3 o=4 b=5
                  2. the move applied to that cube, or "#" on the solved cube ending a solve

Inside each face block the 9 stickers are already row-major as seen from outside, in the
Kociemba layout cube/sim.py uses. Of the 8^6 possible per-face re-orientations, only the
identity makes the recorded seq transitions agree with cube/sim.py. Converting a state is
therefore a block reorder plus a color relabel. Colors are relabeled from the scramble's
own centers (face turns never move centers): yellow U, orange R, green F, white D, red L,
blue B.

Solutions are normalized to the 18 face turns in FACE_TURNS. Whole-cube rotations are
removed by relabeling the moves that follow them, and wide or slice moves become face
turns plus a rotation, which is then relabeled away the same way.

Run from the repo root to verify every solve and print the report:
    .venv\\Scripts\\python -m data.parse_cfop [--check-seq]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from cube import sim

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
ARTIFACTS = REPO_ROOT / "artifacts"

SOLUTION_FACES = "LRUDBF"  # face block order in training.N
SEQ_FACES = "LUFDRB"  # face block order in training.seq.N
SEQ_DIGITS = "rygwob"  # digit k in training.seq.N stands for this color letter
END_OF_SOLVE = "#"

FACE_TURNS = tuple(face + suffix for face in sim.FACES for suffix in ("", "'", "2"))
FACE_TURN_INDEX = {move: i for i, move in enumerate(FACE_TURNS)}
SOLVED = sim.solved(3)


@dataclass
class Solve:
    solve_id: int
    shard: int
    index: int  # record number within the shard, from 0
    scramble: np.ndarray  # (54,) uint8 in our convention
    raw_moves: list[str]  # solution tokens as written in training.N
    moves: list[str]  # the same solution as the 18 face turns


def solve_id(shard: int, index: int) -> int:
    """Stable id for a solve: shard * 1,000,000 + record index (training.99 -> 99_000_000+)."""
    return shard * 1_000_000 + index


def shard_paths(raw_dir: Path = RAW_DIR, *, sequence: bool = False) -> list[Path]:
    """The numbered training.N (or training.seq.N) files under raw_dir, in numeric order."""
    prefix = "training.seq." if sequence else "training."
    paths = [
        p
        for p in Path(raw_dir).rglob("training.*")
        if p.is_file() and p.name.startswith(prefix) and p.name[len(prefix):].isdigit()
    ]
    if not paths:
        raise FileNotFoundError(f"no {prefix}N files under {raw_dir}")
    return sorted(paths, key=shard_number)


def shard_number(path: Path) -> int:
    return int(Path(path).name.rsplit(".", 1)[1])


def iter_records(path: Path) -> Iterator[tuple[int, list[str], str]]:
    """(line number, 54 state fields, label line) for each two-line record."""
    with open(path, encoding="utf-8") as stream:
        lines = ((n, line.strip()) for n, line in enumerate(stream, start=1))
        lines = ((n, line) for n, line in lines if line)
        for number, state in lines:
            label = next(lines, None)
            if label is None:
                raise ValueError(f"{path}:{number}: state line has no label line after it")
            yield number, state.split(), label[1]


def parse_state(fields: Sequence[str], face_order: str = SOLUTION_FACES) -> np.ndarray:
    """54 sticker fields in the dataset's face order -> our uint8 state.

    Each field is relabeled by which of our faces has that color as its center, so this
    works for both the color letters of training.N and the digits of training.seq.N.
    """
    if len(fields) != 54:
        raise ValueError(f"expected 54 stickers, got {len(fields)}")
    blocks = {face: fields[9 * i : 9 * i + 9] for i, face in enumerate(face_order)}
    color_of = {blocks[face][4]: k for k, face in enumerate(sim.FACES)}
    if len(color_of) != 6:
        raise ValueError("the six centers are not six different colors")
    try:
        state = np.array([color_of[x] for face in sim.FACES for x in blocks[face]], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"sticker {exc.args[0]!r} matches no center") from None
    if not (np.bincount(state, minlength=6) == 9).all():
        raise ValueError("each color must appear on exactly 9 stickers")
    return state


_LAYER_FACE = {(0, 0): "R", (0, 2): "L", (1, 0): "U", (1, 2): "D", (2, 0): "F", (2, 2): "B"}
_SUFFIX = {1: "", 2: "2", 3: "'"}


def _face_turn(axis: int, layer: int, turns: int) -> str:
    """Face-turn token for an outer layer; turns are clockwise as seen from the R/U/F side."""
    if layer == 2:  # L/D/B: clockwise from its own side is counterclockwise from R/U/F
        turns = 4 - turns
    return _LAYER_FACE[(axis, layer)] + _SUFFIX[turns]


@lru_cache(maxsize=None)
def _rotation_sources(axis: int, turns: int) -> tuple[str, ...]:
    """For each position U R F D L B, the position whose face a rotation brings there."""
    centers = sim.apply_move(SOLVED, sim.Move(axis, 0, 2, turns))[4::9]
    return tuple(sim.FACES[c] for c in centers)


def normalize_moves(tokens: Iterable[str]) -> list[str]:
    """Rewrite a 3x3 solution as the 18 face turns of a cube that is never rotated.

    ``frame[X]`` is the fixed-frame face currently sitting where the solver calls X. A face
    turn is relabeled through it. A rotation, or the rotation part of a wide or slice move,
    only updates it: on a 3x3, turning layers S of an axis by t equals rotating the whole
    cube by t and turning the outer layers not in S back by t.
    """
    frame = {face: face for face in sim.FACES}
    out = []
    for token in tokens:
        move = sim.parse_move(token, 3)
        layers = range(move.lo, move.hi + 1)
        if 1 not in layers:
            turn = _face_turn(move.axis, move.lo, move.turns)
            out.append(frame[turn[0]] + turn[1:])
            continue
        for layer in (0, 2):
            if layer not in layers:
                turn = _face_turn(move.axis, layer, 4 - move.turns)
                out.append(frame[turn[0]] + turn[1:])
        sources = _rotation_sources(move.axis, move.turns)
        frame = {pos: frame[src] for pos, src in zip(sim.FACES, sources)}
    return out


def token_kind(token: str) -> str:
    """'face turn', 'rotation', 'wide', 'slice' or 'unrecognized' for a 3x3 token."""
    try:
        move = sim.parse_move(token, 3)
    except ValueError:
        return "unrecognized"
    if (move.lo, move.hi) == (0, 2):
        return "rotation"
    if move.lo == move.hi:
        return "face turn" if move.lo != 1 else "slice"
    return "wide"


def iter_solves(
    raw_dir: Path = RAW_DIR,
    *,
    shards: Iterable[int] | None = None,
    limit_per_shard: int | None = None,
) -> Iterator[Solve]:
    """Every solve in the training.N files, converted and normalized."""
    wanted = None if shards is None else set(shards)
    for path in shard_paths(raw_dir):
        shard = shard_number(path)
        if wanted is not None and shard not in wanted:
            continue
        for index, (line, fields, solution) in enumerate(islice(iter_records(path), limit_per_shard)):
            try:
                raw = solution.split()
                yield Solve(solve_id(shard, index), shard, index, parse_state(fields), raw, normalize_moves(raw))
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line}: {exc}") from None


def verify(solve: Solve) -> tuple[bool, bool]:
    """(the raw solution solves the cube, the normalized one ends exactly on SOLVED)."""
    raw_ok = sim.is_solved(sim.apply_alg(solve.scramble, solve.raw_moves))
    normalized_ok = np.array_equal(sim.apply_alg(solve.scramble, solve.moves), SOLVED)
    return raw_ok, normalized_ok


def check_sequences(raw_dir: Path = RAW_DIR, *, limit_per_shard: int | None = None) -> dict:
    """Check training.seq.N against training.N, one recorded state per move.

    For each solve: the seq states must be the scramble replayed move by move through
    cube/sim.py, the seq moves must be the training.N solution, the solve must end with
    '#' on a solved cube, and each digit must be the SEQ_DIGITS color of training.N.
    """
    gather = np.array([SEQ_FACES.index(f) * 9 + k for f in sim.FACES for k in range(9)])
    result = {"solves": 0, "states": 0, "mismatched_solves": 0, "examples": [], "count_mismatch": {}}
    seq_paths = {shard_number(p): p for p in shard_paths(raw_dir, sequence=True)}
    for path in shard_paths(raw_dir):
        shard = shard_number(path)
        if shard not in seq_paths:
            continue
        seq = iter_records(seq_paths[shard])
        solutions = 0
        for index, (_, letters, solution) in enumerate(islice(iter_records(path), limit_per_shard)):
            solutions += 1
            raw = solution.split()
            scramble = parse_state(letters)
            letter_of = {letters[9 * SOLUTION_FACES.index(f) + 4]: f for f in sim.FACES}
            color_of_digit = np.array([sim.FACES.index(letter_of[c]) for c in SEQ_DIGITS], dtype=np.uint8)
            state, problems = scramble, []
            for step in range(len(raw) + 1):
                record = next(seq, None)
                if record is None:
                    problems.append("seq file ended early")
                    break
                _, digits, label = record
                recorded = color_of_digit[np.array(digits, dtype=np.uint8)[gather]]
                result["states"] += 1
                if not np.array_equal(recorded, state):
                    problems.append(f"state {step} differs")
                expected = raw[step] if step < len(raw) else END_OF_SOLVE
                if label != expected:
                    problems.append(f"move {step}: {label!r} != {expected!r}")
                    break
                if step < len(raw):
                    state = sim.apply_move(state, raw[step])
            if not sim.is_solved(state):
                problems.append("final state not solved")
            result["solves"] += 1
            if problems:
                result["mismatched_solves"] += 1
                if len(result["examples"]) < 5:
                    result["examples"].append({"shard": shard, "index": index, "problems": problems[:3]})
        if limit_per_shard is None:
            leftover = sum(1 for record in seq if record[2] == END_OF_SOLVE)
            if leftover:
                result["count_mismatch"][shard] = f"{leftover} extra seq solves after {solutions}"
    return result


def report(raw_dir: Path = RAW_DIR, *, limit_per_shard: int | None = None) -> dict:
    """Solve counts, verification rate, every move token seen, and solution length stats."""
    per_shard = Counter()
    raw_ok = normalized_ok = 0
    failures = []
    tokens = Counter()
    raw_lengths, normalized_lengths = [], []
    shards_of_scramble = defaultdict(set)
    copies = Counter()
    for solve in iter_solves(raw_dir, limit_per_shard=limit_per_shard):
        per_shard[solve.shard] += 1
        a, b = verify(solve)
        raw_ok += a
        normalized_ok += b
        if not (a and b) and len(failures) < 10:
            failures.append({"solve_id": solve.solve_id, "raw_ok": a, "normalized_ok": b})
        tokens.update(solve.raw_moves)
        raw_lengths.append(len(solve.raw_moves))
        normalized_lengths.append(len(solve.moves))
        key = solve.scramble.tobytes()
        copies[key] += 1
        shards_of_scramble[key].add(solve.shard)

    total = sum(per_shard.values())

    def stats(values):
        v = np.asarray(values)
        pct = np.percentile(v, [5, 25, 50, 75, 95])
        return {
            "min": int(v.min()), "p5": float(pct[0]), "p25": float(pct[1]), "median": float(pct[2]),
            "mean": round(float(v.mean()), 2), "p75": float(pct[3]), "p95": float(pct[4]),
            "max": int(v.max()), "std": round(float(v.std()), 2),
        }  # fmt: skip

    by_kind = defaultdict(dict)
    for token, count in sorted(tokens.items(), key=lambda kv: (-kv[1], kv[0])):
        by_kind[token_kind(token)][token] = count
    duplicated = [k for k, c in copies.items() if c > 1]
    return {
        "solves": total,
        "solves_per_shard": dict(sorted(per_shard.items())),
        "raw_solution_solves_cube": raw_ok,
        "normalized_solution_ends_solved": normalized_ok,
        "verified_percent": round(100 * min(raw_ok, normalized_ok) / total, 4) if total else 0.0,
        "failures": failures,
        "tokens_by_kind": dict(by_kind),
        "distinct_tokens": len(tokens),
        "raw_length": stats(raw_lengths),
        "normalized_length": stats(normalized_lengths),
        "duplicate_scrambles": {
            "distinct_scrambles_repeated": len(duplicated),
            "extra_copies": sum(copies[k] - 1 for k in duplicated),
            "repeated_across_shards": sum(len(shards_of_scramble[k]) > 1 for k in duplicated),
        },
    }


def _print_report(rep: dict) -> None:
    print(f"solves: {rep['solves']:,}  per shard: {rep['solves_per_shard']}")
    print(
        f"verified: {rep['verified_percent']}%  "
        f"(raw solution solves the cube: {rep['raw_solution_solves_cube']:,}, "
        f"normalized ends solved: {rep['normalized_solution_ends_solved']:,})"
    )
    for kind, toks in rep["tokens_by_kind"].items():
        flag = "" if kind == "face turn" else "   <-- flagged"
        print(f"{kind} tokens ({len(toks)}){flag}: " + ", ".join(f"{t} {c:,}" for t, c in toks.items()))
    for name in ("raw_length", "normalized_length"):
        print(f"{name}: {rep[name]}")
    print(f"duplicate scrambles: {rep['duplicate_scrambles']}")
    if rep.get("seq_check"):
        print(f"seq check: {rep['seq_check']}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--limit-per-shard", type=int, default=None, help="only the first N solves per shard")
    parser.add_argument("--check-seq", action="store_true", help="also check training.seq.N against training.N")
    parser.add_argument("--json", type=Path, default=ARTIFACTS / "cfop_parse_report.json")
    args = parser.parse_args(argv)
    rep = report(args.raw_dir, limit_per_shard=args.limit_per_shard)
    if args.check_seq:
        rep["seq_check"] = check_sequences(args.raw_dir, limit_per_shard=args.limit_per_shard)
    _print_report(rep)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
