"""Download and read the Kaggle CFOP dataset without loading sequence files whole."""

import argparse
import re
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_HANDLE = "antbob/rubiks-cube-cfop-solutions"
# Version 1 was the latest version when this dataset was added.
DATASET_VERSION = 1
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / f"cfop-v{DATASET_VERSION}"


def download_dataset(version: int = DATASET_VERSION) -> Path:
    """Download a pinned dataset version into the repo's ignored data directory."""
    import kagglehub

    if version < 1:
        raise ValueError("Dataset version must be a positive integer.")
    output_dir = REPO_ROOT / "data" / "raw" / f"cfop-v{version}"
    return Path(
        kagglehub.dataset_download(
            f"{DATASET_HANDLE}/versions/{version}", output_dir=str(output_dir)
        )
    )


def shard_files(data_dir: Path = DEFAULT_DATA_DIR, *, sequence: bool = False) -> list[Path]:
    """Find the numbered shards, preserving their original names and numeric order."""
    pattern = r"training\.seq\.(\d+)" if sequence else r"training\.(\d+)"
    files = [
        p for p in Path(data_dir).rglob("training.*")
        if p.is_file() and re.fullmatch(pattern, p.name)
    ]
    if not files:
        raise FileNotFoundError(
            f"No {'sequence' if sequence else 'solution'} shards in {data_dir}. "
            "Run `uv run python -m arbcube.data` first."
        )
    return sorted(files, key=lambda p: int(p.name.rsplit(".", 1)[1]))


def iter_pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Read alternating state/label lines; allow blank separator lines between pairs."""
    with Path(path).open(encoding="utf-8") as stream:
        lines = enumerate(stream, start=1)
        for line_number, state in lines:
            if not state.strip():
                continue
            label_line = next(lines, None)
            if label_line is None or not label_line[1].strip():
                raise ValueError(f"{path}:{line_number}: state has no label")
            yield " ".join(state.split()), " ".join(label_line[1].split())


def _check_limit(limit_per_shard: int | None) -> None:
    if limit_per_shard is not None and limit_per_shard < 1:
        raise ValueError("limit_per_shard must be positive or None.")


def load_solutions(
    data_dir: Path = DEFAULT_DATA_DIR, *, limit_per_shard: int | None = None
) -> pd.DataFrame:
    """Load complete solutions; each record has a color state and move sequence.

    `n_moves` counts whitespace-separated tokens (a double turn counts once).
    Shard names and within-shard row indices are retained for split auditing.
    """
    _check_limit(limit_per_shard)
    records = []
    for path in shard_files(data_dir):
        for index, (state, solution) in enumerate(islice(iter_pairs(path), limit_per_shard)):
            records.append((path.name, index, state, solution, len(solution.split())))
    return pd.DataFrame(
        records, columns=["shard", "record_index", "state", "solution", "n_moves"]
    )


def load_transitions(
    data_dir: Path = DEFAULT_DATA_DIR, *, limit_per_shard: int = 1_000
) -> pd.DataFrame:
    """Read a bounded prefix of each sequence shard, including terminal `#` labels.

    This is a convenience sample, not a random sample of all transitions. The raw
    records contain a state and label; a next-state relationship is not inferred.
    """
    if limit_per_shard is None:
        raise ValueError("Sequence loading requires a finite limit_per_shard.")
    _check_limit(limit_per_shard)
    records = []
    for path in shard_files(data_dir, sequence=True):
        for index, (state, action) in enumerate(islice(iter_pairs(path), limit_per_shard)):
            records.append((path.name, index, state, action, action == "#"))
    return pd.DataFrame(
        records, columns=["shard", "record_index", "state", "action", "is_terminal"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", type=int, default=DATASET_VERSION,
        help="Kaggle dataset version (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.version < 1:
        parser.error("--version must be a positive integer")
    path = download_dataset(args.version)
    print(f"Path to dataset files: {path}")
    for file in sorted(path.rglob("training.*")):
        if file.is_file():
            print(f"  {file.relative_to(path)}: {file.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
