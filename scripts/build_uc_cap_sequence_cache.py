#!/usr/bin/env python3
"""
Build a cached sequence-level tetramer table for UC/CAP exploration.

Reads per-run sequence tetramer count files from outputs/<cancer_type>/<study_name>/.
Each input file must contain 256 integer columns (no header) with one row per sequence.
Compressed .csv.xz files are expected.

Writes one Parquet file containing:
  - study_name
  - Run
  - sequence_index (1-based row index within the source run file)
  - 256 tetramer count columns (AAAA ... TTTT, lexicographic ACGT order)

The script keeps only the first N rows from each run file (provided via --n-max).
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


RUN_FILE_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")
INPUT_SUFFIX = ".csv.xz"
TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)


def iter_run_files(outputs_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (study_name, run_accession, file_path) for run-level sequence count files."""
    for path in sorted(outputs_dir.rglob(f"*{INPUT_SUFFIX}")):
        if path.name.startswith("."):
            continue
        run = path.name[: -len(INPUT_SUFFIX)]
        if not RUN_FILE_PATTERN.match(run):
            continue
        if len(path.parts) < 3:
            continue
        study_name = path.parent.name
        yield study_name, run, path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Root directory containing per-run sequence count files (default: <repo>/outputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: derived from --n-max under <repo>/outputs/uc_cap/).",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        required=True,
        help="Maximum number of sequences to keep per Run.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=("zstd", "snappy", "gzip", "brotli", "lz4", "none"),
        help="Parquet compression codec (default: zstd).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.n_max <= 0:
        print("--n-max must be a positive integer", file=sys.stderr)
        return 1

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    outputs_dir = args.outputs_dir or (repo_root / "outputs")
    output_path = args.output or (
        outputs_dir / "uc_cap" / f"sequence_counts_first_{args.n_max}_all_runs.parquet"
    )
    parquet_compression = None if args.compression == "none" else args.compression

    if not outputs_dir.is_dir():
        print(f"Error: outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1

    run_files = list(iter_run_files(outputs_dir=outputs_dir))
    if not run_files:
        print(
            (
                f"Error: no run files found in {outputs_dir} with suffix "
                f"'{INPUT_SUFFIX}'"
            ),
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    frames: List[pd.DataFrame] = []
    sequence_columns = list(TETRAMERS)
    out_columns = ["study_name", "Run", "sequence_index"] + sequence_columns

    n_runs = len(run_files)
    n_rows_total = 0
    n_skipped_empty = 0
    print(
        f"Building UC/CAP cache from {n_runs} run files; taking first {args.n_max} rows/run.",
        flush=True,
    )

    for i, (study_name, run, path) in enumerate(run_files, start=1):
        status_line = f"  {i}/{n_runs} runs  (current: {study_name}/{run})"
        sys.stdout.write("\r" + status_line.ljust(100))
        sys.stdout.flush()
        try:
            df = pd.read_csv(
                path,
                header=None,
                nrows=args.n_max,
                dtype="int64",
                compression="infer",
            )
        except Exception as exc:
            sys.stdout.write("\n")
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            return 1

        if df.empty:
            n_skipped_empty += 1
            continue
        if df.shape[1] != 256:
            sys.stdout.write("\n")
            print(
                f"Error: expected 256 columns in {path}, found {df.shape[1]}",
                file=sys.stderr,
            )
            return 1

        df.columns = sequence_columns
        df.insert(0, "sequence_index", range(1, len(df) + 1))
        df.insert(0, "Run", run)
        df.insert(0, "study_name", study_name)
        df = df[out_columns]
        frames.append(df)
        n_rows_total += len(df)

    sys.stdout.write("\n")
    if not frames:
        print("Error: no non-empty run files were processed", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_path, index=False, compression=parquet_compression)

    elapsed = time.perf_counter() - start
    print(f"Wrote {n_rows_total} rows to {output_path}")
    if n_skipped_empty:
        print(f"Skipped empty run files: {n_skipped_empty}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
