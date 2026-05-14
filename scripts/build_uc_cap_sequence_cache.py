#!/usr/bin/env python3
"""
Build a cached sequence-level tetramer table for UC/CAP exploration.

Reads per-run sequence tetramer count files from paths.tetramer_counts_dir
(outputs/tetramer_counts/<cancer_type>/<study_name>/ by default).
Each input file must contain 256 integer columns (no header) with one row per sequence.
Compressed .csv.xz files are expected.

Writes one Parquet file containing:
  - study_name
  - Run
  - sequence_index (1-based row index within the source run file)
  - 256 tetramer count columns (AAAA ... TTTT, lexicographic ACGT order)

Row limit and output path come from defaults.yaml (sequence_cache.n_max_per_run and
paths.uc_cap_root); compression from sequence_cache.parquet_compression.
"""

from __future__ import annotations

import itertools
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import yaml


RUN_FILE_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")
INPUT_SUFFIX = ".csv.xz"
TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)

_PARQUET_COMPRESSIONS = frozenset(
    ("zstd", "snappy", "gzip", "brotli", "lz4", "none")
)


def iter_run_files(counts_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (study_name, run_accession, file_path) for run-level sequence count files."""
    for path in sorted(counts_root.rglob(f"*{INPUT_SUFFIX}")):
        if path.name.startswith("."):
            continue
        run = path.name[: -len(INPUT_SUFFIX)]
        if not RUN_FILE_PATTERN.match(run):
            continue
        if len(path.parts) < 3:
            continue
        study_name = path.parent.name
        yield study_name, run, path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "defaults.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        n_max = int(cfg["sequence_cache"]["n_max_per_run"])
        compression_raw = str(cfg["sequence_cache"]["parquet_compression"]).strip()
        paths = cfg["paths"]

        def _resolve(key: str) -> Path:
            p = Path(str(paths[key]).strip())
            return p if p.is_absolute() else repo_root / p

        counts_dir = _resolve("tetramer_counts_dir")
        uc_cap_root = _resolve("uc_cap_root")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Invalid pipeline config in {cfg_path}: {exc}", file=sys.stderr)
        return 1

    if compression_raw not in _PARQUET_COMPRESSIONS:
        print(
            f"Invalid sequence_cache.parquet_compression in {cfg_path}: "
            f"{compression_raw!r} (expected one of {sorted(_PARQUET_COMPRESSIONS)})",
            file=sys.stderr,
        )
        return 1

    if n_max <= 0:
        print(
            "sequence_cache.n_max_per_run must be a positive integer",
            file=sys.stderr,
        )
        return 1

    output_path = uc_cap_root / f"sequence_counts_first_{n_max}_all_runs.parquet"
    parquet_compression = None if compression_raw == "none" else compression_raw

    if not counts_dir.is_dir():
        print(
            f"Error: tetramer counts directory not found: {counts_dir}",
            file=sys.stderr,
        )
        return 1

    run_files = list(iter_run_files(counts_root=counts_dir))
    if not run_files:
        print(
            (
                f"Error: no run files found in {counts_dir} with suffix "
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
        f"Building UC/CAP cache from {n_runs} run files; taking first {n_max} rows/run.",
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
                nrows=n_max,
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
