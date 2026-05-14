#!/usr/bin/env python3
"""
Compute tetramer frequency profiles per sequencing run (Run), incrementally.

For each CSV row under data/ with sample_used=TRUE (case-insensitive), reads the matching
xzipped tetramer counts produced by count_tetramers.py
(outputs/tetramer_counts/<cancer_type>/<study_name>/<Run>.csv.xz),
sums counts for the run, converts to percentages (rounded to 3 decimals), and appends
new rows to the path configured as paths.tetramer_frequencies_csv in defaults.yaml
(typically outputs/tetramer_frequencies.csv) for ML training.

That CSV is feature-only by design: one `Run` column plus
the 256 lexicographic ACGT tetramer columns. Labels/splits are resolved from study
CSV metadata via shared utilities downstream.

Progress: prints each study data file and run count, updates a single-line counter
while processing runs, then prints a short per-study summary.
"""

from __future__ import annotations

import csv
import itertools
import lzma
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

RUN_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")

TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)


def row_is_sample_used(row: Mapping[str, object]) -> bool:
    """True when sample_used is TRUE (case-insensitive). Missing or other values are False."""
    return (row.get("sample_used") or "").strip().casefold() == "true"


def percentages_from_counts(counts: Sequence[int]) -> List[float]:
    total = sum(counts)
    if total == 0:
        return [0.0] * 256
    return [round(100.0 * c / total, 3) for c in counts]


def open_lzma_text(path: Path, mode: str):
    """
    Open .xz in text mode using lzma-mt with all CPU cores when available.

    Falls back to stdlib lzma if lzma-mt is not installed or does not support
    the threads argument in this environment.
    """
    threads = max(1, os.cpu_count() or 1)
    try:
        import lzma_mt  # type: ignore[import-not-found]

        return lzma_mt.open(path, mode, newline="", threads=threads)
    except (ImportError, TypeError):
        return lzma.open(path, mode, newline="")


def load_existing_runs(output_path: Path) -> set[str]:
    """Read existing Run IDs from output CSV; empty set when file is missing."""
    if not output_path.is_file():
        return set()
    runs: set[str] = set()
    with open(output_path, newline="") as in_f:
        reader = csv.DictReader(in_f)
        for row in reader:
            run = (row.get("Run") or "").strip()
            if run:
                runs.add(run)
    return runs


def counts_from_sequence_rows(path: Path) -> Tuple[Optional[List[int]], Optional[str]]:
    """Sum sequence-level 4-mer rows from an existing .csv.xz into run totals."""
    totals = [0] * 256
    try:
        with open_lzma_text(path, "rt") as in_f:
            reader = csv.reader(in_f)
            for row in reader:
                if not row:
                    continue
                if len(row) != 256:
                    return None, f"expected 256 columns, got {len(row)}"
                for i, value in enumerate(row):
                    totals[i] += int(value)
    except (OSError, ValueError) as exc:
        return None, str(exc)
    return totals, None


def _default_paths_from_defaults_yaml(repo_root: Path) -> Tuple[Path, Path, Path]:
    """``(data_dir, tetramer_counts_dir, tetramer_frequencies_csv)`` from ``defaults.yaml``."""
    cfg_path = repo_root / "defaults.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = cfg["paths"]

    def _resolve(key: str) -> Path:
        p = Path(str(paths[key]).strip())
        return p if p.is_absolute() else repo_root / p

    return (
        _resolve("data_dir"),
        _resolve("tetramer_counts_dir"),
        _resolve("tetramer_frequencies_csv"),
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir, counts_root, output_path = _default_paths_from_defaults_yaml(repo_root)

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    data_files = sorted(data_dir.rglob("*.csv"))
    if not data_files:
        print(f"Error: no CSV files under {data_dir}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Run", *list(TETRAMERS)]

    rows_written = 0
    rows_already_in_output = 0
    rows_missing_counts = 0
    rows_zero_kmers = 0

    status_width = 100

    def show_run_progress(run_i: int, n_runs: int, run: str, note: str = "") -> None:
        tail = f"  {note}" if note else ""
        line = f"  {run_i}/{n_runs} runs  (current: {run}){tail}"
        sys.stdout.write("\r" + line.ljust(status_width))
        sys.stdout.flush()

    existing_runs = load_existing_runs(output_path)
    output_exists = output_path.is_file()
    output_needs_header = (not output_exists) or output_path.stat().st_size == 0
    with open(output_path, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if output_needs_header:
            writer.writeheader()

        for csv_path in data_files:
            study_name = csv_path.stem
            rel_path = csv_path.relative_to(data_dir)
            data_file = rel_path.as_posix()
            cancer_type = rel_path.parent.name if len(rel_path.parts) >= 2 else ""

            with open(csv_path, newline="") as in_f:
                reader = csv.DictReader(in_f)
                rows = list(reader)

            n_runs = sum(
                1
                for row in rows
                if RUN_PATTERN.match((row.get("Run") or "").strip())
                and row_is_sample_used(row)
            )
            print(f"Study data from {data_file}: {n_runs} runs", flush=True)

            study_written = 0
            study_missing = 0
            study_zero = 0
            run_i = 0

            for row in rows:
                run = (row.get("Run") or "").strip()
                if not RUN_PATTERN.match(run):
                    continue
                if not row_is_sample_used(row):
                    continue

                run_i += 1
                show_run_progress(run_i, n_runs, run)

                if run in existing_runs:
                    rows_already_in_output += 1
                    show_run_progress(run_i, n_runs, run, "skipped: row already exists")
                    continue

                run_counts_path = counts_root / cancer_type / study_name / f"{run}.csv.xz"
                if not run_counts_path.is_file():
                    print(
                        f"Warning: missing tetramer counts for {study_name}/{run} "
                        f"(expected {run_counts_path}); run `make tetramer_counts` first",
                        file=sys.stderr,
                    )
                    rows_missing_counts += 1
                    study_missing += 1
                    show_run_progress(run_i, n_runs, run, "skipped: no count file")
                    continue

                counts, seq_err = counts_from_sequence_rows(run_counts_path)
                if seq_err is not None or counts is None:
                    print(
                        f"Warning: could not read {run_counts_path}: {seq_err}",
                        file=sys.stderr,
                    )
                    rows_missing_counts += 1
                    study_missing += 1
                    show_run_progress(run_i, n_runs, run, "skipped: read error")
                    continue

                if sum(counts) == 0:
                    rows_zero_kmers += 1
                    study_zero += 1
                    print(
                        f"Warning: zero tetramers in count file for {study_name}/{run}",
                        file=sys.stderr,
                    )

                pct = percentages_from_counts(counts)
                out_row: Dict[str, object] = {"Run": run}
                for kmer, val in zip(TETRAMERS, pct):
                    out_row[kmer] = val
                writer.writerow(out_row)
                rows_written += 1
                study_written += 1
                existing_runs.add(run)
                show_run_progress(run_i, n_runs, run, "wrote row")

            sys.stdout.write("\n")
            sys.stdout.flush()
            print(
                f"  Finished: wrote {study_written}, "
                f"skipped (missing/unreadable counts) {study_missing}, "
                f"zero tetramer totals {study_zero}",
                flush=True,
            )

    print(f"Appended {rows_written} new rows to {output_path}")
    print(f"Rows already in output: {rows_already_in_output}")
    if rows_missing_counts:
        print(f"Skipped (missing/unreadable count files): {rows_missing_counts}")
    if rows_zero_kmers:
        print(f"Runs with zero tetramer totals: {rows_zero_kmers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
