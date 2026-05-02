#!/usr/bin/env python3
"""
Compute tetramer frequency profiles per sequencing run (Run), incrementally.

For each CSV row under data/ with sample_used=TRUE (case-insensitive), reads the
matching gzip FASTA produced by download_sra_data.py (fasta/<study_name>/<Run>.fasta.gz),
counts 4-mers in each FASTA sequence (no counting across sequence boundaries), sums
counts for the run, converts to percentages (rounded to 3 decimals), and appends
new rows to outputs/tetramer_frequencies.csv for ML training.

Also writes per-run sequence-level 4-mer counts (256 integers per row, no header) to
outputs/<cancer_type>/<study_name>/<Run>.csv.xz for downstream clustering, but only
when the file does not already exist.

Progress: prints each study data file and run count, updates a single-line counter
while processing runs, then prints a short per-study summary.

Use --first-run-per-study to process only the first eligible row per study for a quick
sanity check before a full pass.

Use --profile to print per-study and total wall time split into gzip/read/parse versus
tetramer counting.

If NumPy and Numba are installed, tetramer counting uses a small @njit kernel by default
(fast on large sequences). Use --no-numba to force the pure-Python counter.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import lzma
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

try:
    import numpy as np
    from numba import njit

    _BASE_LUT_NUMBA = np.full(256, -1, dtype=np.int8)
    _BASE_LUT_NUMBA[ord("A")] = 0
    _BASE_LUT_NUMBA[ord("C")] = 1
    _BASE_LUT_NUMBA[ord("G")] = 2
    _BASE_LUT_NUMBA[ord("T")] = 3

    @njit(cache=True)
    def _count_tetramers_numba(
        buf: np.ndarray, counts: np.ndarray, lut: np.ndarray
    ) -> None:
        """buf: uint8 ASCII (uppercase); counts: int64 length 256; lut: int8 length 256."""
        n = buf.shape[0]
        for i in range(n - 3):
            a = lut[buf[i]]
            if a < 0:
                continue
            b = lut[buf[i + 1]]
            if b < 0:
                continue
            c = lut[buf[i + 2]]
            if c < 0:
                continue
            d = lut[buf[i + 3]]
            if d < 0:
                continue
            idx = (((a << 2) | b) << 2 | c) << 2 | d
            counts[idx] += 1

    _NUMBA_KERNEL_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMBA_KERNEL_AVAILABLE = False
    _BASE_LUT_NUMBA = None  # type: ignore[assignment]
    _count_tetramers_numba = None  # type: ignore[assignment]

# Set True in main() when Numba path is active.
_USE_NUMBA_COUNTING = False

# Matches download_sra_data.py
RUN_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")

# Lexicographic ACGT tetramers (256 columns)
TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)

_BASE_BITS = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class FastaPhaseTimings:
    """Wall seconds for one FASTA.gz pass: everything except inner tetramer loops."""

    io_parse_sec: float = 0.0
    count_sec: float = 0.0

    def total(self) -> float:
        return self.io_parse_sec + self.count_sec

    def __iadd__(self, other: "FastaPhaseTimings") -> "FastaPhaseTimings":
        self.io_parse_sec += other.io_parse_sec
        self.count_sec += other.count_sec
        return self


def row_is_sample_used(row: Mapping[str, object]) -> bool:
    """True when sample_used is TRUE (case-insensitive). Missing or other values are False."""
    return (row.get("sample_used") or "").strip().casefold() == "true"


def count_tetramers_in_sequence(seq: str, counts: List[int]) -> None:
    """Add tetramer counts from one sequence into counts (length 256), pure Python."""
    if len(seq) < 4:
        return
    s = seq.upper()
    n = len(s)
    for i in range(0, n - 3):
        a, b, c, d = s[i], s[i + 1], s[i + 2], s[i + 3]
        bits = _BASE_BITS.get(a)
        if bits is None:
            continue
        bbits = _BASE_BITS.get(b)
        if bbits is None:
            continue
        cbits = _BASE_BITS.get(c)
        if cbits is None:
            continue
        dbits = _BASE_BITS.get(d)
        if dbits is None:
            continue
        idx = (((bits << 2) | bbits) << 2 | cbits) << 2 | dbits
        counts[idx] += 1


def _warmup_numba_kernel() -> None:
    """Compile the Numba kernel once so the first real run is not skewed by JIT time."""
    if not (_NUMBA_KERNEL_AVAILABLE and np is not None and _count_tetramers_numba is not None):
        return
    buf = np.array((65, 67, 71, 84), dtype=np.uint8)  # ACGT
    tmp = np.zeros(256, dtype=np.int64)
    _count_tetramers_numba(buf, tmp, _BASE_LUT_NUMBA)


def configure_counting_backend(use_numba: bool) -> None:
    """Select tetramer counting implementation (Numba JIT when available, else pure Python)."""
    global _USE_NUMBA_COUNTING
    if use_numba and _NUMBA_KERNEL_AVAILABLE:
        _USE_NUMBA_COUNTING = True
        _warmup_numba_kernel()
    else:
        _USE_NUMBA_COUNTING = False


def accumulate_tetramers_from_sequence(
    seq: str, counts_buffer: Union[List[int], "np.ndarray"]
) -> None:
    """Add tetramer counts for one sequence into counts_buffer (256 bins)."""
    if _USE_NUMBA_COUNTING:
        if np is None or _count_tetramers_numba is None or _BASE_LUT_NUMBA is None:
            raise RuntimeError("Numba counting requested but dependencies are missing")
        if len(seq) < 4:
            return
        buf = np.frombuffer(memoryview(seq.upper().encode("ascii")), dtype=np.uint8)
        _count_tetramers_numba(buf, counts_buffer, _BASE_LUT_NUMBA)
    else:
        count_tetramers_in_sequence(seq, counts_buffer)  # type: ignore[arg-type]


def iter_fasta_records(gzip_path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (header_without_>, sequence) records from a .fasta.gz file."""
    with gzip.open(gzip_path, "rt", encoding="ascii", errors="replace") as handle:
        header: Optional[str] = None
        chunks: List[str] = []
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    yield header, "".join(chunks)
                    chunks = []
                header = line[1:].strip()
                continue
            chunks.append(line)
        if header is not None and chunks:
            yield header, "".join(chunks)


def tetramer_counts_for_run_and_sequences(
    fasta_gz: Path,
) -> Tuple[List[int], List[List[int]], Optional[str]]:
    """
    One pass over the FASTA: tetramer counts per sequence and summed run totals.

    Returns (run_counts, per_sequence_counts, error_message).
    run_counts is length 256; per_sequence_counts has one 256-count row per sequence.
    error_message is set on I/O failure.
    """
    if _USE_NUMBA_COUNTING and np is not None:
        run_counts_arr = np.zeros(256, dtype=np.int64)
        seq_counts_arr = np.zeros(256, dtype=np.int64)
        per_sequence_rows: List[List[int]] = []
        try:
            for _, seq in iter_fasta_records(fasta_gz):
                seq_counts_arr.fill(0)
                accumulate_tetramers_from_sequence(seq, seq_counts_arr)
                run_counts_arr += seq_counts_arr
                per_sequence_rows.append(seq_counts_arr.tolist())
        except OSError as exc:
            return run_counts_arr.tolist(), per_sequence_rows, str(exc)
        return run_counts_arr.tolist(), per_sequence_rows, None

    run_counts: List[int] = [0] * 256
    seq_counts: List[int] = [0] * 256
    per_sequence_rows: List[List[int]] = []
    try:
        for _, seq in iter_fasta_records(fasta_gz):
            for i in range(256):
                seq_counts[i] = 0
            accumulate_tetramers_from_sequence(seq, seq_counts)
            for i, c in enumerate(seq_counts):
                run_counts[i] += c
            per_sequence_rows.append(seq_counts.copy())
    except OSError as exc:
        return run_counts, per_sequence_rows, str(exc)
    return run_counts, per_sequence_rows, None


def tetramer_counts_for_run_and_sequences_profiled(
    fasta_gz: Path,
) -> Tuple[List[int], List[List[int]], Optional[str], FastaPhaseTimings]:
    """
    Profile one FASTA pass, splitting wall time into:
    - io_parse_sec: gzip read/decompress + line parsing + sequence joins
    - count_sec: time spent in tetramer counting (Python or Numba path)
    """
    if _USE_NUMBA_COUNTING and np is not None:
        counts_buf: Union[List[int], np.ndarray] = np.zeros(256, dtype=np.int64)
        seq_counts_arr = np.zeros(256, dtype=np.int64)
    else:
        counts_buf = [0] * 256
        seq_counts_py = [0] * 256
    per_sequence_rows: List[List[int]] = []
    timings = FastaPhaseTimings()
    try:
        with gzip.open(fasta_gz, "rt", encoding="ascii", errors="replace") as handle:
            header: Optional[str] = None
            chunks: List[str] = []
            for raw in handle:
                t0 = time.perf_counter()
                line = raw.strip()
                if not line:
                    timings.io_parse_sec += time.perf_counter() - t0
                    continue
                if line.startswith(">"):
                    if header is not None and chunks:
                        seq = "".join(chunks)
                        chunks = []
                        timings.io_parse_sec += time.perf_counter() - t0
                        c0 = time.perf_counter()
                        if _USE_NUMBA_COUNTING and np is not None:
                            seq_counts_arr.fill(0)
                            accumulate_tetramers_from_sequence(seq, seq_counts_arr)
                            counts_buf += seq_counts_arr
                            per_sequence_rows.append(seq_counts_arr.tolist())
                        else:
                            for i in range(256):
                                seq_counts_py[i] = 0
                            accumulate_tetramers_from_sequence(seq, seq_counts_py)
                            for i, c in enumerate(seq_counts_py):
                                counts_buf[i] += c  # type: ignore[index]
                            per_sequence_rows.append(seq_counts_py.copy())
                        timings.count_sec += time.perf_counter() - c0
                    else:
                        timings.io_parse_sec += time.perf_counter() - t0
                    header = line[1:].strip()
                    continue
                chunks.append(line)
                timings.io_parse_sec += time.perf_counter() - t0

            if header is not None and chunks:
                t0 = time.perf_counter()
                seq = "".join(chunks)
                timings.io_parse_sec += time.perf_counter() - t0
                c0 = time.perf_counter()
                if _USE_NUMBA_COUNTING and np is not None:
                    seq_counts_arr.fill(0)
                    accumulate_tetramers_from_sequence(seq, seq_counts_arr)
                    counts_buf += seq_counts_arr
                    per_sequence_rows.append(seq_counts_arr.tolist())
                else:
                    for i in range(256):
                        seq_counts_py[i] = 0
                    accumulate_tetramers_from_sequence(seq, seq_counts_py)
                    for i, c in enumerate(seq_counts_py):
                        counts_buf[i] += c  # type: ignore[index]
                    per_sequence_rows.append(seq_counts_py.copy())
                timings.count_sec += time.perf_counter() - c0
    except OSError as exc:
        if isinstance(counts_buf, list):
            return counts_buf, per_sequence_rows, str(exc), timings
        return counts_buf.tolist(), per_sequence_rows, str(exc), timings

    if isinstance(counts_buf, list):
        return counts_buf, per_sequence_rows, None, timings
    return counts_buf.tolist(), per_sequence_rows, None, timings


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
    """``(data_dir, fasta_dir, tetramer_frequencies_csv)`` from ``defaults.yaml`` ``paths``."""
    cfg_path = repo_root / "defaults.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = cfg["paths"]

    def _resolve(key: str) -> Path:
        p = Path(str(paths[key]).strip())
        return p if p.is_absolute() else repo_root / p

    return _resolve("data_dir"), _resolve("fasta_dir"), _resolve("tetramer_frequencies_csv")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing study CSV files (default: paths.data_dir in defaults.yaml)",
    )
    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=None,
        help="Root directory of downloaded FASTA gzip files (default: paths.fasta_dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: paths.tetramer_frequencies_csv)",
    )
    parser.add_argument(
        "--first-run-per-study",
        action="store_true",
        help=(
            "Process only the first eligible row per study (valid Run and sample_used=TRUE), "
            "then move on. Useful to validate output before a full run."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print timing breakdown for FASTA processing: gzip/read/parse versus "
            "tetramer counting (per study and total)."
        ),
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba JIT counting even if NumPy/Numba are installed (use pure Python).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    def_data, def_fasta, def_out = _default_paths_from_defaults_yaml(repo_root)
    data_dir = args.data_dir or def_data
    fasta_root = args.fasta_dir or def_fasta
    output_path = args.output or def_out

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    data_files = sorted(data_dir.rglob("*.csv"))
    if not data_files:
        print(f"Error: no CSV files under {data_dir}", file=sys.stderr)
        return 1

    configure_counting_backend(use_numba=not args.no_numba)

    first_run_per_study = args.first_run_per_study
    profile_mode = args.profile

    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_fieldnames = ["cancer_type", "study_name", "Run", "sample_label"]
    feature_fieldnames = list(TETRAMERS)
    fieldnames = label_fieldnames + feature_fieldnames

    rows_written = 0
    rows_already_in_output = 0
    run_seq_written = 0
    run_seq_already_exists = 0
    rows_missing_fasta = 0
    rows_zero_kmers = 0
    total_profile = FastaPhaseTimings()

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
            fasta_dir = fasta_root / study_name
            seq_output_dir = output_path.parent / rel_path.parent / study_name
            seq_output_dir.mkdir(parents=True, exist_ok=True)

            with open(csv_path, newline="") as in_f:
                reader = csv.DictReader(in_f)
                rows = list(reader)

            n_runs = sum(
                1
                for row in rows
                if RUN_PATTERN.match((row.get("Run") or "").strip())
                and row_is_sample_used(row)
            )
            progress_total = 1 if first_run_per_study else n_runs
            print(f"Study data from {data_file}: {n_runs} runs", flush=True)

            study_written = 0
            study_missing = 0
            study_zero = 0
            run_i = 0
            study_profile = FastaPhaseTimings()

            for row in rows:
                run = (row.get("Run") or "").strip()
                if not RUN_PATTERN.match(run):
                    continue
                if not row_is_sample_used(row):
                    continue

                run_i += 1
                show_run_progress(run_i, progress_total, run)
                run_needs_output_row = run not in existing_runs
                fasta_gz = fasta_dir / f"{run}.fasta.gz"
                run_seq_output_path = seq_output_dir / f"{run}.csv.xz"
                run_needs_seq_output = not run_seq_output_path.is_file()

                if not run_needs_output_row and not run_needs_seq_output:
                    rows_already_in_output += 1
                    run_seq_already_exists += 1
                    show_run_progress(
                        run_i, progress_total, run, "skipped: row+seq already exist"
                    )
                    if first_run_per_study:
                        break
                    continue

                counts: Optional[List[int]] = None
                sequence_rows: Optional[List[List[int]]] = None
                read_counts_from_existing_seq = False

                if run_needs_output_row and not run_needs_seq_output:
                    counts, seq_err = counts_from_sequence_rows(run_seq_output_path)
                    if seq_err is None and counts is not None:
                        read_counts_from_existing_seq = True
                    else:
                        print(
                            f"Warning: could not read existing sequence counts "
                            f"{run_seq_output_path}: {seq_err}; recomputing from FASTA",
                            file=sys.stderr,
                        )

                if counts is None:
                    if not fasta_gz.is_file():
                        print(
                            f"Warning: missing FASTA for {study_name}/{run} "
                            f"(expected {fasta_gz})",
                            file=sys.stderr,
                        )
                        rows_missing_fasta += 1
                        study_missing += 1
                        show_run_progress(run_i, progress_total, run, "skipped: no FASTA")
                        if first_run_per_study:
                            break
                        continue

                    if profile_mode:
                        counts, sequence_rows, err, run_profile = (
                            tetramer_counts_for_run_and_sequences_profiled(fasta_gz)
                        )
                        study_profile += run_profile
                        total_profile += run_profile
                    else:
                        counts, sequence_rows, err = tetramer_counts_for_run_and_sequences(
                            fasta_gz
                        )
                    if err:
                        print(
                            f"Warning: could not read {fasta_gz}: {err}",
                            file=sys.stderr,
                        )
                        rows_missing_fasta += 1
                        study_missing += 1
                        show_run_progress(run_i, progress_total, run, "skipped: read error")
                        if first_run_per_study:
                            break
                        continue

                if counts is None:
                    # Defensive guard; should not happen, but keeps types explicit.
                    show_run_progress(run_i, progress_total, run, "skipped: no counts")
                    if first_run_per_study:
                        break
                    continue

                if run_needs_seq_output:
                    if sequence_rows is None:
                        # We have counts loaded from existing seq rows; this branch means
                        # sequence output is missing, so recompute from FASTA to write rows.
                        if not fasta_gz.is_file():
                            print(
                                f"Warning: missing FASTA for {study_name}/{run} "
                                f"(expected {fasta_gz})",
                                file=sys.stderr,
                            )
                            rows_missing_fasta += 1
                            study_missing += 1
                            show_run_progress(
                                run_i, progress_total, run, "skipped: no FASTA for seq output"
                            )
                            if first_run_per_study:
                                break
                            continue
                        if profile_mode:
                            counts, sequence_rows, err, run_profile = (
                                tetramer_counts_for_run_and_sequences_profiled(fasta_gz)
                            )
                            study_profile += run_profile
                            total_profile += run_profile
                        else:
                            counts, sequence_rows, err = tetramer_counts_for_run_and_sequences(
                                fasta_gz
                            )
                        if err:
                            print(
                                f"Warning: could not read {fasta_gz}: {err}",
                                file=sys.stderr,
                            )
                            rows_missing_fasta += 1
                            study_missing += 1
                            show_run_progress(
                                run_i, progress_total, run, "skipped: read error"
                            )
                            if first_run_per_study:
                                break
                            continue
                    with open_lzma_text(run_seq_output_path, "wt") as run_out_f:
                        csv.writer(run_out_f).writerows(sequence_rows or [])
                    run_seq_written += 1
                else:
                    run_seq_already_exists += 1

                if sum(counts) == 0:
                    rows_zero_kmers += 1
                    study_zero += 1
                    print(
                        f"Warning: zero tetramers counted for {study_name}/{run}",
                        file=sys.stderr,
                    )

                if run_needs_output_row:
                    pct = percentages_from_counts(counts)
                    out_row: Dict[str, object] = {
                        "cancer_type": cancer_type,
                        "study_name": study_name,
                        "Run": run,
                        "sample_label": (row.get("sample_label") or "").strip(),
                    }
                    for kmer, val in zip(TETRAMERS, pct):
                        out_row[kmer] = val
                    writer.writerow(out_row)
                    rows_written += 1
                    study_written += 1
                    existing_runs.add(run)
                    if run_needs_seq_output:
                        show_run_progress(run_i, progress_total, run, "wrote row+seq")
                    elif read_counts_from_existing_seq:
                        show_run_progress(
                            run_i, progress_total, run, "wrote row (from existing seq)"
                        )
                    else:
                        show_run_progress(run_i, progress_total, run, "wrote row")
                else:
                    rows_already_in_output += 1
                    if run_needs_seq_output:
                        show_run_progress(run_i, progress_total, run, "wrote seq only")
                    else:
                        show_run_progress(
                            run_i, progress_total, run, "skipped: row already exists"
                        )
                if first_run_per_study:
                    break

            sys.stdout.write("\n")
            sys.stdout.flush()
            print(
                f"  Finished: wrote {study_written}, "
                f"skipped (missing/unreadable FASTA) {study_missing}, "
                f"zero tetramer counts {study_zero}",
                flush=True,
            )
            if profile_mode and study_written > 0:
                total_sec = study_profile.total()
                io_pct = (100.0 * study_profile.io_parse_sec / total_sec) if total_sec else 0.0
                count_pct = (100.0 * study_profile.count_sec / total_sec) if total_sec else 0.0
                print(
                    f"    Profile: io+parse {study_profile.io_parse_sec:.3f}s "
                    f"({io_pct:.1f}%), count {study_profile.count_sec:.3f}s "
                    f"({count_pct:.1f}%), total {total_sec:.3f}s",
                    flush=True,
                )

    print(f"Appended {rows_written} new rows to {output_path}")
    print(f"Rows already in output: {rows_already_in_output}")
    print(f"Per-run sequence files written: {run_seq_written}")
    print(f"Per-run sequence files already present: {run_seq_already_exists}")
    if rows_missing_fasta:
        print(f"Skipped (missing/unreadable FASTA): {rows_missing_fasta}")
    if rows_zero_kmers:
        print(f"Runs with zero tetramer counts: {rows_zero_kmers}")
    if profile_mode:
        total_sec = total_profile.total()
        io_pct = (100.0 * total_profile.io_parse_sec / total_sec) if total_sec else 0.0
        count_pct = (100.0 * total_profile.count_sec / total_sec) if total_sec else 0.0
        print(
            f"Profile total: io+parse {total_profile.io_parse_sec:.3f}s "
            f"({io_pct:.1f}%), count {total_profile.count_sec:.3f}s "
            f"({count_pct:.1f}%), total {total_sec:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
