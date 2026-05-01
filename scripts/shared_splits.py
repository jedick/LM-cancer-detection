#!/usr/bin/env python3
"""Canonical run-level split assignment.

Split assignments are computed from a **deterministic** run metadata table: each
row in ``datasets.csv`` (in file order), then each matching study CSV under
``paths.data_dir`` / ``<cancer_type>`` / ``<study_name>.csv`` (in file order),
keeping only rows with a valid SRA ``Run``, ``sample_used`` = TRUE
(case-insensitive, same rule as ``calculate_tetramer_frequencies.py``), and a
non-empty ``sample_label``. That ordering is reproducible from version-controlled
inputs and does not depend on ``tetramer_frequencies.csv`` (which is built
incrementally and is not deterministic).

The merged table is joined to ``datasets.csv`` partitions for holdout vs
development, then development runs are stratified 70/10/20 train/val/test.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


SplitName = Literal["train", "val", "test", "holdout"]
TRAIN: SplitName = "train"
VAL: SplitName = "val"
TEST: SplitName = "test"
HOLDOUT: SplitName = "holdout"
SPLITS: Tuple[SplitName, ...] = (TRAIN, VAL, TEST, HOLDOUT)

# Matches calculate_tetramer_frequencies.py / download_sra_data.py
RUN_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")


def stratified_split_70_10_20(
    items: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lower-level stratified 70% / 10% / 20% train / val / test helper."""
    items_tv, items_test, labels_tv, labels_test = train_test_split(
        items,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    val_fraction_of_tv = 0.1 / 0.8
    items_train, items_val, labels_train, labels_val = train_test_split(
        items_tv,
        labels_tv,
        test_size=val_fraction_of_tv,
        stratify=labels_tv,
        random_state=random_state,
    )
    return items_train, items_val, items_test, labels_train, labels_val, labels_test


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_config_path() -> Path:
    return _repo_root() / "defaults.yaml"


def _resolve_repo_path(raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else _repo_root() / path


def _load_split_config(config_path: Optional[Path] = None) -> Tuple[Path, int, Path]:
    cfg_path = config_path if config_path is not None else _default_config_path()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    try:
        datasets_csv = _resolve_repo_path(cfg["paths"]["datasets_csv"])
        random_state = int(cfg["shared_splits"]["random_state"])
        data_dir = _resolve_repo_path(cfg["paths"]["data_dir"])
    except (TypeError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"Invalid shared split configuration in {cfg_path}: {exc}"
        ) from exc
    return datasets_csv, random_state, data_dir


def _row_is_sample_used(row: Mapping[str, object]) -> bool:
    """Same rule as ``calculate_tetramer_frequencies.row_is_sample_used``."""
    return (row.get("sample_used") or "").strip().casefold() == "true"


def _run_metadata_from_study_csvs(datasets_csv: Path, data_dir: Path) -> pd.DataFrame:
    """Build ordered Run metadata from ``datasets.csv`` and per-study ``data/`` CSVs."""
    try:
        datasets_order = pd.read_csv(
            datasets_csv,
            dtype="string",
            usecols=["study_name", "cancer_type"],
        )
    except ValueError as exc:
        raise SystemExit(
            f"datasets.csv at {datasets_csv} must include columns "
            f"study_name and cancer_type: {exc}"
        ) from exc

    records: List[dict[str, str]] = []
    run_first: Dict[str, Tuple[str, str]] = {}

    for _, ds_row in datasets_order.iterrows():
        study_name = str(ds_row.get("study_name") or "").strip()
        cancer_type = str(ds_row.get("cancer_type") or "").strip()
        if not study_name or not cancer_type:
            continue

        study_path = data_dir / cancer_type / f"{study_name}.csv"
        if not study_path.is_file():
            raise SystemExit(
                "Shared splits: study data CSV not found for "
                f"study_name={study_name!r} cancer_type={cancer_type!r} "
                f"(expected {study_path})."
            )

        with open(study_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run = (row.get("Run") or "").strip()
                if not run or RUN_PATTERN.match(run) is None:
                    continue
                if not _row_is_sample_used(row):
                    continue
                sample_label = (row.get("sample_label") or "").strip()
                if not sample_label:
                    continue

                seen = run_first.get(run)
                if seen is not None:
                    seen_study, seen_label = seen
                    if seen_study != study_name:
                        raise SystemExit(
                            f"Run {run!r} appears under multiple studies "
                            f"({seen_study!r} and {study_name!r})."
                        )
                    if seen_label != sample_label:
                        raise SystemExit(
                            f"Run {run!r} has conflicting sample_label values "
                            f"({seen_label!r} vs {sample_label!r}) in study {study_name!r}."
                        )
                    continue

                run_first[run] = (study_name, sample_label)
                records.append(
                    {
                        "Run": run,
                        "sample_label": sample_label,
                        "study_name": study_name,
                        "cancer_type": cancer_type,
                    }
                )

    if not records:
        raise SystemExit(
            "No eligible runs found in study data CSVs for shared splits "
            "(need valid Run, sample_used=TRUE, non-empty sample_label)."
        )

    metadata = pd.DataFrame.from_records(records)
    for col in ("Run", "sample_label", "study_name", "cancer_type"):
        metadata[col] = metadata[col].astype(str).str.strip()

    conflicts = metadata.groupby("Run")[["sample_label", "study_name"]].nunique()
    bad_runs = conflicts[(conflicts["sample_label"] > 1) | (conflicts["study_name"] > 1)]
    if not bad_runs.empty:
        raise SystemExit(
            "A run has conflicting sample_label or study_name values. "
            f"Example runs: {bad_runs.index[:5].tolist()}"
        )
    return metadata


def build_run_metadata(*, config_path: Optional[Path] = None) -> pd.DataFrame:
    """Return the deterministic run metadata table used for shared split assignment.

    Row order: ``datasets.csv`` study order, then each study file's row order.
    Columns: ``Run``, ``sample_label``, ``study_name``, ``cancer_type`` (from
    ``datasets.csv`` for the study's row).
    """
    datasets_csv, _, data_dir = _load_split_config(config_path)
    return _run_metadata_from_study_csvs(datasets_csv, data_dir)


def _study_partitions(datasets_csv: Path) -> pd.DataFrame:
    datasets = pd.read_csv(
        datasets_csv,
        usecols=["study_name", "partition"],
        dtype={"study_name": "string", "partition": "string"},
    )
    datasets = datasets.dropna(subset=["study_name", "partition"]).copy()
    datasets["study_name"] = datasets["study_name"].astype(str).str.strip()
    datasets["partition"] = datasets["partition"].astype(str).str.strip().str.lower()

    valid_partitions = {"development", "holdout"}
    bad = sorted(set(datasets["partition"]) - valid_partitions)
    if bad:
        raise SystemExit(
            f"Invalid datasets.csv partition values: {bad}. "
            "Expected 'development' or 'holdout'."
        )
    return datasets.drop_duplicates(subset=["study_name"])


def load_run_split_map(*, config_path: Optional[Path] = None) -> Dict[str, SplitName]:
    """Return the canonical split assignment for every Run in the metadata table."""
    datasets_csv, random_state, data_dir = _load_split_config(config_path)
    metadata = _run_metadata_from_study_csvs(datasets_csv, data_dir)
    datasets = _study_partitions(datasets_csv)
    run_table = metadata.merge(datasets, on="study_name", how="left")

    missing_partition = run_table["partition"].isna()
    if missing_partition.any():
        examples = sorted(run_table.loc[missing_partition, "study_name"].unique())[:5]
        raise SystemExit(
            "Some run metadata study_name values are missing from datasets.csv. "
            f"Example studies: {examples}"
        )

    split_map: Dict[str, SplitName] = {
        str(run): HOLDOUT
        for run in run_table.loc[run_table["partition"] == "holdout", "Run"]
    }

    development = run_table[run_table["partition"] == "development"]
    if development.empty:
        raise SystemExit("No development runs found for shared split assignment.")

    runs = development["Run"].to_numpy(dtype=object)
    labels = development["sample_label"].to_numpy(dtype=object)
    runs_train, runs_val, runs_test, _, _, _ = stratified_split_70_10_20(
        runs,
        labels,
        random_state=random_state,
    )
    for run in runs_train:
        split_map[str(run)] = TRAIN
    for run in runs_val:
        split_map[str(run)] = VAL
    for run in runs_test:
        split_map[str(run)] = TEST
    return split_map


def assign_splits_for_runs(
    runs: Iterable[object],
    *,
    config_path: Optional[Path] = None,
) -> list[SplitName]:
    """Return split names for `runs`, preserving input order."""
    split_map = load_run_split_map(config_path=config_path)
    out: list[SplitName] = []
    missing: list[str] = []
    for run in runs:
        run_id = str(run)
        split = split_map.get(run_id)
        if split is None:
            missing.append(run_id)
        else:
            out.append(split)
    if missing:
        raise SystemExit(
            "Some Runs were not assigned by shared split logic. "
            f"Example Runs: {missing[:5]}"
        )
    return out


def add_split_column(
    df: pd.DataFrame,
    *,
    config_path: Optional[Path] = None,
    run_column: str = "Run",
    split_column: str = "split",
) -> pd.DataFrame:
    """Return a copy of `df` with canonical split assignments attached."""
    split_map = load_run_split_map(config_path=config_path)
    out = df.copy()
    out[run_column] = out[run_column].astype(str)
    out[split_column] = out[run_column].map(split_map)
    if out[split_column].isna().any():
        examples = sorted(out.loc[out[split_column].isna(), run_column].unique())[:5]
        raise SystemExit(
            "Some Runs were not assigned by shared split logic. "
            f"Example Runs: {examples}"
        )
    return out
