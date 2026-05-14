#!/usr/bin/env python3
"""Shared split-assignment and run metadata utilities.

Public API
----------
build_run_table()             Run + sample_label + study_name + cancer_type + split
build_run_task_table(task)    extends the above with a task_label column
require_binary_classes()      validates a label array has exactly two classes
binary_roc_auc_from_scores()  ROC AUC from positive-class probability scores

Constants: RUN_PATTERN, TETRAMERS, CANCER_LABELS, SPLITS, TRAIN, VAL, TEST, HOLDOUT

Consuming scripts merge their own feature data (tetramer CSV, CAP CSV, torch tensors)
onto the table returned by build_run_task_table on Run, then subset by split:

    df = build_run_task_table("cancer_diagnosis")
    train_df = df[df["split"] == "train"]
"""

from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SplitName = Literal["train", "val", "test", "holdout"]
TRAIN: SplitName = "train"
VAL: SplitName = "val"
TEST: SplitName = "test"
HOLDOUT: SplitName = "holdout"
SPLITS: Tuple[SplitName, ...] = (TRAIN, VAL, TEST, HOLDOUT)

TETRAMERS: Tuple[str, ...] = tuple("".join(p) for p in itertools.product("ACGT", repeat=4))
CANCER_LABELS: Tuple[str, str] = ("breast_cancer", "colorectal_cancer")

# Matches count_tetramers.py / calculate_tetramer_frequencies.py / download_sra_data.py
RUN_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _stratified_split(
    items: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified (train, val, test) arrays using configured fractions."""
    items_tv, items_test, labels_tv, _ = train_test_split(
        items, labels, test_size=test_fraction, stratify=labels, random_state=random_state,
    )
    val_fraction_of_tv = val_fraction / (train_fraction + val_fraction)
    items_train, items_val, _, _ = train_test_split(
        items_tv, labels_tv, test_size=val_fraction_of_tv, stratify=labels_tv,
        random_state=random_state,
    )
    return items_train, items_val, items_test


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_repo_path(raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else _repo_root() / path


def _load_split_config(
    config_path: Optional[Path] = None,
) -> Tuple[Path, int, Path, float, float, float]:
    cfg_path = config_path if config_path is not None else _repo_root() / "defaults.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    try:
        datasets_csv = _resolve_repo_path(cfg["paths"]["datasets_csv"])
        data_dir = _resolve_repo_path(cfg["paths"]["data_dir"])
        shared_splits_cfg = cfg["shared_splits"]
        random_state = int(shared_splits_cfg["random_state"])
        train_fraction = float(shared_splits_cfg["train_fraction"])
        val_fraction = float(shared_splits_cfg["val_fraction"])
        test_fraction = float(shared_splits_cfg["test_fraction"])
    except (TypeError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"Invalid shared split configuration in {cfg_path}: {exc}"
        ) from exc
    for name, value in (
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise SystemExit(
                f"Invalid shared split configuration in {cfg_path}: {name} must be in (0, 1)."
            )
    fraction_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fraction_sum, 1.0, atol=1e-9):
        raise SystemExit(
            "Invalid shared split configuration in "
            f"{cfg_path}: train_fraction + val_fraction + test_fraction must equal 1.0, "
            f"got {fraction_sum:.12f}."
        )
    return datasets_csv, random_state, data_dir, train_fraction, val_fraction, test_fraction


def _row_is_sample_used(row: Mapping[str, object]) -> bool:
    return (row.get("sample_used") or "").strip().casefold() == "true"


def _run_metadata_from_study_csvs(datasets_csv: Path, data_dir: Path) -> pd.DataFrame:
    """Build ordered run metadata from datasets.csv and per-study data/ CSVs."""
    try:
        datasets_order = pd.read_csv(
            datasets_csv, dtype="string", usecols=["study_name", "cancer_type"],
        )
    except ValueError as exc:
        raise SystemExit(
            f"datasets.csv at {datasets_csv} must include columns "
            f"study_name and cancer_type: {exc}"
        ) from exc

    records: List[dict] = []
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
            for row in csv.DictReader(handle):
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
                    {"Run": run, "sample_label": sample_label,
                     "study_name": study_name, "cancer_type": cancer_type}
                )

    if not records:
        raise SystemExit(
            "No eligible runs found in study data CSVs "
            "(need valid Run, sample_used=TRUE, non-empty sample_label)."
        )

    metadata = pd.DataFrame.from_records(records)
    for col in ("Run", "sample_label", "study_name", "cancer_type"):
        metadata[col] = metadata[col].astype(str).str.strip()
    return metadata


def _study_partitions(datasets_csv: Path) -> pd.DataFrame:
    datasets = pd.read_csv(
        datasets_csv, usecols=["study_name", "partition"],
        dtype={"study_name": "string", "partition": "string"},
    )
    datasets = datasets.dropna(subset=["study_name", "partition"]).copy()
    datasets["study_name"] = datasets["study_name"].astype(str).str.strip()
    datasets["partition"] = datasets["partition"].astype(str).str.strip().str.lower()

    bad = sorted(set(datasets["partition"]) - {"development", "holdout"})
    if bad:
        raise SystemExit(
            f"Invalid datasets.csv partition values: {bad}. "
            "Expected 'development' or 'holdout'."
        )
    return datasets.drop_duplicates(subset=["study_name"])


def _apply_task_labels(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Add a task_label column; for cancer_type, also drops non-cancer rows."""
    out = df.copy()
    if task == "cancer_diagnosis":
        out["task_label"] = np.where(
            out["sample_label"].isin(CANCER_LABELS), "cancer", "healthy"
        )
        return out
    if task == "cancer_type":
        out = out[out["sample_label"].isin(CANCER_LABELS)].copy()
        if out.empty:
            raise SystemExit("Task cancer_type selected, but no cancer samples were found.")
        out["task_label"] = out["sample_label"].astype(str)
        return out
    raise SystemExit(f"Unknown task value: {task!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_run_table(*, config_path: Optional[Path] = None) -> pd.DataFrame:
    """Return run metadata with canonical split assignments.

    Columns: Run, sample_label, study_name, cancer_type, split.
    Row order matches datasets.csv study order, then per-study CSV row order.
    Split assignment is deterministic: stratified train/val/test over development
    runs using defaults.yaml shared_splits fractions; holdout studies are assigned
    the 'holdout' split directly.
    """
    (
        datasets_csv,
        random_state,
        data_dir,
        train_fraction,
        val_fraction,
        test_fraction,
    ) = _load_split_config(config_path)
    metadata = _run_metadata_from_study_csvs(datasets_csv, data_dir)
    datasets = _study_partitions(datasets_csv)
    run_table = metadata.merge(datasets, on="study_name", how="left")

    missing = run_table["partition"].isna()
    if missing.any():
        examples = sorted(run_table.loc[missing, "study_name"].unique())[:5]
        raise SystemExit(
            "Some studies in run metadata are missing from datasets.csv. "
            f"Example studies: {examples}"
        )

    split_map: Dict[str, SplitName] = {
        str(run): HOLDOUT
        for run in run_table.loc[run_table["partition"] == "holdout", "Run"]
    }

    development = run_table[run_table["partition"] == "development"]
    if development.empty:
        raise SystemExit("No development runs found for shared split assignment.")

    runs_train, runs_val, runs_test = _stratified_split(
        development["Run"].to_numpy(dtype=object),
        development["sample_label"].to_numpy(dtype=object),
        random_state=random_state,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    for run in runs_train:
        split_map[str(run)] = TRAIN
    for run in runs_val:
        split_map[str(run)] = VAL
    for run in runs_test:
        split_map[str(run)] = TEST

    out = metadata.copy()
    out["split"] = out["Run"].map(split_map)
    return out


def build_run_task_table(task: str, *, config_path: Optional[Path] = None) -> pd.DataFrame:
    """Return run metadata with canonical splits and task-specific labels.

    Columns: Run, sample_label, study_name, cancer_type, split, task_label.
    For task='cancer_type', rows for non-cancer runs are dropped.

    Consuming scripts merge their feature data onto this table by Run, then
    subset by split:
        df = build_run_task_table("cancer_diagnosis")
        train_df = df[df["split"] == "train"]
    """
    return _apply_task_labels(build_run_table(config_path=config_path), task)


def require_binary_classes(y: np.ndarray, *, split_name: str, task: str) -> None:
    """Raise SystemExit unless the label array contains exactly two distinct classes."""
    classes = np.unique(np.asarray(y, dtype=object))
    if classes.size != 2:
        raise SystemExit(
            f"Task {task!r} requires exactly 2 classes in {split_name}; "
            f"got {classes.tolist()}."
        )


def binary_roc_auc_from_scores(y_true_obj: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC AUC from positive-class scores; return NaN if undefined."""
    y_true_obj = np.asarray(y_true_obj, dtype=object)
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    classes = np.unique(y_true_obj)
    if classes.size != 2:
        return float("nan")
    pos_label = classes[1]
    try:
        return float(roc_auc_score(y_true_obj == pos_label, y_score))
    except ValueError:
        return float("nan")
