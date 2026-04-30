#!/usr/bin/env python3
"""Canonical run-level split assignment.

Split assignments are always computed from the full run metadata table, then
filtered to whatever subset a caller asks about. That keeps a Run in the same
split whether a script is processing the whole dataset or a partial feature
table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

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


def _load_split_config(config_path: Optional[Path] = None) -> Tuple[Path, int]:
    cfg_path = config_path if config_path is not None else _default_config_path()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    try:
        datasets_csv = _resolve_repo_path(cfg["paths"]["datasets_csv"])
        random_state = int(cfg["shared_splits"]["random_state"])
    except (TypeError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"Invalid shared split configuration in {cfg_path}: {exc}"
        ) from exc
    return datasets_csv, random_state


def _canonical_run_metadata(run_metadata_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(run_metadata_csv, dtype="string")
    label_column = "sample_labels" if "sample_labels" in metadata.columns else "sample_label"
    required = {"Run", "study_name", label_column}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise SystemExit(
            f"Run metadata CSV missing required columns for shared splits: {missing}"
        )
    metadata = metadata.loc[:, ["Run", label_column, "study_name"]].rename(
        columns={label_column: "sample_label"}
    )
    metadata = metadata.dropna(subset=["Run", "sample_label", "study_name"]).copy()
    for col in ["Run", "sample_label", "study_name"]:
        metadata[col] = metadata[col].astype(str).str.strip()
    metadata = metadata[metadata["Run"] != ""]

    conflicts = metadata.groupby("Run")[["sample_label", "study_name"]].nunique()
    bad_runs = conflicts[(conflicts["sample_label"] > 1) | (conflicts["study_name"] > 1)]
    if not bad_runs.empty:
        raise SystemExit(
            "A run has conflicting sample_label or study_name values. "
            f"Example runs: {bad_runs.index[:5].tolist()}"
        )
    return metadata.drop_duplicates(subset=["Run"])


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


def load_run_split_map(
    run_metadata_csv: Path,
    *,
    config_path: Optional[Path] = None,
) -> Dict[str, SplitName]:
    """Return the canonical split assignment for every Run in the metadata table."""
    datasets_csv, random_state = _load_split_config(config_path)
    metadata = _canonical_run_metadata(run_metadata_csv)
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
    run_metadata_csv: Path,
    *,
    config_path: Optional[Path] = None,
) -> list[SplitName]:
    """Return split names for `runs`, preserving input order."""
    split_map = load_run_split_map(
        run_metadata_csv,
        config_path=config_path,
    )
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
    run_metadata_csv: Path,
    config_path: Optional[Path] = None,
    run_column: str = "Run",
    split_column: str = "split",
) -> pd.DataFrame:
    """Return a copy of `df` with canonical split assignments attached."""
    split_map = load_run_split_map(
        run_metadata_csv,
        config_path=config_path,
    )
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
