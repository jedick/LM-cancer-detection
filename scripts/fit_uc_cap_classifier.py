#!/usr/bin/env python3
"""
Train a classifier on UC/CAP run-level features.

This script expects a CAP CSV produced by run_uc_cap_pipeline.py with:
  - Run identifiers in column "Run"
  - Labels in column "sample_label"
  - Optional recorded split in column "split"
  - Feature columns named "cluster_*"

Train/val/test/holdout assignments are always derived from scripts/shared_splits.py.
If the CSV contains a "split" column, it must exactly match the derived split
for every run or the script exits with an error.

Tasks:
  - cancer_diagnosis: all samples, mapped to cancer vs healthy
  - cancer_type: cancer-only samples, breast_cancer vs colorectal_cancer

Models (default hyperparameters):
  - random_forest
  - logistic_regression
  - svm
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from shared_splits import add_split_column, load_run_split_map
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC


CANCER_LABELS: Tuple[str, str] = ("breast_cancer", "colorectal_cancer")


def _float_for_json(x: float) -> Optional[float]:
    if not math.isfinite(x):
        return None
    return float(x)


def _results_json_out_path(
    repo_root: Path, raw: Optional[str], *, task: str, classifier: str
) -> Optional[Path]:
    if raw is None:
        return None
    if raw == "":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(__file__).stem
        name = f"{stem}_{task}_{classifier}_{ts}.json"
        return repo_root / "results" / "scratch" / name
    return Path(raw).expanduser()


def _load_cap_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("Input CSV has no data rows.")
    required = {"Run", "sample_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    feature_cols = [c for c in df.columns if c.startswith("cluster_")]
    if not feature_cols:
        raise SystemExit("CSV has no cluster feature columns (expected prefix 'cluster_').")

    if df["Run"].isna().any():
        raise SystemExit("Found empty Run values in CSV.")
    if df["sample_label"].isna().any():
        raise SystemExit("Found empty sample_label values in CSV.")
    return df


def _validate_csv_splits(df: pd.DataFrame, expected: Dict[str, str]) -> None:
    if "split" not in df.columns:
        return
    run_split_unique = df.loc[:, ["Run", "split"]].drop_duplicates()
    split_counts = run_split_unique.groupby("Run")["split"].nunique()
    bad = split_counts[split_counts > 1].index.tolist()
    if bad:
        raise SystemExit(
            "A run has multiple split values in CSV. "
            f"Example runs: {bad[:5]}"
        )

    mismatches = []
    for _, row in run_split_unique.iterrows():
        run = str(row["Run"])
        observed = str(row["split"]).strip()
        expected_split = expected.get(run)
        if expected_split is None:
            mismatches.append((run, observed, "missing_in_expected"))
            continue
        if observed != expected_split:
            mismatches.append((run, observed, expected_split))
    if mismatches:
        msg = ", ".join(
            [f"{run}: csv={obs}, expected={exp}" for run, obs, exp in mismatches[:10]]
        )
        raise SystemExit(
            "CSV split column does not match shared_splits.py-derived splits. "
            f"First mismatches: {msg}"
        )


def _prepare_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
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
    raise SystemExit(f"Unknown task: {task}")


def _require_binary(y: np.ndarray, split_name: str, task: str) -> None:
    classes = np.unique(y)
    if classes.size != 2:
        raise SystemExit(
            f"Task {task!r} requires 2 classes in {split_name}; got {classes.tolist()}."
        )


def _build_model(name: str, random_state: int):
    if name == "random_forest":
        return RandomForestClassifier(random_state=random_state)
    if name == "logistic_regression":
        return LogisticRegression()
    if name == "svm":
        return SVC()
    raise SystemExit(f"Unknown classifier: {name}")


def _binary_roc_auc(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        return float("nan")
    try:
        return float(roc_auc_score(y_test, y_score))
    except ValueError:
        return float("nan")


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "pipeline.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        default_task = str(cfg["tetramer"]["task"]).strip()
        default_classifier = str(cfg["uc_cap_classifiers"][0]).strip()
        first_grid = cfg["uc_cap_pipeline_grid"][0]
        n_uc = int(first_grid["n_uc"])
        n_clusters = int(first_grid["n_clusters"])
        n_cap = int(first_grid["n_cap"])
        cap_pattern = str(cfg["paths"]["cap_csv_pattern"]).strip()
        default_cap_csv = root / cap_pattern.format(
            n_uc=n_uc, n_clusters=n_clusters, n_cap=n_cap
        )
        default_run_metadata = root / str(cfg["paths"]["tetramer_frequencies_csv"]).strip()
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise SystemExit(f"Invalid pipeline config defaults in {config_path}: {exc}") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_cap_csv,
        help="Input CAP feature CSV (default: first uc_cap_pipeline_grid entry in pipeline config).",
    )
    parser.add_argument(
        "--task",
        choices=("cancer_diagnosis", "cancer_type"),
        default=default_task,
        help="Classification task (default: tetramer.task in pipeline config).",
    )
    parser.add_argument(
        "--classifier",
        choices=("random_forest", "logistic_regression", "svm"),
        default=default_classifier,
        help="Classifier family (default: first uc_cap_classifiers entry in pipeline config).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state used by classifiers that need one.",
    )
    parser.add_argument(
        "--run-metadata-csv",
        type=Path,
        default=default_run_metadata,
        help="Metadata CSV used to derive shared splits (must include Run,sample_label,study_name).",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="?",
        const="",
        default=None,
        help=(
            "Write run metadata and metrics (script, timestamp, task, model, config, "
            "test ROC AUC, validation score) to a JSON file after normal output. "
            "With no path, writes under <repo>/results/scratch/ with an auto-generated name. "
            "Omit the option entirely to skip writing."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    results_json_path = _results_json_out_path(
        root, args.results_json, task=args.task, classifier=args.classifier
    )

    df = _load_cap_csv(args.csv)
    expected_split_map = load_run_split_map(
        args.run_metadata_csv,
    )
    _validate_csv_splits(df, expected_split_map)

    df = add_split_column(
        df,
        run_metadata_csv=args.run_metadata_csv,
        split_column="split_from_shared",
    )

    task_df = _prepare_task(df, args.task)
    feature_cols = [c for c in task_df.columns if c.startswith("cluster_")]

    train_df = task_df[task_df["split_from_shared"] == "train"]
    val_df = task_df[task_df["split_from_shared"] == "val"]
    test_df = task_df[task_df["split_from_shared"] == "test"]
    if train_df.empty or val_df.empty or test_df.empty:
        raise SystemExit("One or more splits are empty after task filtering.")

    X_train = train_df.loc[:, feature_cols].to_numpy(dtype=np.float64, copy=False)
    y_train = train_df["task_label"].to_numpy(dtype=object)
    X_val = val_df.loc[:, feature_cols].to_numpy(dtype=np.float64, copy=False)
    y_val = val_df["task_label"].to_numpy(dtype=object)
    X_test = test_df.loc[:, feature_cols].to_numpy(dtype=np.float64, copy=False)
    y_test = test_df["task_label"].to_numpy(dtype=object)

    _require_binary(y_train, "train split", args.task)
    _require_binary(y_val, "validation split", args.task)
    _require_binary(y_test, "test split", args.task)
    y_all = np.concatenate((y_train, y_val, y_test))

    label_order = np.unique(y_train)
    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    y_train_i = np.asarray([label_to_int[v] for v in y_train], dtype=np.int64)
    y_val_i = np.asarray([label_to_int[v] for v in y_val], dtype=np.int64)
    y_test_i = np.asarray([label_to_int[v] for v in y_test], dtype=np.int64)

    clf = _build_model(args.classifier, random_state=args.random_state)
    clf.fit(X_train, y_train_i)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    val_acc = float(accuracy_score(y_val_i, val_pred))
    test_acc = float(accuracy_score(y_test_i, test_pred))
    test_auc = _binary_roc_auc(clf, X_test, y_test_i)
    auc_str = f"{test_auc:.6f}" if np.isfinite(test_auc) else "nan"

    print(
        f"Split sizes - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}",
        flush=True,
    )
    class_labels, class_counts = np.unique(y_all, return_counts=True)
    class_counts_dict = {
        str(label): int(count) for label, count in zip(class_labels.tolist(), class_counts.tolist())
    }
    print(f"Class counts: {class_counts_dict}", flush=True)
    print(f"Validation accuracy: {val_acc:.6f}", flush=True)
    print(f"Test accuracy: {test_acc:.6f}", flush=True)
    print(f"Test ROC AUC: {auc_str}", flush=True)

    if results_json_path is not None:
        payload = {
            "script": Path(__file__).name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "model": args.classifier,
            "results_json": str(results_json_path.resolve()),
            "config": {
                "csv": str(args.csv.resolve()),
                "random_state": args.random_state,
                "run_metadata_csv": str(args.run_metadata_csv.resolve()),
                "n_features": len(feature_cols),
            },
            "metrics": {
                "test_roc_auc": _float_for_json(test_auc),
                "validation_score": val_acc,
                "validation_metric": "accuracy",
            },
        }
        results_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
