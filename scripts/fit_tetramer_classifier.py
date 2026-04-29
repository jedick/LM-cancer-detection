#!/usr/bin/env python3
"""
Train a classifier on tetramer frequency profiles (256 ACGT 4-mers).

Pipeline: optional **CLR** (on by default) → optional ``StandardScaler`` → **PCA** →
one of **KNN**, **Random Forest**, or **Logistic Regression**.
PCA ``n_components`` starts at ``min(256, n_train - 1)`` (“off”: no reduction below the
train-feasible rank), then halves (128, 64, …), keeping only sizes whose **cumulative**
explained variance on the **training** fold is at least ``--pca-min-variance`` (default 0.9).

Stratified 70/10/20 train/val/test; hyperparameters tuned on the validation set only
(no cross-validation). We first split on original labels, then apply one of two binary
tasks within each split: ``cancer_diagnosis`` (all samples; cancer vs healthy) or
``cancer_type`` (cancer-only; breast vs colorectal). We report binary ROC AUC on the
test set. Use ``--baselines`` for majority-class and stratified-random dummy baselines.

The frequency table produced by calculate_tetramer_frequencies.py uses the
column name ``sample_label``. You can override with ``--label-column`` (e.g. if your
file uses ``sample_labels``).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared_splits import stratified_split_70_10_20

# Lexicographic ACGT tetramers (256 columns), matching calculate_tetramer_frequencies.py
TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)

# Halving grid anchored at 256 (capped to train-feasible max components).
_PCA_POW2_GRID: Tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2, 1)
CANCER_LABELS: Tuple[str, str] = ("breast_cancer", "colorectal_cancer")


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio transform for compositional nonnegative features."""

    def __init__(self, pseudocount: float = 1e-6):
        self.pseudocount = float(pseudocount)

    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be positive.")
        if np.any(X < 0):
            raise ValueError("CLR expects nonnegative compositions before pseudocount.")
        log_x = np.log(X + self.pseudocount)
        return log_x - np.mean(log_x, axis=1, keepdims=True)


def _resolve_label_column(fieldnames: Sequence[str], explicit: Optional[str]) -> str:
    names = set(fieldnames)
    if explicit is not None:
        if explicit not in names:
            raise SystemExit(f"Label column {explicit!r} not found in CSV.")
        return explicit
    if "sample_labels" in names:
        return "sample_labels"
    if "sample_label" in names:
        return "sample_label"
    raise SystemExit(
        "No label column found. Expected 'sample_label' or 'sample_labels'; "
        "use --label-column."
    )


def _load_xy(
    csv_path: Path,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row.")
        missing = [c for c in TETRAMERS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                f"CSV is missing {len(missing)} expected tetramer columns "
                f"(first few: {missing[:5]!r})."
            )
        xs: List[List[float]] = []
        ys: List[str] = []
        study_names: List[str] = []
        for row in reader:
            lab = (row.get(label_column) or "").strip()
            if not lab:
                raise SystemExit(
                    f"Empty {label_column!r} in row {len(xs) + 1}; fix labels before training."
                )
            study = (row.get("study_name") or "").strip()
            if not study:
                raise SystemExit(
                    f"Empty 'study_name' in row {len(xs) + 1}; required for partition split."
                )
            try:
                xs.append([float(row[k]) for k in TETRAMERS])
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Non-numeric tetramer value near data row {len(xs) + 1}: {exc}"
                ) from exc
            ys.append(lab)
            study_names.append(study)
    if not xs:
        raise SystemExit("No data rows in CSV.")
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys),
        np.asarray(study_names, dtype=object),
    )


def _load_study_partition_map(datasets_csv: Path) -> Dict[str, str]:
    if not datasets_csv.is_file():
        raise SystemExit(f"Datasets CSV not found: {datasets_csv}")
    with open(datasets_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("Datasets CSV has no header row.")
        required = {"study_name", "partition"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise SystemExit(f"Datasets CSV missing required columns: {missing}")
        out: Dict[str, str] = {}
        for row in reader:
            study = (row.get("study_name") or "").strip()
            part = (row.get("partition") or "").strip().lower()
            if not study:
                continue
            if part not in {"development", "holdout"}:
                raise SystemExit(
                    f"Invalid partition {part!r} for study {study!r}; "
                    "expected 'development' or 'holdout'."
                )
            out[study] = part
    if not out:
        raise SystemExit("Datasets CSV contains no study_name/partition mappings.")
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if len(out) < 1:
        raise SystemExit("Expected at least one comma-separated integer.")
    return out


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if len(out) < 1:
        raise SystemExit("Expected at least one comma-separated numeric value.")
    return out


def _parse_csv_strs(s: str) -> List[str]:
    out = [p.strip() for p in s.split(",") if p.strip()]
    if len(out) < 1:
        raise SystemExit("Expected at least one comma-separated string.")
    return out


def _parse_csv_optional_ints(s: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p == "none":
            out.append(None)
        else:
            out.append(int(p))
    if len(out) < 1:
        raise SystemExit("Expected at least one value for optional int grid.")
    return out


def _parse_csv_class_weights(s: str) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p == "none":
            out.append(None)
        elif p == "balanced":
            out.append("balanced")
        else:
            raise SystemExit("Class-weight grid values must be 'none' or 'balanced'.")
    if len(out) < 1:
        raise SystemExit("Expected at least one class-weight value.")
    return out


def _pre_pca_train_matrix(
    X_train: np.ndarray,
    *,
    use_clr: bool,
    pseudocount: float,
    use_scaler: bool,
) -> np.ndarray:
    """Match pipeline input to PCA for explained-variance grid (train fold only)."""
    z = X_train
    if use_clr:
        z = CLRTransformer(pseudocount=pseudocount).fit_transform(z)
    if use_scaler:
        z = StandardScaler().fit_transform(z)
    return z


def build_pca_n_components_grid(
    X_train: np.ndarray,
    *,
    use_clr: bool,
    pseudocount: float,
    use_scaler: bool,
    min_explained_variance: float,
    pca_random_state: int,
    n_feature_dims: int = 256,
) -> List[int]:
    """
    PCA component counts: ``min(256, n_train-1)`` (off), then powers of two downward,
    capped to train-feasible size; keep only ``k`` with cumulative explained variance
    >= ``min_explained_variance`` on the training fold (after optional CLR [+ scaler]).
    """
    n_samples, n_feat = X_train.shape
    if n_samples < 2:
        raise SystemExit("Training set too small for PCA (need at least 2 samples).")
    max_comp = int(min(n_feature_dims, n_feat, n_samples - 1))
    if max_comp < 1:
        raise SystemExit("Cannot fit PCA: max components < 1.")

    z_train = _pre_pca_train_matrix(
        X_train,
        use_clr=use_clr,
        pseudocount=pseudocount,
        use_scaler=use_scaler,
    )
    pca_full = PCA(
        n_components=max_comp, svd_solver="full", random_state=pca_random_state
    )
    pca_full.fit(z_train)
    evr = pca_full.explained_variance_ratio_
    csum = np.cumsum(evr)

    def cumulative_var(k: int) -> float:
        kk = min(max(k, 1), len(csum))
        return float(csum[kk - 1])

    # Descending unique caps from the 256, 128, … ladder (each capped to max_comp).
    raw_ks: List[int] = []
    seen: set[int] = set()
    for p in _PCA_POW2_GRID:
        k = min(p, max_comp)
        if k not in seen:
            seen.add(k)
            raw_ks.append(k)

    candidates: List[int] = []
    for k in raw_ks:
        if cumulative_var(k) >= min_explained_variance:
            candidates.append(k)

    if not candidates:
        raise SystemExit(
            "No PCA component counts met --pca-min-variance on the training fold."
        )
    return candidates


def make_pipeline(
    *,
    model: str,
    use_clr: bool,
    pseudocount: float,
    use_scaler: bool,
    pca_n_components: Optional[int],
    pca_random_state: int,
) -> Pipeline:
    """Optional CLR → optional StandardScaler → PCA → classifier."""
    steps: List[Tuple[str, object]] = []
    if use_clr:
        steps.append(("clr", CLRTransformer(pseudocount=pseudocount)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if pca_n_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_n_components,
                    svd_solver="full",
                    random_state=pca_random_state,
                ),
            )
        )
    if model == "knn":
        steps.append(("clf", KNeighborsClassifier()))
    elif model == "random_forest":
        steps.append(("clf", RandomForestClassifier(random_state=pca_random_state)))
    elif model == "logistic_regression":
        # Use a higher iteration cap to avoid frequent lbfgs convergence warnings.
        steps.append(
            ("clf", LogisticRegression(random_state=pca_random_state, max_iter=1000))
        )
    else:
        raise SystemExit(f"Unknown --model value: {model!r}")
    return Pipeline(steps)


def _score_val(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if scoring == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    if scoring == "f1_weighted":
        return float(f1_score(y_true, y_pred, average="weighted"))
    raise SystemExit(f"Unknown scoring: {scoring!r}")


def tune_knn_on_val(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components_list: Sequence[int],
    n_neighbors_list: Sequence[int],
    weights_list: Sequence[str],
    scoring: str,
) -> Tuple[Dict[str, object], float]:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components in n_components_list:
        for n_neighbors in n_neighbors_list:
            for weights in weights_list:
                pipe.set_params(
                    pca__n_components=n_components,
                    clf__n_neighbors=n_neighbors,
                    clf__weights=weights,
                )
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                s = _score_val(y_val, y_pred, scoring)
                if s > best_score:
                    best_score = s
                    best_params = {
                        "n_components": n_components,
                        "n_neighbors": n_neighbors,
                        "weights": weights,
                    }
    return best_params, best_score


def tune_random_forest_on_val(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators_list: Sequence[int],
    max_depth_list: Sequence[Optional[int]],
    min_samples_leaf_list: Sequence[int],
    scoring: str,
) -> Tuple[Dict[str, object], float]:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                pipe.set_params(
                    clf__n_estimators=n_estimators,
                    clf__max_depth=max_depth,
                    clf__min_samples_leaf=min_samples_leaf,
                )
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                s = _score_val(y_val, y_pred, scoring)
                if s > best_score:
                    best_score = s
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }
    return best_params, best_score


def tune_logistic_regression_on_val(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components_list: Sequence[int],
    c_list: Sequence[float],
    solver_list: Sequence[str],
    class_weight_list: Sequence[Optional[str]],
    scoring: str,
) -> Tuple[Dict[str, object], float]:
    allowed_solvers = {"lbfgs", "liblinear", "saga"}
    bad_solvers = [s for s in solver_list if s not in allowed_solvers]
    if bad_solvers:
        raise SystemExit(
            f"Unsupported logistic-regression solver(s): {bad_solvers}. "
            "Allowed: lbfgs, liblinear, saga."
        )
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components in n_components_list:
        for c_val in c_list:
            for solver in solver_list:
                for class_weight in class_weight_list:
                    pipe.set_params(
                        pca__n_components=n_components,
                        clf__C=c_val,
                        clf__solver=solver,
                        clf__class_weight=class_weight,
                    )
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_val)
                    s = _score_val(y_val, y_pred, scoring)
                    if s > best_score:
                        best_score = s
                        best_params = {
                            "n_components": n_components,
                            "C": c_val,
                            "solver": solver,
                            "class_weight": class_weight,
                        }
    return best_params, best_score


def test_binary_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Binary ROC AUC from class probabilities; NaN if undefined."""
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    if classes.size != 2 or np.unique(y_true).size < 2:
        return float("nan")
    pos_label = classes[1]
    y_score = y_proba[:, 1]
    try:
        auc = float(roc_auc_score(y_true == pos_label, y_score))
    except ValueError:
        return float("nan")
    return auc


def _format_auc(auc: float, *, digits: int = 6) -> str:
    def fmt(x: float) -> str:
        return f"{x:.{digits}f}" if np.isfinite(x) else "nan"

    return f"ROC AUC = {fmt(auc)}"


def _label_counts(y: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lab in y:
        out[lab] = out.get(lab, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _prepare_task_data(
    X: np.ndarray, y: np.ndarray, task: str
) -> Tuple[np.ndarray, np.ndarray]:
    if task == "cancer_diagnosis":
        mapped = np.where(np.isin(y, CANCER_LABELS), "cancer", "healthy")
        return X, mapped
    if task == "cancer_type":
        mask = np.isin(y, CANCER_LABELS)
        if not np.any(mask):
            raise SystemExit("Task cancer_type selected, but no cancer samples were found.")
        X_task = X[mask]
        y_task = y[mask]
        return X_task, y_task
    raise SystemExit(f"Unknown --task value: {task!r}")


def _require_binary_classes(y: np.ndarray, *, split_name: str, task: str) -> None:
    classes = np.unique(y)
    if classes.size != 2:
        raise SystemExit(
            f"Task {task!r} requires exactly 2 classes in {split_name}; got {classes.tolist()}."
        )


def _float_for_json(x: float) -> Optional[float]:
    """JSON-serializable float; None for NaN/inf."""
    if not math.isfinite(x):
        return None
    return float(x)


def _results_json_out_path(
    repo_root: Path, raw: Optional[str], *, task: str, model: str
) -> Optional[Path]:
    """None = skip; empty str = auto path under results/scratch/."""
    if raw is None:
        return None
    if raw == "":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(__file__).stem
        name = f"{stem}_{task}_{model}_{ts}.json"
        return repo_root / "results" / "scratch" / name
    return Path(raw).expanduser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "outputs" / "tetramer_frequencies.csv",
        help="Path to tetramer frequency CSV (default: repo root outputs/ file).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Target column name (default: sample_labels if present else sample_label).",
    )
    parser.add_argument(
        "--datasets-csv",
        type=Path,
        default=root / "configs" / "datasets.csv",
        help="Study partition table with study_name,partition columns.",
    )
    parser.add_argument(
        "--model",
        choices=("knn", "random_forest", "logistic_regression"),
        default="knn",
        help="Classifier to train (default: knn).",
    )
    parser.add_argument(
        "--task",
        choices=("cancer_diagnosis", "cancer_type"),
        default="cancer_diagnosis",
        help="Binary task: cancer_diagnosis (all samples) or cancer_type (cancer-only).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for stratified splits and PCA (deterministic SVD path).",
    )
    parser.add_argument(
        "--no-scaler",
        action="store_true",
        help="Disable StandardScaler before PCA.",
    )
    parser.add_argument(
        "--no-clr",
        action="store_true",
        help="Disable CLR; use raw frequency features before scaler/PCA (default: CLR on).",
    )
    parser.add_argument(
        "--clr-pseudocount",
        type=float,
        default=1e-6,
        help="Additive constant before log in CLR (default: 1e-6; ignored with --no-clr).",
    )
    parser.add_argument(
        "--pca-min-variance",
        type=float,
        default=0.9,
        help="Minimum cumulative explained variance on the training fold (after preprocessing "
        "before PCA) for a candidate n_components (default: 0.9).",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also report test ROC AUC for majority-class and stratified-random baselines.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=str,
        default="5,15",
        help="Comma-separated n_neighbors values for KNN tuning (default: 5,15).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="uniform,distance",
        help="Comma-separated weights values for KNN tuning (default: uniform,distance).",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=str,
        default="200,500",
        help="Comma-separated n_estimators values for random-forest tuning.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=str,
        default="none,10",
        help="Comma-separated max_depth values for random-forest tuning (use 'none').",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=str,
        default="1,2",
        help="Comma-separated min_samples_leaf values for random-forest tuning.",
    )
    parser.add_argument(
        "--lr-c",
        type=str,
        default="0.1,1.0,10.0",
        help="Comma-separated C values for logistic-regression tuning.",
    )
    parser.add_argument(
        "--lr-solver",
        type=str,
        default="lbfgs,liblinear",
        help="Comma-separated solver values for logistic-regression tuning.",
    )
    parser.add_argument(
        "--lr-class-weight",
        type=str,
        default="none,balanced",
        help="Comma-separated class_weight values for logistic-regression tuning.",
    )
    parser.add_argument(
        "--scoring",
        choices=("accuracy", "f1_weighted", "f1_macro"),
        default="f1_weighted",
        help="Metric to maximize on the validation set when picking hyperparameters.",
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
        root, args.results_json, task=args.task, model=args.model
    )
    if args.clr_pseudocount <= 0:
        raise SystemExit("--clr-pseudocount must be positive.")
    if not 0.0 < args.pca_min_variance <= 1.0:
        raise SystemExit("--pca-min-variance must be in (0, 1].")

    csv_path: Path = args.csv
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row.")
        label_column = _resolve_label_column(reader.fieldnames, args.label_column)

    X_raw, y_raw, study_raw = _load_xy(csv_path, label_column)
    study_partition_map = _load_study_partition_map(args.datasets_csv)
    partitions: List[str] = []
    missing_studies: set[str] = set()
    for study in study_raw:
        part = study_partition_map.get(str(study))
        if part is None:
            missing_studies.add(str(study))
            partitions.append("missing")
        else:
            partitions.append(part)
    if missing_studies:
        sample = sorted(missing_studies)[:5]
        raise SystemExit(
            "Some study_name values in the tetramer CSV are missing from datasets.csv. "
            f"Example studies: {sample}"
        )
    partition_arr = np.asarray(partitions, dtype=object)
    dev_mask = partition_arr == "development"
    holdout_mask = partition_arr == "holdout"
    if not np.any(dev_mask):
        raise SystemExit("No development rows found in tetramer CSV for training.")
    if not np.any(holdout_mask):
        raise SystemExit("No holdout rows found in tetramer CSV for evaluation.")

    n_neighbors_list = _parse_csv_ints(args.n_neighbors)
    weights_list = _parse_csv_strs(args.weights)
    rf_n_estimators_list = _parse_csv_ints(args.rf_n_estimators)
    rf_max_depth_list = _parse_csv_optional_ints(args.rf_max_depth)
    rf_min_samples_leaf_list = _parse_csv_ints(args.rf_min_samples_leaf)
    lr_c_list = _parse_csv_floats(args.lr_c)
    lr_solver_list = _parse_csv_strs(args.lr_solver)
    lr_class_weight_list = _parse_csv_class_weights(args.lr_class_weight)

    # Split once on development labels so both tasks share a consistent base partition.
    X_dev_raw = X_raw[dev_mask]
    y_dev_raw = y_raw[dev_mask]
    X_holdout_raw = X_raw[holdout_mask]
    y_holdout_raw = y_raw[holdout_mask]
    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = (
        stratified_split_70_10_20(X_dev_raw, y_dev_raw, random_state=args.random_state)
    )
    X_train, y_train = _prepare_task_data(X_train_raw, y_train_raw, args.task)
    X_val, y_val = _prepare_task_data(X_val_raw, y_val_raw, args.task)
    X_test, y_test = _prepare_task_data(X_test_raw, y_test_raw, args.task)
    X_holdout, y_holdout = _prepare_task_data(X_holdout_raw, y_holdout_raw, args.task)
    _require_binary_classes(y_train, split_name="train split", task=args.task)
    _require_binary_classes(y_val, split_name="validation split", task=args.task)
    _require_binary_classes(y_test, split_name="test split", task=args.task)
    y_dev_all = np.concatenate((y_train, y_val, y_test))

    print(
        f"Samples: development={len(y_dev_all)}, holdout={len(y_holdout)}, "
        f"features={X_train.shape[1]}",
        flush=True,
    )
    print(f"Class counts (development): {_label_counts(y_dev_all)}", flush=True)
    print(f"Class counts (holdout): {_label_counts(y_holdout)}", flush=True)
    print(
        f"Split sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}",
        flush=True,
    )

    use_scaler = not args.no_scaler
    use_clr = not args.no_clr
    n_comp_grid: List[int] = []
    if args.model != "random_forest":
        n_comp_grid = build_pca_n_components_grid(
            X_train,
            use_clr=use_clr,
            pseudocount=args.clr_pseudocount,
            use_scaler=use_scaler,
            min_explained_variance=args.pca_min_variance,
            pca_random_state=args.random_state,
        )
        print(
            f"PCA n_components candidates (min cumulative EV >= {args.pca_min_variance}): "
            f"{n_comp_grid} (CLR: {'on' if use_clr else 'off'})",
            flush=True,
        )
    else:
        print("PCA: off for random_forest", flush=True)

    pipe = make_pipeline(
        model=args.model,
        use_clr=use_clr,
        pseudocount=args.clr_pseudocount,
        use_scaler=use_scaler,
        pca_n_components=(n_comp_grid[0] if n_comp_grid else None),
        pca_random_state=args.random_state,
    )
    if args.model == "knn":
        print(
            "Tuning on validation, grid "
            f"n_components={n_comp_grid}, "
            f"n_neighbors={n_neighbors_list}, weights={weights_list}",
            flush=True,
        )
        best_params, best_val_score = tune_knn_on_val(
            pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            n_components_list=n_comp_grid,
            n_neighbors_list=n_neighbors_list,
            weights_list=weights_list,
            scoring=args.scoring,
        )
        pipe.set_params(
            pca__n_components=best_params["n_components"],
            clf__n_neighbors=best_params["n_neighbors"],
            clf__weights=best_params["weights"],
        )
    elif args.model == "random_forest":
        print(
            "Tuning on validation, grid "
            f"n_estimators={rf_n_estimators_list}, "
            f"max_depth={rf_max_depth_list}, "
            f"min_samples_leaf={rf_min_samples_leaf_list}",
            flush=True,
        )
        best_params, best_val_score = tune_random_forest_on_val(
            pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            n_estimators_list=rf_n_estimators_list,
            max_depth_list=rf_max_depth_list,
            min_samples_leaf_list=rf_min_samples_leaf_list,
            scoring=args.scoring,
        )
        pipe.set_params(
            clf__n_estimators=best_params["n_estimators"],
            clf__max_depth=best_params["max_depth"],
            clf__min_samples_leaf=best_params["min_samples_leaf"],
        )
    else:
        print(
            "Tuning on validation, grid "
            f"n_components={n_comp_grid}, "
            f"C={lr_c_list}, solver={lr_solver_list}, class_weight={lr_class_weight_list}",
            flush=True,
        )
        best_params, best_val_score = tune_logistic_regression_on_val(
            pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            n_components_list=n_comp_grid,
            c_list=lr_c_list,
            solver_list=lr_solver_list,
            class_weight_list=lr_class_weight_list,
            scoring=args.scoring,
        )
        pipe.set_params(
            pca__n_components=best_params["n_components"],
            clf__C=best_params["C"],
            clf__solver=best_params["solver"],
            clf__class_weight=best_params["class_weight"],
        )
    print(f"Best validation {args.scoring}: {best_val_score:.6f}", flush=True)
    print(f"Best hyperparameters: {best_params}", flush=True)
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)
    clf_classes = pipe.named_steps["clf"].classes_
    model_auc = test_binary_roc_auc(y_test, y_proba, clf_classes)
    holdout_auc = float("nan")
    if len(y_holdout) > 0:
        holdout_auc = test_binary_roc_auc(
            y_holdout, pipe.predict_proba(X_holdout), clf_classes
        )
    print("\nEvaluation (binary ROC AUC):", flush=True)
    print(f"  {args.model} test: {_format_auc(model_auc)}", flush=True)
    print(f"  {args.model} holdout: {_format_auc(holdout_auc)}", flush=True)

    if args.baselines:
        print("\nTest set baselines (binary ROC AUC):", flush=True)
        maj = DummyClassifier(strategy="most_frequent")
        maj.fit(X_train, y_train)
        m_auc = test_binary_roc_auc(y_test, maj.predict_proba(X_test), maj.classes_)
        print(f"  Majority class: {_format_auc(m_auc)}", flush=True)

        strat = DummyClassifier(
            strategy="stratified", random_state=args.random_state
        )
        strat.fit(X_train, y_train)
        s_auc = test_binary_roc_auc(y_test, strat.predict_proba(X_test), strat.classes_)
        print(f"  Stratified random: {_format_auc(s_auc)}", flush=True)

    if results_json_path is not None:
        config = {
            "csv": str(csv_path.resolve()),
            "datasets_csv": str(args.datasets_csv.resolve()),
            "label_column": label_column,
            "model": args.model,
            "task": args.task,
            "random_state": args.random_state,
            "no_scaler": args.no_scaler,
            "no_clr": args.no_clr,
            "clr_pseudocount": args.clr_pseudocount,
            "pca_min_variance": args.pca_min_variance,
            "baselines": args.baselines,
            "n_neighbors": args.n_neighbors,
            "weights": args.weights,
            "rf_n_estimators": args.rf_n_estimators,
            "rf_max_depth": args.rf_max_depth,
            "rf_min_samples_leaf": args.rf_min_samples_leaf,
            "lr_c": args.lr_c,
            "lr_solver": args.lr_solver,
            "lr_class_weight": args.lr_class_weight,
            "scoring": args.scoring,
            "pca_n_components_candidates": [int(x) for x in n_comp_grid],
            "best_hyperparameters": best_params,
        }
        payload = {
            "script": Path(__file__).name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "model": args.model,
            "results_json": str(results_json_path.resolve()),
            "config": config,
            "metrics": {
                "test_roc_auc": _float_for_json(model_auc),
                "holdout_roc_auc": _float_for_json(holdout_auc),
                "validation_score": float(best_val_score),
                "validation_metric": args.scoring,
            },
        }
        results_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
