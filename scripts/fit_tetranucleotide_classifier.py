#!/usr/bin/env python3
"""
Train a classifier on tetranucleotide frequency profiles (256 ACGT 4-mers).

Pipeline: optional **CLR** (on by default) → optional ``StandardScaler`` → **PCA** → **KNN**.
PCA ``n_components`` starts at ``min(256, n_train - 1)`` (“off”: no reduction below the
train-feasible rank), then halves (128, 64, …), keeping only sizes whose **cumulative**
explained variance on the **training** fold is at least ``--pca-min-variance`` (default 0.9).

Stratified 70/10/20 train/val/test; hyperparameters tuned on the validation set only
(no cross-validation). We first split on original labels, then apply one of two binary
tasks within each split: ``cancer_diagnosis`` (all samples; cancer vs healthy) or
``cancer_type`` (cancer-only; breast vs colorectal). We report binary ROC AUC on the
test set. Use ``--baselines`` for majority-class and stratified-random dummy baselines.

The frequency table produced by calculate_tetranucleotide_frequencies.py uses the
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared_splits import stratified_split_70_10_20

# Lexicographic ACGT tetramers (256 columns), matching calculate_tetranucleotide_frequencies.py
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
) -> Tuple[np.ndarray, np.ndarray]:
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
        for row in reader:
            lab = (row.get(label_column) or "").strip()
            if not lab:
                raise SystemExit(
                    f"Empty {label_column!r} in row {len(xs) + 1}; fix labels before training."
                )
            try:
                xs.append([float(row[k]) for k in TETRAMERS])
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Non-numeric tetramer value near data row {len(xs) + 1}: {exc}"
                ) from exc
            ys.append(lab)
    if not xs:
        raise SystemExit("No data rows in CSV.")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys)


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if len(out) < 2:
        raise SystemExit("Expected at least two comma-separated integers.")
    return out


def _parse_csv_strs(s: str) -> List[str]:
    out = [p.strip() for p in s.split(",") if p.strip()]
    if len(out) < 2:
        raise SystemExit("Expected at least two comma-separated strings.")
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
    use_clr: bool,
    pseudocount: float,
    use_scaler: bool,
    pca_n_components: int,
    pca_random_state: int,
) -> Pipeline:
    """Optional CLR → optional StandardScaler → PCA → KNeighborsClassifier."""
    steps: List[Tuple[str, object]] = []
    if use_clr:
        steps.append(("clr", CLRTransformer(pseudocount=pseudocount)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
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
    steps.append(("clf", KNeighborsClassifier()))
    return Pipeline(steps)


def tune_knn_pca_on_val(
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
    def score_val(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if scoring == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if scoring == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro"))
        if scoring == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted"))
        raise SystemExit(f"Unknown scoring: {scoring!r}")

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
                s = score_val(y_val, y_pred)
                if s > best_score:
                    best_score = s
                    best_params = {
                        "n_components": n_components,
                        "n_neighbors": n_neighbors,
                        "weights": weights,
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


def _task_description(task: str) -> str:
    if task == "cancer_diagnosis":
        return "All samples: cancer vs healthy."
    if task == "cancer_type":
        return "Cancer-only samples: breast_cancer vs colorectal_cancer."
    raise SystemExit(f"Unknown --task value: {task!r}")


def _float_for_json(x: float) -> Optional[float]:
    """JSON-serializable float; None for NaN/inf."""
    if not math.isfinite(x):
        return None
    return float(x)


def _results_json_out_path(repo_root: Path, raw: Optional[str], *, task: str) -> Optional[Path]:
    """None = skip; empty str = auto path under results/scratch/."""
    if raw is None:
        return None
    if raw == "":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(__file__).stem
        name = f"{stem}_{task}_knn_{ts}.json"
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
        default=root / "outputs" / "tetranucleotide_frequencies.csv",
        help="Path to tetranucleotide frequency CSV (default: repo root outputs/ file).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Target column name (default: sample_labels if present else sample_label).",
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
        help="Comma-separated n_neighbors values for the tuning grid (default: 5,15).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="uniform,distance",
        help="Comma-separated weights values for the tuning grid (default: uniform,distance).",
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
    results_json_path = _results_json_out_path(root, args.results_json, task=args.task)
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

    X_raw, y_raw = _load_xy(csv_path, label_column)
    n_neighbors_list = _parse_csv_ints(args.n_neighbors)
    weights_list = _parse_csv_strs(args.weights)

    # Split once on original labels so both tasks share a consistent base partition.
    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = (
        stratified_split_70_10_20(X_raw, y_raw, random_state=args.random_state)
    )
    X_train, y_train = _prepare_task_data(X_train_raw, y_train_raw, args.task)
    X_val, y_val = _prepare_task_data(X_val_raw, y_val_raw, args.task)
    X_test, y_test = _prepare_task_data(X_test_raw, y_test_raw, args.task)
    task_desc = _task_description(args.task)
    _require_binary_classes(y_train, split_name="train split", task=args.task)
    _require_binary_classes(y_val, split_name="validation split", task=args.task)
    _require_binary_classes(y_test, split_name="test split", task=args.task)
    y_all = np.concatenate((y_train, y_val, y_test))

    print(f"CSV: {csv_path}", flush=True)
    print(f"Task: {args.task} ({task_desc})", flush=True)
    print(f"Samples: {len(y_all)}, features: {X_train.shape[1]}", flush=True)
    print(f"Class counts: {_label_counts(y_all)}", flush=True)
    print(
        f"Split sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}",
        flush=True,
    )
    print(f"CLR: {'off' if args.no_clr else 'on'}", flush=True)

    use_scaler = not args.no_scaler
    use_clr = not args.no_clr
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
        f"{n_comp_grid}",
        flush=True,
    )

    pipe = make_pipeline(
        use_clr=use_clr,
        pseudocount=args.clr_pseudocount,
        use_scaler=use_scaler,
        pca_n_components=n_comp_grid[0],
        pca_random_state=args.random_state,
    )
    print(
        "Tuning on validation, grid "
        f"n_components={n_comp_grid}, "
        f"n_neighbors={n_neighbors_list}, weights={weights_list}",
        flush=True,
    )
    best_params, best_val_score = tune_knn_pca_on_val(
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
    print(f"Best validation {args.scoring}: {best_val_score:.6f}", flush=True)
    print(f"Best hyperparameters: {best_params}", flush=True)

    pipe.set_params(
        pca__n_components=best_params["n_components"],
        clf__n_neighbors=best_params["n_neighbors"],
        clf__weights=best_params["weights"],
    )
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)
    clf_classes = pipe.named_steps["clf"].classes_
    knn_auc = test_binary_roc_auc(y_test, y_proba, clf_classes)
    print("\nTest set (binary ROC AUC):", flush=True)
    print(f"  KNN: {_format_auc(knn_auc)}", flush=True)

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
            "label_column": label_column,
            "task": args.task,
            "random_state": args.random_state,
            "no_scaler": args.no_scaler,
            "no_clr": args.no_clr,
            "clr_pseudocount": args.clr_pseudocount,
            "pca_min_variance": args.pca_min_variance,
            "baselines": args.baselines,
            "n_neighbors": args.n_neighbors,
            "weights": args.weights,
            "scoring": args.scoring,
            "pca_n_components_candidates": [int(x) for x in n_comp_grid],
            "best_hyperparameters": {
                "n_components": int(best_params["n_components"]),
                "n_neighbors": int(best_params["n_neighbors"]),
                "weights": str(best_params["weights"]),
            },
        }
        payload = {
            "script": Path(__file__).name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "model": "knn",
            "results_json": str(results_json_path.resolve()),
            "config": config,
            "metrics": {
                "test_roc_auc": _float_for_json(knn_auc),
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
