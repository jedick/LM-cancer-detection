#!/usr/bin/env python3
"""
Train a classifier on run-level tetramer frequency profiles.

The workflow is:
  1. Read the tetramer frequency table produced by calculate_tetramer_frequencies.py.
  2. Ask shared_splits.py for the canonical Run -> split assignment.
  3. Apply one binary task: cancer diagnosis or cancer type.
  4. Tune hyperparameters on the validation split, then report test and holdout ROC AUC.

Feature preprocessing is optional CLR, optional StandardScaler, and PCA for KNN and
logistic regression. PCA candidates are chosen from the training split only so the
validation, test, and holdout rows do not influence dimensionality selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import yaml
from shared_splits import HOLDOUT, TEST, TRAIN, VAL, add_split_column
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----- Constants -----

# Lexicographic ACGT tetramers (256 columns), matching calculate_tetramer_frequencies.py.
TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)

# Halving grid anchored at 256 (capped to train-feasible max components).
_PCA_POW2_GRID: Tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2, 1)
CANCER_LABELS: Tuple[str, str] = ("breast_cancer", "colorectal_cancer")
T = TypeVar("T")


@dataclass(frozen=True)
class TaskSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray

    @property
    def y_development(self) -> np.ndarray:
        return np.concatenate((self.y_train, self.y_val, self.y_test))


@dataclass(frozen=True)
class ModelGrids:
    n_neighbors: List[int]
    weights: List[str]
    rf_n_estimators: List[int]
    rf_max_depth: List[Optional[int]]
    rf_min_samples_leaf: List[int]
    lr_c: List[float]
    lr_solver: List[str]
    lr_class_weight: List[Optional[str]]


@dataclass(frozen=True)
class TuningResult:
    best_params: Dict[str, object]
    validation_score: float


@dataclass(frozen=True)
class EvaluationResult:
    test_auc: float
    holdout_auc: float


# ----- CLI -----


def parse_args(argv: Optional[Sequence[str]], root: Path) -> argparse.Namespace:
    config_path = root / "configs" / "pipeline.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tet_cfg = cfg["tetramer"]
        paths_cfg = cfg["paths"]
        default_csv = root / str(paths_cfg["tetramer_frequencies_csv"]).strip()
        default_model = str(tet_cfg["model"]).strip()
        default_task = str(tet_cfg["task"]).strip()
        default_random_state = int(tet_cfg["random_state"])
        default_scoring = str(tet_cfg["val_scoring"]).strip()
        default_clr_pseudocount = float(tet_cfg["clr_pseudocount"])
        default_pca_min_variance = float(tet_cfg["pca_min_variance"])
        default_n_neighbors_grid = str(tet_cfg["n_neighbors_grid"]).strip()
        default_weights_grid = str(tet_cfg["weights_grid"]).strip()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid pipeline config defaults in {config_path}: {exc}") from exc

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inputs = parser.add_argument_group("input files")
    inputs.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="Path to tetramer frequency CSV (default: paths.tetramer_frequencies_csv in pipeline config).",
    )
    inputs.add_argument(
        "--label-column",
        default=None,
        help="Target column name (default: sample_labels if present else sample_label).",
    )

    task_model = parser.add_argument_group("task and model")
    task_model.add_argument(
        "--model",
        choices=("knn", "random_forest", "logistic_regression"),
        default=default_model,
        help="Classifier to train (default: tetramer.model in pipeline config).",
    )
    task_model.add_argument(
        "--task",
        choices=("cancer_diagnosis", "cancer_type"),
        default=default_task,
        help="Binary task (default: tetramer.task in pipeline config).",
    )
    task_model.add_argument(
        "--random-state",
        type=int,
        default=default_random_state,
        help="Random seed for PCA and model initialization (default: tetramer.random_state).",
    )
    task_model.add_argument(
        "--scoring",
        choices=("accuracy", "f1_weighted", "f1_macro"),
        default=default_scoring,
        help="Metric to maximize on validation (default: tetramer.val_scoring).",
    )

    preprocessing = parser.add_argument_group("preprocessing and PCA")
    preprocessing.add_argument(
        "--no-scaler",
        action="store_true",
        help="Disable StandardScaler before PCA.",
    )
    preprocessing.add_argument(
        "--no-clr",
        action="store_true",
        help="Disable CLR; use raw frequency features before scaler/PCA (default: CLR on).",
    )
    preprocessing.add_argument(
        "--clr-pseudocount",
        type=float,
        default=default_clr_pseudocount,
        help="Additive constant before log in CLR (default: tetramer.clr_pseudocount; ignored with --no-clr).",
    )
    preprocessing.add_argument(
        "--pca-min-variance",
        type=float,
        default=default_pca_min_variance,
        help="Minimum cumulative explained variance on training fold (default: tetramer.pca_min_variance).",
    )

    grids = parser.add_argument_group("model grids")
    grids.add_argument(
        "--n-neighbors",
        type=str,
        default=default_n_neighbors_grid,
        help="Comma-separated n_neighbors values for KNN tuning (default: tetramer.n_neighbors_grid).",
    )
    grids.add_argument(
        "--weights",
        type=str,
        default=default_weights_grid,
        help="Comma-separated weights values for KNN tuning (default: tetramer.weights_grid).",
    )
    grids.add_argument(
        "--rf-n-estimators",
        type=str,
        default="200,500",
        help="Comma-separated n_estimators values for random-forest tuning.",
    )
    grids.add_argument(
        "--rf-max-depth",
        type=str,
        default="none,10",
        help="Comma-separated max_depth values for random-forest tuning (use 'none').",
    )
    grids.add_argument(
        "--rf-min-samples-leaf",
        type=str,
        default="1,2",
        help="Comma-separated min_samples_leaf values for random-forest tuning.",
    )
    grids.add_argument(
        "--lr-c",
        type=str,
        default="0.1,1.0,10.0",
        help="Comma-separated C values for logistic-regression tuning.",
    )
    grids.add_argument(
        "--lr-solver",
        type=str,
        default="lbfgs,liblinear",
        help="Comma-separated solver values for logistic-regression tuning.",
    )
    grids.add_argument(
        "--lr-class-weight",
        type=str,
        default="none,balanced",
        help="Comma-separated class_weight values for logistic-regression tuning.",
    )

    output = parser.add_argument_group("reporting and output")
    output.add_argument(
        "--baselines",
        action="store_true",
        help="Also report test ROC AUC for majority-class and stratified-random baselines.",
    )
    output.add_argument(
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
    return parser.parse_args(list(argv) if argv is not None else None)


def _validate_basic_args(args: argparse.Namespace) -> None:
    if args.clr_pseudocount <= 0:
        raise SystemExit("--clr-pseudocount must be positive.")
    if not 0.0 < args.pca_min_variance <= 1.0:
        raise SystemExit("--pca-min-variance must be in (0, 1].")


# ----- Comma-separated CLI grid parsing -----


def _split_arg_list(raw: str, description: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"Expected at least one value for {description}.")
    return values


def _parse_arg_grid(
    raw: str,
    convert: Callable[[str], T],
    *,
    description: str,
) -> List[T]:
    return [convert(value) for value in _split_arg_list(raw, description)]


def _parse_int_grid(raw: str, description: str) -> List[int]:
    return _parse_arg_grid(raw, int, description=description)


def _parse_float_grid(raw: str, description: str) -> List[float]:
    return _parse_arg_grid(raw, float, description=description)


def _parse_str_grid(raw: str, description: str) -> List[str]:
    return _split_arg_list(raw, description)


def _parse_optional_int_grid(raw: str, description: str) -> List[Optional[int]]:
    def convert(value: str) -> Optional[int]:
        return None if value.lower() == "none" else int(value)

    return _parse_arg_grid(raw, convert, description=description)


def _parse_class_weight_grid(raw: str) -> List[Optional[str]]:
    def convert(value: str) -> Optional[str]:
        lowered = value.lower()
        if lowered == "none":
            return None
        if lowered == "balanced":
            return "balanced"
        raise SystemExit("Class-weight grid values must be 'none' or 'balanced'.")

    return _parse_arg_grid(raw, convert, description="logistic regression class weights")


def _build_model_grids(args: argparse.Namespace) -> ModelGrids:
    return ModelGrids(
        n_neighbors=_parse_int_grid(args.n_neighbors, "KNN n_neighbors"),
        weights=_parse_str_grid(args.weights, "KNN weights"),
        rf_n_estimators=_parse_int_grid(args.rf_n_estimators, "random-forest n_estimators"),
        rf_max_depth=_parse_optional_int_grid(args.rf_max_depth, "random-forest max_depth"),
        rf_min_samples_leaf=_parse_int_grid(
            args.rf_min_samples_leaf,
            "random-forest min_samples_leaf",
        ),
        lr_c=_parse_float_grid(args.lr_c, "logistic regression C"),
        lr_solver=_parse_str_grid(args.lr_solver, "logistic regression solvers"),
        lr_class_weight=_parse_class_weight_grid(args.lr_class_weight),
    )


# ----- Data loading and task splits -----


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


def _load_feature_table(csv_path: Path, label_column_arg: Optional[str]) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("No data rows in CSV.")
    label_column = _resolve_label_column(df.columns, label_column_arg)

    required = {"Run", "study_name", label_column, *TETRAMERS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"CSV is missing {len(missing)} required columns (first few: {missing[:5]!r})."
        )

    out = df.copy()
    for col in ["Run", "study_name", label_column]:
        out[col] = out[col].astype(str).str.strip()
        if (out[col] == "").any():
            raise SystemExit(f"Found empty {col!r} values in CSV.")
    return out, label_column


def _attach_shared_splits(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # shared_splits computes assignments from the complete metadata table, then
    # filters them to the Runs present here. This keeps subset callers stable.
    return add_split_column(
        df,
        run_metadata_csv=args.csv,
        split_column="split",
    )


def _prepare_task_table(df: pd.DataFrame, label_column: str, task: str) -> pd.DataFrame:
    out = df.copy()
    if task == "cancer_diagnosis":
        out["task_label"] = np.where(
            out[label_column].isin(CANCER_LABELS),
            "cancer",
            "healthy",
        )
        return out
    if task == "cancer_type":
        out = out[out[label_column].isin(CANCER_LABELS)].copy()
        if out.empty:
            raise SystemExit("Task cancer_type selected, but no cancer samples were found.")
        out["task_label"] = out[label_column].astype(str)
        return out
    raise SystemExit(f"Unknown --task value: {task!r}")


def _xy_from_frame(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.loc[:, list(TETRAMERS)].to_numpy(dtype=np.float64, copy=False)
    y = df["task_label"].to_numpy(dtype=object)
    return X, y


def _build_task_splits(df: pd.DataFrame, label_column: str, task: str) -> TaskSplits:
    # One split column carries the old partition and split concepts:
    # train/val/test are development studies; holdout is external evaluation.
    task_df = _prepare_task_table(df, label_column, task)
    frames = {
        TRAIN: task_df[task_df["split"] == TRAIN],
        VAL: task_df[task_df["split"] == VAL],
        TEST: task_df[task_df["split"] == TEST],
        HOLDOUT: task_df[task_df["split"] == HOLDOUT],
    }
    for split_name in (TRAIN, VAL, TEST):
        if frames[split_name].empty:
            raise SystemExit(f"No {split_name} rows found after task filtering.")

    X_train, y_train = _xy_from_frame(frames[TRAIN])
    X_val, y_val = _xy_from_frame(frames[VAL])
    X_test, y_test = _xy_from_frame(frames[TEST])
    X_holdout, y_holdout = _xy_from_frame(frames[HOLDOUT])
    return TaskSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
    )


def _require_binary_classes(y: np.ndarray, *, split_name: str, task: str) -> None:
    classes = np.unique(y)
    if classes.size != 2:
        raise SystemExit(
            f"Task {task!r} requires exactly 2 classes in {split_name}; got {classes.tolist()}."
        )


# ----- Feature preprocessing and pipelines -----


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
    PCA component counts: ``min(256, n_train-1)`` (off), then powers of two downward.

    The explained-variance check fits PCA on the training fold only, after the same
    optional CLR and scaler steps used by the final sklearn pipeline.
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
    csum = np.cumsum(pca_full.explained_variance_ratio_)

    def cumulative_var(k: int) -> float:
        kk = min(max(k, 1), len(csum))
        return float(csum[kk - 1])

    raw_ks: List[int] = []
    seen: set[int] = set()
    for p in _PCA_POW2_GRID:
        k = min(p, max_comp)
        if k not in seen:
            seen.add(k)
            raw_ks.append(k)

    candidates = [k for k in raw_ks if cumulative_var(k) >= min_explained_variance]
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
    """Optional CLR -> optional StandardScaler -> optional PCA -> classifier."""
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
        steps.append(
            ("clf", LogisticRegression(random_state=pca_random_state, max_iter=1000))
        )
    else:
        raise SystemExit(f"Unknown --model value: {model!r}")
    return Pipeline(steps)


# ----- Validation tuning -----


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
    splits: TaskSplits,
    n_components_list: Sequence[int],
    n_neighbors_list: Sequence[int],
    weights_list: Sequence[str],
    scoring: str,
) -> TuningResult:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components, n_neighbors, weights in itertools.product(
        n_components_list,
        n_neighbors_list,
        weights_list,
    ):
        pipe.set_params(
            pca__n_components=n_components,
            clf__n_neighbors=n_neighbors,
            clf__weights=weights,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _score_val(splits.y_val, pipe.predict(splits.X_val), scoring)
        if score > best_score:
            best_score = score
            best_params = {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "weights": weights,
            }
    return TuningResult(best_params=best_params, validation_score=best_score)


def tune_random_forest_on_val(
    pipe: Pipeline,
    splits: TaskSplits,
    n_estimators_list: Sequence[int],
    max_depth_list: Sequence[Optional[int]],
    min_samples_leaf_list: Sequence[int],
    scoring: str,
) -> TuningResult:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_estimators, max_depth, min_samples_leaf in itertools.product(
        n_estimators_list,
        max_depth_list,
        min_samples_leaf_list,
    ):
        pipe.set_params(
            clf__n_estimators=n_estimators,
            clf__max_depth=max_depth,
            clf__min_samples_leaf=min_samples_leaf,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _score_val(splits.y_val, pipe.predict(splits.X_val), scoring)
        if score > best_score:
            best_score = score
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
    return TuningResult(best_params=best_params, validation_score=best_score)


def tune_logistic_regression_on_val(
    pipe: Pipeline,
    splits: TaskSplits,
    n_components_list: Sequence[int],
    c_list: Sequence[float],
    solver_list: Sequence[str],
    class_weight_list: Sequence[Optional[str]],
    scoring: str,
) -> TuningResult:
    allowed_solvers = {"lbfgs", "liblinear", "saga"}
    bad_solvers = [s for s in solver_list if s not in allowed_solvers]
    if bad_solvers:
        raise SystemExit(
            f"Unsupported logistic-regression solver(s): {bad_solvers}. "
            "Allowed: lbfgs, liblinear, saga."
        )

    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components, c_val, solver, class_weight in itertools.product(
        n_components_list,
        c_list,
        solver_list,
        class_weight_list,
    ):
        pipe.set_params(
            pca__n_components=n_components,
            clf__C=c_val,
            clf__solver=solver,
            clf__class_weight=class_weight,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _score_val(splits.y_val, pipe.predict(splits.X_val), scoring)
        if score > best_score:
            best_score = score
            best_params = {
                "n_components": n_components,
                "C": c_val,
                "solver": solver,
                "class_weight": class_weight,
            }
    return TuningResult(best_params=best_params, validation_score=best_score)


def _tune_model_on_validation(
    pipe: Pipeline,
    *,
    args: argparse.Namespace,
    splits: TaskSplits,
    grids: ModelGrids,
    n_components_grid: Sequence[int],
) -> TuningResult:
    if args.model == "knn":
        print(
            "Tuning on validation, grid "
            f"n_components={list(n_components_grid)}, "
            f"n_neighbors={grids.n_neighbors}, weights={grids.weights}",
            flush=True,
        )
        result = tune_knn_on_val(
            pipe,
            splits,
            n_components_list=n_components_grid,
            n_neighbors_list=grids.n_neighbors,
            weights_list=grids.weights,
            scoring=args.scoring,
        )
        pipe.set_params(
            pca__n_components=result.best_params["n_components"],
            clf__n_neighbors=result.best_params["n_neighbors"],
            clf__weights=result.best_params["weights"],
        )
        return result

    if args.model == "random_forest":
        print(
            "Tuning on validation, grid "
            f"n_estimators={grids.rf_n_estimators}, "
            f"max_depth={grids.rf_max_depth}, "
            f"min_samples_leaf={grids.rf_min_samples_leaf}",
            flush=True,
        )
        result = tune_random_forest_on_val(
            pipe,
            splits,
            n_estimators_list=grids.rf_n_estimators,
            max_depth_list=grids.rf_max_depth,
            min_samples_leaf_list=grids.rf_min_samples_leaf,
            scoring=args.scoring,
        )
        pipe.set_params(
            clf__n_estimators=result.best_params["n_estimators"],
            clf__max_depth=result.best_params["max_depth"],
            clf__min_samples_leaf=result.best_params["min_samples_leaf"],
        )
        return result

    print(
        "Tuning on validation, grid "
        f"n_components={list(n_components_grid)}, "
        f"C={grids.lr_c}, solver={grids.lr_solver}, class_weight={grids.lr_class_weight}",
        flush=True,
    )
    result = tune_logistic_regression_on_val(
        pipe,
        splits,
        n_components_list=n_components_grid,
        c_list=grids.lr_c,
        solver_list=grids.lr_solver,
        class_weight_list=grids.lr_class_weight,
        scoring=args.scoring,
    )
    pipe.set_params(
        pca__n_components=result.best_params["n_components"],
        clf__C=result.best_params["C"],
        clf__solver=result.best_params["solver"],
        clf__class_weight=result.best_params["class_weight"],
    )
    return result


# ----- Evaluation and reporting -----


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
    value = f"{auc:.{digits}f}" if np.isfinite(auc) else "nan"
    return f"ROC AUC = {value}"


def _label_counts(y: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lab in y:
        out[lab] = out.get(lab, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _print_dataset_summary(splits: TaskSplits) -> None:
    print(
        f"Samples: development={len(splits.y_development)}, "
        f"holdout={len(splits.y_holdout)}, features={splits.X_train.shape[1]}",
        flush=True,
    )
    print(f"Class counts (development): {_label_counts(splits.y_development)}", flush=True)
    print(f"Class counts (holdout): {_label_counts(splits.y_holdout)}", flush=True)
    print(
        f"Split sizes - train: {len(splits.y_train)}, "
        f"val: {len(splits.y_val)}, test: {len(splits.y_test)}",
        flush=True,
    )


def _evaluate_model(pipe: Pipeline, splits: TaskSplits) -> EvaluationResult:
    pipe.fit(splits.X_train, splits.y_train)
    clf_classes = pipe.named_steps["clf"].classes_
    test_auc = test_binary_roc_auc(
        splits.y_test,
        pipe.predict_proba(splits.X_test),
        clf_classes,
    )
    holdout_auc = float("nan")
    if len(splits.y_holdout) > 0:
        holdout_auc = test_binary_roc_auc(
            splits.y_holdout,
            pipe.predict_proba(splits.X_holdout),
            clf_classes,
        )
    return EvaluationResult(test_auc=test_auc, holdout_auc=holdout_auc)


def _print_evaluation(model: str, result: EvaluationResult) -> None:
    print("\nEvaluation (binary ROC AUC):", flush=True)
    print(f"  {model} test: {_format_auc(result.test_auc)}", flush=True)
    print(f"  {model} holdout: {_format_auc(result.holdout_auc)}", flush=True)


def _print_baselines(splits: TaskSplits, random_state: int) -> None:
    print("\nTest set baselines (binary ROC AUC):", flush=True)
    maj = DummyClassifier(strategy="most_frequent")
    maj.fit(splits.X_train, splits.y_train)
    maj_auc = test_binary_roc_auc(
        splits.y_test,
        maj.predict_proba(splits.X_test),
        maj.classes_,
    )
    print(f"  Majority class: {_format_auc(maj_auc)}", flush=True)

    strat = DummyClassifier(strategy="stratified", random_state=random_state)
    strat.fit(splits.X_train, splits.y_train)
    strat_auc = test_binary_roc_auc(
        splits.y_test,
        strat.predict_proba(splits.X_test),
        strat.classes_,
    )
    print(f"  Stratified random: {_format_auc(strat_auc)}", flush=True)


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


def _write_results_json(
    path: Path,
    *,
    args: argparse.Namespace,
    label_column: str,
    n_components_grid: Sequence[int],
    tuning: TuningResult,
    evaluation: EvaluationResult,
) -> None:
    config = {
        "csv": str(args.csv.resolve()),
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
        "pca_n_components_candidates": [int(x) for x in n_components_grid],
        "best_hyperparameters": tuning.best_params,
    }
    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model": args.model,
        "results_json": str(path.resolve()),
        "config": config,
        "metrics": {
            "test_roc_auc": _float_for_json(evaluation.test_auc),
            "holdout_roc_auc": _float_for_json(evaluation.holdout_auc),
            "validation_score": float(tuning.validation_score),
            "validation_metric": args.scoring,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ----- Orchestration -----


def _build_pca_grid(args: argparse.Namespace, splits: TaskSplits) -> List[int]:
    if args.model == "random_forest":
        # The existing analysis treats tree models separately and skips PCA for RF.
        print("PCA: off for random_forest", flush=True)
        return []

    use_scaler = not args.no_scaler
    use_clr = not args.no_clr
    n_components_grid = build_pca_n_components_grid(
        splits.X_train,
        use_clr=use_clr,
        pseudocount=args.clr_pseudocount,
        use_scaler=use_scaler,
        min_explained_variance=args.pca_min_variance,
        pca_random_state=args.random_state,
    )
    print(
        f"PCA n_components candidates (min cumulative EV >= {args.pca_min_variance}): "
        f"{n_components_grid} (CLR: {'on' if use_clr else 'off'})",
        flush=True,
    )
    return n_components_grid


def run_classifier(args: argparse.Namespace, root: Path) -> int:
    _validate_basic_args(args)
    results_json_path = _results_json_out_path(
        root,
        args.results_json,
        task=args.task,
        model=args.model,
    )

    feature_df, label_column = _load_feature_table(args.csv, args.label_column)
    feature_df = _attach_shared_splits(feature_df, args)
    splits = _build_task_splits(feature_df, label_column, args.task)
    _require_binary_classes(splits.y_train, split_name="train split", task=args.task)
    _require_binary_classes(splits.y_val, split_name="validation split", task=args.task)
    _require_binary_classes(splits.y_test, split_name="test split", task=args.task)
    _print_dataset_summary(splits)

    grids = _build_model_grids(args)
    n_components_grid = _build_pca_grid(args, splits)
    pipe = make_pipeline(
        model=args.model,
        use_clr=not args.no_clr,
        pseudocount=args.clr_pseudocount,
        use_scaler=not args.no_scaler,
        pca_n_components=(n_components_grid[0] if n_components_grid else None),
        pca_random_state=args.random_state,
    )
    tuning = _tune_model_on_validation(
        pipe,
        args=args,
        splits=splits,
        grids=grids,
        n_components_grid=n_components_grid,
    )
    print(f"Best validation {args.scoring}: {tuning.validation_score:.6f}", flush=True)
    print(f"Best hyperparameters: {tuning.best_params}", flush=True)

    evaluation = _evaluate_model(pipe, splits)
    _print_evaluation(args.model, evaluation)
    if args.baselines:
        _print_baselines(splits, args.random_state)

    if results_json_path is not None:
        _write_results_json(
            results_json_path,
            args=args,
            label_column=label_column,
            n_components_grid=n_components_grid,
            tuning=tuning,
            evaluation=evaluation,
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parent.parent
    args = parse_args(argv, root)
    return run_classifier(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
