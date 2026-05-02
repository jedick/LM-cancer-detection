#!/usr/bin/env python3
"""
Train a classifier on run-level tetramer frequencies or UC/CAP cluster features.

Modes (mutually exclusive):
  --tetramer   Inputs: paths.tetramer_frequencies_csv (256 ACGT tetramer columns).
  --uc_cap     Inputs: CAP CSV from run_uc_cap_pipeline (cluster_* columns).

Both modes use scripts/shared_splits.py for train/val/test/holdout, support the same
model families and hyperparameter grids from defaults.yaml (section fit_classifier), and
optional experiment overlays from experiments.yaml (fit_classifier.experiments).

UC/CAP mode:
  --feat N   1-based index into experiments.yaml run_uc_cap_pipeline (merged over the
             defaults.yaml baseline). Omit --feat to use the baseline-only merged config.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import yaml
from shared_splits import HOLDOUT, TEST, TRAIN, VAL, add_split_column, load_run_split_map
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ----- Constants -----

TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)
_PCA_POW2_GRID: Tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2, 1)
CANCER_LABELS: Tuple[str, str] = ("breast_cancer", "colorectal_cancer")
T = TypeVar("T")

MODEL_CHOICES = ("baseline", "knn", "random_forest", "logistic_regression", "svm")


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
    svm_c: List[float]
    svm_gamma: List[Union[float, str]]
    svm_kernel: List[str]


@dataclass(frozen=True)
class TuningResult:
    best_params: Dict[str, object]
    validation_score: float


@dataclass(frozen=True)
class EvaluationResult:
    test_auc: float
    holdout_auc: float


# ----- UC/CAP path resolution (aligned with helpers/list_uc_cap_feature_outputs.py) -----


def _merge_uc_cap_row(
    defaults_cfg: Mapping[str, Any],
    experiments_cfg: Mapping[str, Any],
    *,
    feat: Optional[int],
) -> Dict[str, Any]:
    """feat None = defaults run_uc_cap_pipeline list only; feat >= 1 adds experiments row."""
    baseline = defaults_cfg.get("run_uc_cap_pipeline")
    if not isinstance(baseline, list):
        raise SystemExit("defaults.yaml run_uc_cap_pipeline must be a list")
    merged: Dict[str, Any] = {}
    for frag in baseline:
        if not isinstance(frag, dict):
            raise SystemExit("defaults.yaml run_uc_cap_pipeline entries must be mappings")
        merged = {**merged, **frag}
    if feat is None:
        return merged
    if feat < 1:
        raise SystemExit("--feat must be >= 1 when provided.")
    rows = experiments_cfg.get("run_uc_cap_pipeline") or []
    if not isinstance(rows, list):
        raise SystemExit("experiments.yaml run_uc_cap_pipeline must be a list")
    if feat > len(rows):
        raise SystemExit(
            f"--feat {feat} is out of range. experiments.yaml defines {len(rows)} UC/CAP rows."
        )
    row = rows[feat - 1]
    if not isinstance(row, dict):
        raise SystemExit("experiments.yaml run_uc_cap_pipeline entries must be mappings")
    return {**merged, **row}


def _cap_csv_path(repo_root: Path, paths_cfg: Mapping[str, Any], merged: Dict[str, Any]) -> Path:
    uc_root = str(paths_cfg["uc_cap_root"]).strip()
    n_uc = int(merged["n_uc"])
    n_clusters = int(merged["n_clusters"])
    n_cap = merged["n_cap"]
    tag = (
        "all"
        if isinstance(n_cap, str) and str(n_cap).strip().lower() == "all"
        else str(int(n_cap))
    )
    cap_transform = str(merged["cap_transform"]).strip()
    stem = f"cap{tag}" if cap_transform == "none" else f"cap{tag}_{cap_transform}"
    return repo_root / uc_root / f"uc{n_uc}_k{n_clusters}" / f"{stem}.csv"


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
            "A run has multiple split values in CSV. " f"Example runs: {bad[:5]}"
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


# ----- Config loading -----


def _load_experiment_args(
    root: Path,
    *,
    expt: int,
    features: str,
    feat: Optional[int],
    results_json_cli: Optional[str],
) -> SimpleNamespace:
    defaults_path = root / "defaults.yaml"
    experiments_path = root / "experiments.yaml"
    try:
        defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
        experiments_cfg = (
            yaml.safe_load(experiments_path.read_text(encoding="utf-8"))
            if experiments_path.is_file()
            else {}
        )
    except OSError as exc:
        raise SystemExit(f"Failed to read config file: {exc}") from exc

    try:
        defaults = dict(defaults_cfg["fit_classifier"])
        paths_cfg = defaults_cfg["paths"]
        experiments_section = experiments_cfg.get("fit_classifier", {})
        experiments = experiments_section.get("experiments", [])
        results_json_template = experiments_section.get("results_json_template")
    except (TypeError, KeyError) as exc:
        raise SystemExit(f"Invalid defaults/experiment configuration: {exc}") from exc

    if not isinstance(experiments, list):
        raise SystemExit(f"Invalid experiments list in {experiments_path}")

    experiment_name = None
    if expt == 0:
        selected: Dict[str, Any] = {}
    else:
        if not experiments:
            raise SystemExit(f"No experiments found in {experiments_path}")
        if expt > len(experiments):
            raise SystemExit(
                f"--expt {expt} is out of range. experiments.yaml defines {len(experiments)} experiments."
            )
        selected = experiments[expt - 1]
        experiment_name = selected.get("name")

    if not isinstance(selected, dict):
        raise SystemExit("Selected experiment entry must be a mapping.")
    overrides = selected.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise SystemExit(f"Experiment {expt} overrides must be a mapping.")

    config = {**defaults, **overrides}
    if expt != 0 and config.get("results_json") is None and results_json_template is not None:
        if not isinstance(results_json_template, str) or not results_json_template.strip():
            raise SystemExit(
                "experiments.yaml results_json_template must be a non-empty string."
            )
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise SystemExit(
                "Each experiment must define a non-empty 'name' when using results_json_template."
            )
        config["results_json"] = results_json_template.format(
            name=experiment_name.strip(),
            features=features,
        )

    results_json = config.get("results_json")
    if results_json_cli is not None:
        results_json = results_json_cli

    if features == "tetramer":
        csv_path = root / str(paths_cfg["tetramer_frequencies_csv"]).strip()
    else:
        merged_uc = _merge_uc_cap_row(defaults_cfg, experiments_cfg, feat=feat)
        csv_path = _cap_csv_path(root, paths_cfg, merged_uc)

    args_dict: Dict[str, Any] = {
        "features": features,
        "feat_index": feat,
        "csv": csv_path,
        "label_column": config.get("label_column"),
        "model": str(config["model"]).strip(),
        "task": str(config["task"]).strip(),
        "random_state": int(config["random_state"]),
        "scoring": str(config["val_scoring"]).strip(),
        "no_scaler": not bool(config["use_scaler"]),
        "no_clr": not bool(config["use_clr"]),
        "clr_pseudocount": float(config["clr_pseudocount"]),
        "pca_min_variance": float(config["pca_min_variance"]),
        "n_neighbors": str(config["n_neighbors_grid"]).strip(),
        "weights": str(config["weights_grid"]).strip(),
        "rf_n_estimators": str(config["rf_n_estimators_grid"]).strip(),
        "rf_max_depth": str(config["rf_max_depth_grid"]).strip(),
        "rf_min_samples_leaf": str(config["rf_min_samples_leaf_grid"]).strip(),
        "lr_c": str(config["lr_c_grid"]).strip(),
        "lr_solver": str(config["lr_solver_grid"]).strip(),
        "lr_class_weight": str(config["lr_class_weight_grid"]).strip(),
        "svm_c": str(config["svm_c_grid"]).strip(),
        "svm_gamma": str(config["svm_gamma_grid"]).strip(),
        "svm_kernel": str(config["svm_kernel_grid"]).strip(),
        "results_json": results_json,
        "experiment_index": expt,
        "experiment_overrides": dict(overrides),
        "log_prefix": (f"E{expt}" if expt > 0 else ""),
    }
    return SimpleNamespace(**args_dict)


def _print_experiment_line(args: argparse.Namespace) -> None:
    expt = int(getattr(args, "experiment_index", 0))
    model = str(getattr(args, "model", ""))
    task = str(getattr(args, "task", ""))
    prefix = str(getattr(args, "log_prefix", ""))
    features = str(getattr(args, "features", ""))
    feat = getattr(args, "feat_index", None)
    if expt == 0:
        print("Default config", flush=True)
    else:
        print(_prefixed(prefix, f"Config - model: {model}, task: {task}"), flush=True)
    if features == "uc_cap":
        fi = "baseline" if feat is None else str(feat)
        print(_prefixed(prefix, f"UC/CAP feature set: {fi}"), flush=True)


def _validate_basic_args(args: argparse.Namespace) -> None:
    if args.clr_pseudocount <= 0:
        raise SystemExit("--clr-pseudocount must be positive.")
    if not 0.0 < args.pca_min_variance <= 1.0:
        raise SystemExit("--pca-min-variance must be in (0, 1].")
    if args.model not in MODEL_CHOICES:
        raise SystemExit(f"Unknown model {args.model!r}. Expected one of {MODEL_CHOICES}.")


def _prefixed(prefix: str, text: str) -> str:
    return f"{prefix} {text}" if prefix else text


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


def _parse_svm_gamma_grid(raw: str) -> List[Union[float, str]]:
    out: List[Union[float, str]] = []
    for value in _split_arg_list(raw, "SVM gamma"):
        lv = value.lower()
        if lv in ("scale", "auto"):
            out.append(lv)
        else:
            out.append(float(value))
    return out


def _parse_svm_kernel_grid(raw: str) -> List[str]:
    allowed = {"rbf", "linear", "poly", "sigmoid"}
    kernels = _parse_str_grid(raw, "SVM kernel")
    bad = [k for k in kernels if k not in allowed]
    if bad:
        raise SystemExit(f"Unsupported SVM kernel(s): {bad}. Allowed: {sorted(allowed)}.")
    return kernels


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
        svm_c=_parse_float_grid(args.svm_c, "SVM C"),
        svm_gamma=_parse_svm_gamma_grid(args.svm_gamma),
        svm_kernel=_parse_svm_kernel_grid(args.svm_kernel),
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
        "use label_column in defaults.yaml."
    )


def _load_tetramer_table(csv_path: Path, label_column_arg: Optional[str]) -> Tuple[pd.DataFrame, str]:
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


def _attach_shared_splits(df: pd.DataFrame, *, config_path: Path) -> pd.DataFrame:
    return add_split_column(df, config_path=config_path, split_column="split")


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
    raise SystemExit(f"Unknown task value: {task!r}")


def _xy_tetramer(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.loc[:, list(TETRAMERS)].to_numpy(dtype=np.float64, copy=False)
    y = df["task_label"].to_numpy(dtype=object)
    return X, y


def _xy_uc_cap(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float64, copy=False)
    y = df["task_label"].to_numpy(dtype=object)
    return X, y


def _build_task_splits_tetramer(df: pd.DataFrame, label_column: str, task: str) -> TaskSplits:
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
    X_train, y_train = _xy_tetramer(frames[TRAIN])
    X_val, y_val = _xy_tetramer(frames[VAL])
    X_test, y_test = _xy_tetramer(frames[TEST])
    X_holdout, y_holdout = _xy_tetramer(frames[HOLDOUT])
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


def _build_task_splits_uc_cap(
    df: pd.DataFrame, label_column: str, task: str, feature_cols: Sequence[str]
) -> TaskSplits:
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
    X_train, y_train = _xy_uc_cap(frames[TRAIN], feature_cols)
    X_val, y_val = _xy_uc_cap(frames[VAL], feature_cols)
    X_test, y_test = _xy_uc_cap(frames[TEST], feature_cols)
    X_holdout, y_holdout = _xy_uc_cap(frames[HOLDOUT], feature_cols)
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
            "No PCA component counts met pca_min_variance on the training fold."
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
    elif model == "svm":
        steps.append(
            (
                "clf",
                SVC(random_state=pca_random_state),
            )
        )
    elif model == "baseline":
        steps.append(("clf", DummyClassifier(strategy="most_frequent")))
    else:
        raise SystemExit(f"Unknown model value: {model!r}")
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


def tune_svm_on_val(
    pipe: Pipeline,
    splits: TaskSplits,
    n_components_list: Sequence[int],
    c_list: Sequence[float],
    gamma_list: Sequence[Union[float, str]],
    kernel_list: Sequence[str],
    scoring: str,
) -> TuningResult:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components, c_val, gamma, kernel in itertools.product(
        n_components_list,
        c_list,
        gamma_list,
        kernel_list,
    ):
        pipe.set_params(
            pca__n_components=n_components,
            clf__C=c_val,
            clf__gamma=gamma,
            clf__kernel=kernel,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _score_val(splits.y_val, pipe.predict(splits.X_val), scoring)
        if score > best_score:
            best_score = score
            best_params = {
                "n_components": n_components,
                "C": c_val,
                "gamma": gamma,
                "kernel": kernel,
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
    prefix = str(getattr(args, "log_prefix", ""))
    if args.model == "baseline":
        print(
            _prefixed(
                prefix,
                "Grid - baseline: DummyClassifier(strategy=most_frequent) (no hyperparameter search)",
            ),
            flush=True,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _score_val(splits.y_val, pipe.predict(splits.X_val), args.scoring)
        return TuningResult(
            best_params={"strategy": "most_frequent"},
            validation_score=score,
        )

    if args.model == "knn":
        print(
            _prefixed(
                prefix,
                f"Grid - n_components: {list(n_components_grid)}, "
                f"n_neighbors: {grids.n_neighbors}, weights: {grids.weights}",
            ),
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
            _prefixed(
                prefix,
                f"Grid - n_estimators: {grids.rf_n_estimators}, "
                f"max_depth: {grids.rf_max_depth}, "
                f"min_samples_leaf: {grids.rf_min_samples_leaf}",
            ),
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

    if args.model == "svm":
        print(
            _prefixed(
                prefix,
                f"Grid - n_components: {list(n_components_grid)}, "
                f"C: {grids.svm_c}, gamma: {grids.svm_gamma}, kernel: {grids.svm_kernel}",
            ),
            flush=True,
        )
        result = tune_svm_on_val(
            pipe,
            splits,
            n_components_list=n_components_grid,
            c_list=grids.svm_c,
            gamma_list=grids.svm_gamma,
            kernel_list=grids.svm_kernel,
            scoring=args.scoring,
        )
        pipe.set_params(
            pca__n_components=result.best_params["n_components"],
            clf__C=result.best_params["C"],
            clf__gamma=result.best_params["gamma"],
            clf__kernel=result.best_params["kernel"],
        )
        return result

    if args.model == "logistic_regression":
        print(
            _prefixed(
                prefix,
                f"Grid - n_components: {list(n_components_grid)}, "
                f"C: {grids.lr_c}, solver: {grids.lr_solver}, class_weight: {grids.lr_class_weight}",
            ),
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

    raise SystemExit(f"Unhandled model for tuning: {args.model!r}")


# ----- Evaluation and reporting -----


def _binary_roc_auc_from_scores(y_true_obj: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC AUC from scores (higher = positive class); NaN if undefined."""
    y_true_obj = np.asarray(y_true_obj)
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    classes = np.unique(y_true_obj)
    if classes.size != 2 or np.unique(y_true_obj).size < 2:
        return float("nan")
    pos_label = classes[1]
    y_bin = y_true_obj == pos_label
    try:
        return float(roc_auc_score(y_bin, y_score))
    except ValueError:
        return float("nan")


def _evaluation_scores(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba(X)
        classes = clf.classes_
        if classes.size != 2:
            raise SystemExit("Binary classification expected.")
        pos = list(classes).index(classes[1])
        return proba[:, pos]
    if hasattr(pipe, "decision_function"):
        return np.asarray(pipe.decision_function(X), dtype=np.float64).ravel()
    raise SystemExit("Classifier has neither predict_proba nor decision_function.")


def _evaluate_model(pipe: Pipeline, splits: TaskSplits) -> EvaluationResult:
    pipe.fit(splits.X_train, splits.y_train)
    test_scores = _evaluation_scores(pipe, splits.X_test)
    test_auc = _binary_roc_auc_from_scores(splits.y_test, test_scores)
    holdout_auc = float("nan")
    if len(splits.y_holdout) > 0:
        hold_scores = _evaluation_scores(pipe, splits.X_holdout)
        holdout_auc = _binary_roc_auc_from_scores(splits.y_holdout, hold_scores)
    return EvaluationResult(test_auc=test_auc, holdout_auc=holdout_auc)


def _label_counts(y: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lab in y:
        out[str(lab)] = out.get(str(lab), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _print_dataset_summary(splits: TaskSplits, *, prefix: str = "") -> None:
    dev_counts = _label_counts(splits.y_development)
    holdout_counts = _label_counts(splits.y_holdout)
    dev_counts_line = ", ".join(f"{k}: {v}" for k, v in dev_counts.items())
    holdout_counts_line = ", ".join(f"{k}: {v}" for k, v in holdout_counts.items())
    print(
        _prefixed(
            prefix,
            f"Sizes - development: {len(splits.y_development)}, "
            f"holdout: {len(splits.y_holdout)}, features: {splits.X_train.shape[1]}",
        ),
        flush=True,
    )
    print(
        _prefixed(prefix, f"Development - {dev_counts_line}"),
        flush=True,
    )
    print(
        _prefixed(
            prefix,
            f"  Splits - train: {len(splits.y_train)}, "
            f"val: {len(splits.y_val)}, test: {len(splits.y_test)}",
        ),
        flush=True,
    )
    print(
        _prefixed(prefix, f"Holdout - {holdout_counts_line}"),
        flush=True,
    )


def _print_evaluation(model: str, result: EvaluationResult, *, prefix: str = "") -> None:
    del model
    test_value = f"{result.test_auc:.6f}" if np.isfinite(result.test_auc) else "nan"
    holdout_value = f"{result.holdout_auc:.6f}" if np.isfinite(result.holdout_auc) else "nan"
    print(_prefixed(prefix, "Evaluation (binary ROC AUC):"), flush=True)
    print(_prefixed(prefix, f"  test: {test_value}"), flush=True)
    print(_prefixed(prefix, f"  holdout: {holdout_value}"), flush=True)


def _float_for_json(x: float) -> Optional[float]:
    if not math.isfinite(x):
        return None
    return float(x)


def _format_hyperparameters(params: Dict[str, object]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in params.items())


def _results_json_out_path(
    repo_root: Path,
    raw: Optional[str],
    *,
    features: str,
    task: str,
    model: str,
) -> Optional[Path]:
    if raw is None:
        return None
    if raw == "":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{features}_{task}_{model}_{ts}.json"
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
    n_features: int,
) -> None:
    config = {
        "features": getattr(args, "features", ""),
        "feat_index": getattr(args, "feat_index", None),
        "csv": str(Path(args.csv).resolve()),
        "label_column": label_column,
        "model": args.model,
        "task": args.task,
        "random_state": args.random_state,
        "no_scaler": args.no_scaler,
        "no_clr": args.no_clr,
        "clr_pseudocount": args.clr_pseudocount,
        "pca_min_variance": args.pca_min_variance,
        "n_neighbors": args.n_neighbors,
        "weights": args.weights,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "lr_c": args.lr_c,
        "lr_solver": args.lr_solver,
        "lr_class_weight": args.lr_class_weight,
        "svm_c": args.svm_c,
        "svm_gamma": args.svm_gamma,
        "svm_kernel": args.svm_kernel,
        "scoring": args.scoring,
        "pca_n_components_candidates": [int(x) for x in n_components_grid],
        "best_hyperparameters": tuning.best_params,
        "n_features": n_features,
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


def _build_pca_grid(args: argparse.Namespace, splits: TaskSplits) -> List[int]:
    prefix = str(getattr(args, "log_prefix", ""))
    if args.model in ("random_forest", "baseline"):
        print(_prefixed(prefix, "PCA - min_explained_variance: n/a, CLR: n/a"), flush=True)
        return []

    use_scaler = not args.no_scaler
    use_clr = not args.no_clr
    n_feat_cap = min(256, int(splits.X_train.shape[1]))
    n_components_grid = build_pca_n_components_grid(
        splits.X_train,
        use_clr=use_clr,
        pseudocount=args.clr_pseudocount,
        use_scaler=use_scaler,
        min_explained_variance=args.pca_min_variance,
        pca_random_state=args.random_state,
        n_feature_dims=n_feat_cap,
    )
    print(
        _prefixed(
            prefix,
            f"PCA - min_explained_variance: {args.pca_min_variance}, "
            f"CLR: {'on' if use_clr else 'off'}",
        ),
        flush=True,
    )
    return n_components_grid


def run_classifier(args: argparse.Namespace, root: Path) -> int:
    _validate_basic_args(args)
    _print_experiment_line(args)
    prefix = str(getattr(args, "log_prefix", ""))
    results_json_path = _results_json_out_path(
        root,
        args.results_json,
        features=str(args.features),
        task=args.task,
        model=args.model,
    )
    config_path = root / "defaults.yaml"

    if args.features == "tetramer":
        feature_df, label_column = _load_tetramer_table(args.csv, args.label_column)
    else:
        expected = load_run_split_map(config_path=config_path)
        cap_df = _load_cap_csv(Path(args.csv))
        _validate_csv_splits(cap_df, expected)
        feature_df = cap_df
        label_column = _resolve_label_column(feature_df.columns, args.label_column)

    feature_df = _attach_shared_splits(feature_df, config_path=config_path)
    if args.features == "tetramer":
        splits = _build_task_splits_tetramer(feature_df, label_column, args.task)
        n_features = len(TETRAMERS)
    else:
        task_df_preview = _prepare_task_table(feature_df, label_column, args.task)
        feature_cols = [c for c in task_df_preview.columns if c.startswith("cluster_")]
        splits = _build_task_splits_uc_cap(feature_df, label_column, args.task, feature_cols)
        n_features = len(feature_cols)

    _require_binary_classes(splits.y_train, split_name="train split", task=args.task)
    _require_binary_classes(splits.y_val, split_name="validation split", task=args.task)
    _require_binary_classes(splits.y_test, split_name="test split", task=args.task)
    _print_dataset_summary(splits, prefix=prefix)

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
    print(_prefixed(prefix, "Best validation:"), flush=True)
    print(_prefixed(prefix, f"  {args.scoring}: {tuning.validation_score:.6f}"), flush=True)
    print(
        _prefixed(prefix, f"  Hyperparameters - {_format_hyperparameters(tuning.best_params)}"),
        flush=True,
    )

    evaluation = _evaluate_model(pipe, splits)
    _print_evaluation(args.model, evaluation, prefix=prefix)

    if results_json_path is not None:
        _write_results_json(
            results_json_path,
            args=args,
            label_column=label_column,
            n_components_grid=n_components_grid,
            tuning=tuning,
            evaluation=evaluation,
            n_features=n_features,
        )
    return 0


def _parse_main_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--tetramer", action="store_true", help="Use tetramer frequency features.")
    mx.add_argument("--uc_cap", action="store_true", help="Use UC/CAP cluster features.")
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help=(
            "Optional 1-based experiment index from experiments.yaml fit_classifier. "
            "If omitted, use defaults.yaml fit_classifier only."
        ),
    )
    parser.add_argument(
        "--feat",
        type=int,
        default=None,
        help="UC/CAP only: 1-based feature-set index (experiments.yaml run_uc_cap_pipeline). "
        "Omit for baseline-only merged config from defaults.yaml.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="?",
        const="",
        default=argparse.SUPPRESS,
        help=(
            "Override results JSON path from config. With no path, writes under results/scratch/ "
            "with an auto-generated name ({features}_{task}_{model}_{utc}.json). "
            "Omit entirely to use defaults.yaml / experiments only."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.expt is not None and args.expt <= 0:
        raise SystemExit(
            "--expt must be a positive integer (1-based experiment index), or omit for defaults."
        )
    if args.tetramer and args.feat is not None:
        raise SystemExit("--feat is only valid with --uc_cap.")
    if args.uc_cap and args.feat is not None and args.feat < 1:
        raise SystemExit("--feat must be >= 1 when provided.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parent.parent
    cli = _parse_main_argv(argv)
    expt = int(cli.expt) if cli.expt is not None else 0
    features = "tetramer" if cli.tetramer else "uc_cap"
    if hasattr(cli, "results_json"):
        results_json_cli: Optional[str] = cli.results_json
    else:
        results_json_cli = None

    args = _load_experiment_args(
        root,
        expt=expt,
        features=features,
        feat=cli.feat,
        results_json_cli=results_json_cli,
    )
    return run_classifier(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
