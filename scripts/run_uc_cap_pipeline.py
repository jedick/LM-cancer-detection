#!/usr/bin/env python3
"""
Run UC/CAP with decoupled sequence budgets for clustering vs abundance profiling.

UC (unsupervised clustering):
  - Fit MiniBatchKMeans on the first n_uc sequences per run from a cache Parquet.

CAP (cluster abundance profiles):
  - Assign n_cap sequences per run to nearest centroid and aggregate K-dimensional
    cluster count/abundance vectors per run.
  - n_cap may be an integer or "all". When "all", sequence rows are streamed from
    per-run sequence count files under outputs/<cancer_type>/<study_name>/<Run>.csv(.xz).

Input sequence features are tetramer count vectors (256 columns).

Configuration is read from <repo>/defaults.yaml (run_uc_cap_pipeline baseline list,
merged in order, then overlaid by experiments.yaml when --feat is set)
and optional <repo>/experiments.yaml (--feat 1-based index into the flat list there,
merged shallowly over the baseline). The only CLI flag is --feat.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from shared_splits import build_run_metadata, load_run_split_map


RUN_FILE_PATTERN = re.compile(r"^(SRR|ERR|DRR)\d+$")
TETRAMERS: Tuple[str, ...] = tuple(
    "".join(p) for p in itertools.product("ACGT", repeat=4)
)


class SequenceTransform:
    """Feature transform for sequence-level 4-mer count vectors."""

    def __init__(self, normalize: bool, log1p: bool):
        self.normalize = normalize
        self.log1p = log1p

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X.astype(np.float64, copy=True)
        if self.normalize:
            totals = Z.sum(axis=1, keepdims=True)
            np.divide(Z, totals, out=Z, where=(totals > 0))
        if self.log1p:
            Z = np.log1p(Z)
        return Z


def parse_n_cap(raw: str) -> Union[int, str]:
    if raw.strip().lower() == "all":
        return "all"
    value = int(raw)
    if value <= 0:
        raise ValueError("n_cap must be positive or 'all'")
    return value


def _default_cache_parquet(repo_root: Path, defaults_cfg: Mapping[str, Any]) -> Path:
    try:
        paths_cfg = defaults_cfg["paths"]
        uc_cap_root_rel = str(paths_cfg["uc_cap_root"]).strip()
        n_max = int(defaults_cfg["sequence_cache"]["n_max_per_run"])
    except (TypeError, KeyError, ValueError) as exc:
        raise SystemExit(f"Invalid pipeline config for cache path defaults: {exc}") from exc
    return repo_root / uc_cap_root_rel / f"sequence_counts_first_{n_max}_all_runs.parquet"


def iter_run_files(outputs_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (study_name, run, path) for per-run sequence count files."""
    for path in sorted(outputs_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        run = ""
        if path.name.endswith(".csv.xz"):
            run = path.name[: -len(".csv.xz")]
        elif path.name.endswith(".csv"):
            run = path.name[: -len(".csv")]
        if not run or not RUN_FILE_PATTERN.match(run):
            continue
        study_name = path.parent.name
        yield study_name, run, path


def load_cache_subset(cache_parquet: Path, n_limit: int) -> pd.DataFrame:
    cols = ["study_name", "Run", "sequence_index"] + list(TETRAMERS)
    dataset = ds.dataset(str(cache_parquet), format="parquet")
    table = dataset.to_table(
        columns=cols,
        filter=pc.field("sequence_index") <= pc.scalar(n_limit),
    )
    return table.to_pandas()


def fit_uc_model(
    uc_df: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int,
    transform: SequenceTransform,
    pca_components: Optional[int],
    pca_variance: float,
    batch_size: int,
    max_iter: int,
) -> Tuple[MiniBatchKMeans, PCA, np.ndarray]:
    X = uc_df.loc[:, list(TETRAMERS)].to_numpy(dtype=np.float64, copy=False)
    X_t = transform.transform(X)
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state)
    else:
        pca = PCA(n_components=pca_variance, random_state=random_state)
    X_fit = pca.fit_transform(X_t)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init="auto",
    )
    km.fit(X_fit)
    labels = km.predict(X_fit)
    return km, pca, labels


def assign_matrix(
    X_counts: np.ndarray,
    *,
    transform: SequenceTransform,
    pca: PCA,
    km: MiniBatchKMeans,
) -> np.ndarray:
    X_t = transform.transform(X_counts)
    X_t = pca.transform(X_t)
    return km.predict(X_t)


def update_run_cluster_counts(
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray],
    run_key: Tuple[str, str],
    cluster_ids: np.ndarray,
    n_clusters: int,
) -> None:
    acc = run_cluster_counts.get(run_key)
    if acc is None:
        acc = np.zeros(n_clusters, dtype=np.int64)
        run_cluster_counts[run_key] = acc
    acc += np.bincount(cluster_ids, minlength=n_clusters).astype(np.int64, copy=False)


def build_cap_from_cache(
    *,
    cache_df: pd.DataFrame,
    n_cap: int,
    transform: SequenceTransform,
    pca: PCA,
    km: MiniBatchKMeans,
    n_clusters: int,
) -> Dict[Tuple[str, str], np.ndarray]:
    subset = cache_df[cache_df["sequence_index"] <= n_cap]
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray] = {}
    for (study_name, run), grp in subset.groupby(["study_name", "Run"], sort=True):
        X = grp.loc[:, list(TETRAMERS)].to_numpy(dtype=np.float64, copy=False)
        cluster_ids = assign_matrix(X, transform=transform, pca=pca, km=km)
        update_run_cluster_counts(
            run_cluster_counts, (study_name, run), cluster_ids, n_clusters
        )
    return run_cluster_counts


def stream_cap_all_sequences(
    *,
    outputs_dir: Path,
    transform: SequenceTransform,
    pca: PCA,
    km: MiniBatchKMeans,
    n_clusters: int,
    chunk_size: int,
) -> Dict[Tuple[str, str], np.ndarray]:
    run_files = list(iter_run_files(outputs_dir))
    if not run_files:
        raise SystemExit(f"No per-run sequence count files found under {outputs_dir}")

    run_cluster_counts: Dict[Tuple[str, str], np.ndarray] = {}
    n_runs = len(run_files)
    for i, (study_name, run, path) in enumerate(run_files, start=1):
        line = f"  CAP all-seq: {i}/{n_runs} runs (current: {study_name}/{run})"
        sys.stdout.write("\r" + line.ljust(100))
        sys.stdout.flush()
        try:
            chunks: Iterator[pd.DataFrame] = pd.read_csv(
                path,
                header=None,
                chunksize=chunk_size,
                dtype="int64",
                compression="infer",
            )
            for chunk in chunks:
                if chunk.shape[1] != 256:
                    raise SystemExit(
                        f"Expected 256 columns in {path}, found {chunk.shape[1]}"
                    )
                X = chunk.to_numpy(dtype=np.float64, copy=False)
                cluster_ids = assign_matrix(X, transform=transform, pca=pca, km=km)
                update_run_cluster_counts(
                    run_cluster_counts, (study_name, run), cluster_ids, n_clusters
                )
        except Exception as exc:
            sys.stdout.write("\n")
            raise SystemExit(f"Failed reading/assigning {path}: {exc}") from exc
    sys.stdout.write("\n")
    return run_cluster_counts


def make_cap_dataframe(
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray],
    n_clusters: int,
    cap_transform: str,
    clr_pseudocount: float,
) -> pd.DataFrame:
    cluster_cols = [f"cluster_{i:03d}" for i in range(n_clusters)]
    rows: List[Dict[str, object]] = []
    for (study_name, run), counts in sorted(run_cluster_counts.items()):
        total = int(counts.sum())
        abund = counts.astype(np.float64)
        if total > 0:
            abund /= total
        if cap_transform == "clr":
            abund = np.log(abund + clr_pseudocount)
            abund = abund - abund.mean()
        row: Dict[str, object] = {
            "study_name": study_name,
            "Run": run,
            "n_assigned_sequences": total,
        }
        row.update({cluster_cols[i]: float(abund[i]) for i in range(n_clusters)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cap_sparsity(
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray],
) -> Tuple[float, float, int, int]:
    if not run_cluster_counts:
        return 0.0, 0.0, 0, 0
    nnz = np.asarray(
        [int(np.count_nonzero(counts)) for counts in run_cluster_counts.values()],
        dtype=np.int64,
    )
    return float(nnz.mean()), float(np.median(nnz)), int(nnz.min()), int(nnz.max())


def _n_cap_raw_to_str(raw: object) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, int):
        return str(raw)
    raise SystemExit(f"n_cap must be an int or string, got {type(raw).__name__}")


_FEAT_HELP = (
    "Optional 1-based index into experiments.yaml run_uc_cap_pipeline (selects a CAP "
    "feature-set row, merged over the defaults.yaml baseline). Omit to run the baseline only."
)


def _parse_feature_cli(argv: Optional[Sequence[str]]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feat", type=int, default=None, help=_FEAT_HELP)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.feat is not None and args.feat <= 0:
        raise SystemExit("--feat must be a positive integer (1-based index), or omit it.")
    return int(args.feat) if args.feat is not None else 0


def _shallow_merge_uc_cap(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(overlay)
    return out


def merge_defaults_uc_cap_baseline_fragments(baseline_rows: List[Any]) -> Dict[str, Any]:
    """Merge defaults.yaml run_uc_cap_pipeline (list of dicts) in order into one mapping."""
    merged: Dict[str, Any] = {}
    for i, frag in enumerate(baseline_rows):
        if not isinstance(frag, dict):
            raise SystemExit(
                f"defaults.yaml run_uc_cap_pipeline[{i}] must be a mapping, got {type(frag).__name__}."
            )
        merged = {**merged, **frag}
    return merged


def load_merged_uc_cap_config(repo_root: Path, *, feat: int) -> Tuple[Dict[str, Any], int]:
    """Return (merged pipeline dict, feature_index for logging). feat 0 = baseline only."""
    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"
    try:
        defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Failed to read {defaults_path}: {exc}") from exc

    try:
        baseline_rows = defaults_cfg["run_uc_cap_pipeline"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"defaults.yaml missing run_uc_cap_pipeline: {exc}") from exc
    if not isinstance(baseline_rows, list) or not baseline_rows:
        raise SystemExit(
            "defaults.yaml run_uc_cap_pipeline must be a non-empty list of mappings."
        )
    base = merge_defaults_uc_cap_baseline_fragments(baseline_rows)

    if feat == 0:
        return base, 0

    if not experiments_path.is_file():
        raise SystemExit(f"--feat {feat} requires {experiments_path}")
    try:
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Failed to read {experiments_path}: {exc}") from exc

    grid = experiments_cfg.get("run_uc_cap_pipeline")
    if not isinstance(grid, list) or not grid:
        raise SystemExit(f"No run_uc_cap_pipeline list in {experiments_path}")
    if feat > len(grid):
        raise SystemExit(
            f"--feat {feat} is out of range; experiments.yaml defines {len(grid)} feature-set rows."
        )
    row = grid[feat - 1]
    if not isinstance(row, dict):
        raise SystemExit(f"run_uc_cap_pipeline[{feat - 1}] must be a mapping.")
    return _shallow_merge_uc_cap(base, row), feat


def _uc_refit_error(uc_dir: Path, uc_assign_out: Path, model_out: Path, reason: str) -> None:
    raise SystemExit(
        f"{reason}\n"
        "Delete the cached UC artifacts and re-run this script, for example:\n"
        f"  rm -f {uc_assign_out} {model_out}\n"
        f"(or remove the whole directory {uc_dir}/ if you prefer a clean slate.)"
    )


def run_pipeline_from_merged(
    repo_root: Path,
    *,
    config_path: Path,
    merged: Dict[str, Any],
    feature_index: int,
) -> int:
    paths_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["paths"]

    def _resolve_cfg_path(raw: object) -> Path:
        p = Path(str(raw).strip())
        return p if p.is_absolute() else repo_root / p

    try:
        n_uc = int(merged["n_uc"])
        n_clusters = int(merged["n_clusters"])
        random_state = int(merged["random_state"])
        pca_variance = float(merged["pca_variance"])
        pca_components = merged.get("pca_components")
        if pca_components is not None:
            pca_components = int(pca_components)
        seq_normalize = bool(merged["seq_normalize"])
        seq_log1p = bool(merged["seq_log1p"])
        cap_transform = str(merged["cap_transform"]).strip()
        clr_pseudocount = float(merged["clr_pseudocount"])
        batch_size = int(merged["batch_size"])
        max_iter = int(merged["max_iter"])
        chunk_size = int(merged["chunk_size"])
        outputs_dir = _resolve_cfg_path(paths_cfg["outputs_dir"])
        out_dir = _resolve_cfg_path(paths_cfg["uc_cap_root"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid merged run_uc_cap_pipeline config: {exc}") from exc

    if cap_transform not in ("none", "clr"):
        raise SystemExit(f"cap_transform must be 'none' or 'clr', got {cap_transform!r}")

    n_cap_raw = _n_cap_raw_to_str(merged["n_cap"])
    try:
        n_cap = parse_n_cap(n_cap_raw)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if n_uc <= 0:
        raise SystemExit("n_uc must be positive.")
    if n_clusters <= 1:
        raise SystemExit("n_clusters must be > 1.")
    if pca_components is not None and pca_components <= 0:
        raise SystemExit("pca_components must be positive when provided.")
    if not (0.0 < pca_variance <= 1.0):
        raise SystemExit("pca_variance must be in (0, 1].")
    if clr_pseudocount <= 0:
        raise SystemExit("clr_pseudocount must be positive.")
    if batch_size <= 0 or max_iter <= 0 or chunk_size <= 0:
        raise SystemExit("batch_size, max_iter, and chunk_size must be positive.")

    defaults_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_parquet = _default_cache_parquet(repo_root, defaults_cfg)
    if not cache_parquet.is_file():
        raise SystemExit(f"Cache parquet not found: {cache_parquet}")

    prefix = f"F{feature_index} " if feature_index > 0 else ""
    print(f"{prefix}n_uc={n_uc} n_clusters={n_clusters} n_cap={n_cap_raw}", flush=True)

    start = time.perf_counter()
    transform = SequenceTransform(normalize=seq_normalize, log1p=seq_log1p)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_needed_from_cache = max(n_uc, n_cap if isinstance(n_cap, int) else n_uc)
    print(
        f"Loading cache rows with sequence_index <= {n_needed_from_cache}",
        flush=True,
    )
    cache_df = load_cache_subset(cache_parquet, n_needed_from_cache)
    if cache_df.empty:
        raise SystemExit("No sequence rows found in cache for requested settings.")
    if len(cache_df.columns) != 259:
        raise SystemExit("Cache schema mismatch: expected 259 columns.")
    metadata = build_run_metadata(config_path=config_path).loc[
        :, ["cancer_type", "study_name", "Run", "sample_label"]
    ]
    metadata = metadata.drop_duplicates(subset=["study_name", "Run"])
    run_split_map = load_run_split_map(config_path=config_path)
    train_runs = {run for run, split in run_split_map.items() if split == "train"}

    uc_dir = out_dir / f"uc{n_uc}_k{n_clusters}"
    uc_dir.mkdir(parents=True, exist_ok=True)
    uc_assign_out = uc_dir / "uc_assignments.csv"
    model_out = uc_dir / "uc_model.pkl"
    reuse_uc = uc_assign_out.is_file() and model_out.is_file()

    if reuse_uc:
        print("Reusing existing UC model and assignments", flush=True)
        with open(model_out, "rb") as f:
            model_payload = pickle.load(f)
        km = model_payload["kmeans"]
        pca = model_payload["pca"]
        if int(km.n_clusters) != n_clusters:
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                f"Saved UC model has n_clusters={km.n_clusters} but config requests "
                f"n_clusters={n_clusters}.",
            )
        saved_train_runs = model_payload.get("split_train_runs")
        if saved_train_runs is None:
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                "Existing UC model was fit without shared train-only runs metadata.",
            )
        if set(saved_train_runs) != train_runs:
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                "Existing UC model train split does not match the current shared split.",
            )
        saved_transform = model_payload.get("transform", {})
        if (
            saved_transform.get("normalize") != transform.normalize
            or saved_transform.get("log1p") != transform.log1p
        ):
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                "Existing UC model sequence transform settings do not match config.",
            )
        uc_df = pd.read_csv(uc_assign_out)
    else:
        uc_df = cache_df[
            (cache_df["sequence_index"] <= n_uc) & (cache_df["Run"].isin(train_runs))
        ]
        if uc_df.empty:
            raise SystemExit(f"No UC rows found for training runs with n_uc={n_uc}")
        print(
            f"UC fit set (train runs only): {len(uc_df)} sequences, "
            f"{uc_df[['study_name', 'Run']].drop_duplicates().shape[0]} runs",
            flush=True,
        )
        km, pca, uc_labels = fit_uc_model(
            uc_df,
            n_clusters=n_clusters,
            random_state=random_state,
            transform=transform,
            pca_components=pca_components,
            pca_variance=pca_variance,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        uc_df = uc_df.loc[:, ["study_name", "Run", "sequence_index"]].copy()
        uc_df["cluster_id"] = uc_labels
        uc_df.to_csv(uc_assign_out, index=False)
        with open(model_out, "wb") as f:
            pickle.dump(
                {
                    "kmeans": km,
                    "pca": pca,
                    "transform": {
                        "normalize": transform.normalize,
                        "log1p": transform.log1p,
                    },
                    "split_train_runs": sorted(train_runs),
                    "tetramers": list(TETRAMERS),
                },
                f,
            )

    uc_unique_clusters = int(uc_df["cluster_id"].nunique())
    print(f"PCA components retained: {pca.n_components_}", flush=True)
    pca_components_used = int(pca.n_components_)

    print("Building CAP", flush=True)
    if isinstance(n_cap, int):
        run_cluster_counts = build_cap_from_cache(
            cache_df=cache_df,
            n_cap=n_cap,
            transform=transform,
            pca=pca,
            km=km,
            n_clusters=n_clusters,
        )
    else:
        run_cluster_counts = stream_cap_all_sequences(
            outputs_dir=outputs_dir,
            transform=transform,
            pca=pca,
            km=km,
            n_clusters=n_clusters,
            chunk_size=chunk_size,
        )

    cap_df = make_cap_dataframe(
        run_cluster_counts,
        n_clusters=n_clusters,
        cap_transform=cap_transform,
        clr_pseudocount=clr_pseudocount,
    )
    cap_df = cap_df.merge(metadata, on=["study_name", "Run"], how="left")
    cap_df["split"] = cap_df["Run"].map(run_split_map).fillna("unsplit")

    cap_n_assigned_min = int(cap_df["n_assigned_sequences"].min())
    cap_n_assigned_max = int(cap_df["n_assigned_sequences"].max())
    cap_nnz_mean, cap_nnz_median, cap_nnz_min, cap_nnz_max = summarize_cap_sparsity(
        run_cluster_counts
    )

    run_tag = "all" if n_cap == "all" else str(n_cap)
    cap_name = (
        f"cap{run_tag}" if cap_transform == "none" else f"cap{run_tag}_{cap_transform}"
    )
    cap_out = uc_dir / f"{cap_name}.csv"
    config_out = uc_dir / f"{cap_name}.json"

    cap_df.to_csv(cap_out, index=False)
    split_counts = cap_df["split"].value_counts().to_dict()

    datasets_csv_abs = _resolve_cfg_path(paths_cfg["datasets_csv"])
    data_dir_abs = _resolve_cfg_path(paths_cfg["data_dir"])
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_parquet": str(cache_parquet),
                "outputs_dir": str(outputs_dir),
                "datasets_csv": str(datasets_csv_abs),
                "data_dir": str(data_dir_abs),
                "n_uc": n_uc,
                "n_cap": n_cap,
                "n_clusters": n_clusters,
                "random_state": random_state,
                "pca_components": pca_components,
                "pca_variance": pca_variance,
                "pca_components_used": pca_components_used,
                "seq_normalize": transform.normalize,
                "seq_log1p": transform.log1p,
                "cap_transform": cap_transform,
                "clr_pseudocount": clr_pseudocount,
                "batch_size": batch_size,
                "max_iter": max_iter,
                "chunk_size": chunk_size,
                "cap_output_csv": str(cap_out),
                "uc_assignments_csv": str(uc_assign_out),
                "uc_model_pkl": str(model_out),
                "uc_unique_clusters": uc_unique_clusters,
                "cap_runs": int(len(cap_df)),
                "cap_split_counts": split_counts,
                "cap_n_assigned_min": cap_n_assigned_min,
                "cap_n_assigned_max": cap_n_assigned_max,
                "cap_nonzero_clusters_mean": cap_nnz_mean,
                "cap_nonzero_clusters_median": cap_nnz_median,
                "cap_nonzero_clusters_min": cap_nnz_min,
                "cap_nonzero_clusters_max": cap_nnz_max,
                "feature_index": feature_index,
            },
            f,
            indent=2,
        )

    elapsed = time.perf_counter() - start
    print(
        f"UC clusters used: {uc_unique_clusters}/{n_clusters}",
        flush=True,
    )
    print(
        "CAP nonzero clusters per run "
        f"(mean/median/min/max): {cap_nnz_mean:.2f}/{cap_nnz_median:.1f}/{cap_nnz_min}/{cap_nnz_max}",
        flush=True,
    )
    print(f"Output directory: {uc_dir}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "defaults.yaml"
    feat = _parse_feature_cli(argv)
    merged, feature_index = load_merged_uc_cap_config(repo_root, feat=feat)
    if feature_index == 0:
        print("Baseline config (defaults.yaml only)", flush=True)
    return run_pipeline_from_merged(
        repo_root,
        config_path=config_path,
        merged=merged,
        feature_index=feature_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
