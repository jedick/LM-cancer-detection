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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

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


def _default_cache_parquet_from_config(repo_root: Path, config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        uc_cap_root_rel = str(cfg["paths"]["uc_cap_root"]).strip()
        n_max = int(cfg["sequence_cache"]["n_max_per_run"])
    except (TypeError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"Invalid pipeline config for cache path defaults in {config_path}: {exc}"
        ) from exc
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_config_path = root / "defaults.yaml"
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=default_config_path)
    bootstrap_args, _ = bootstrap.parse_known_args(list(argv) if argv is not None else None)
    config_path = bootstrap_args.config
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        paths_cfg = cfg["paths"]
        uc_cfg = cfg["run_uc_cap_defaults"]
        first_grid = cfg["uc_cap_pipeline_grid"][0]
        default_outputs_dir = root / str(paths_cfg["outputs_dir"]).strip()
        default_out_dir = root / str(paths_cfg["uc_cap_root"]).strip()
        default_n_uc = int(first_grid["n_uc"])
        default_n_cap = str(first_grid["n_cap"])
        default_n_clusters = int(first_grid["n_clusters"])
        default_random_state = int(uc_cfg["random_state"])
        default_pca_variance = float(uc_cfg["pca_variance"])
        default_pca_components = uc_cfg.get("pca_components")
        if default_pca_components is not None:
            default_pca_components = int(default_pca_components)
        default_no_seq_normalize = not bool(uc_cfg["seq_normalize"])
        default_seq_log1p = bool(uc_cfg["seq_log1p"])
        default_cap_transform = str(uc_cfg["cap_transform"]).strip()
        default_clr_pseudocount = float(uc_cfg["clr_pseudocount"])
        default_batch_size = int(uc_cfg["batch_size"])
        default_max_iter = int(uc_cfg["max_iter"])
        default_chunk_size = int(uc_cfg["chunk_size"])
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise SystemExit(f"Invalid pipeline defaults in {config_path}: {exc}") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Path to defaults.yaml (used to derive default cache parquet path).",
    )
    parser.add_argument(
        "--cache-parquet",
        type=Path,
        default=None,
        help="Cache Parquet with sequence-level tetramer counts (default: derived from defaults.yaml).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=default_outputs_dir,
        help="Root directory containing per-run sequence count files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory to write UC/CAP artifacts.",
    )
    parser.add_argument(
        "--n-uc",
        type=int,
        default=default_n_uc,
        help="Sequences per run for UC fit (default: first uc_cap_pipeline_grid entry).",
    )
    parser.add_argument(
        "--n-cap",
        type=str,
        default=default_n_cap,
        help="Sequences per run for CAP assignment, or 'all' (default: first uc_cap_pipeline_grid entry).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=default_n_clusters,
        help="K for k-means (default: first uc_cap_pipeline_grid entry).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=default_random_state,
        help="Random seed (default: run_uc_cap_defaults.random_state).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=default_pca_components,
        help="Optional fixed PCA component count before k-means (default: use --pca-variance).",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=default_pca_variance,
        help="Cumulative explained-variance target for PCA when --pca-components is omitted.",
    )
    parser.add_argument(
        "--no-seq-normalize",
        action="store_true",
        default=default_no_seq_normalize,
        help="Disable per-sequence normalization to relative 4-mer frequencies.",
    )
    parser.add_argument(
        "--seq-log1p",
        action="store_true",
        default=default_seq_log1p,
        help="Apply log1p after optional per-sequence normalization.",
    )
    parser.add_argument(
        "--cap-transform",
        choices=("none", "clr"),
        default=default_cap_transform,
        help="Optional transform applied to run-level CAP vectors.",
    )
    parser.add_argument(
        "--clr-pseudocount",
        type=float,
        default=default_clr_pseudocount,
        help="Pseudocount used when --cap-transform=clr.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="MiniBatchKMeans batch size.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=default_max_iter,
        help="MiniBatchKMeans max iterations.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=default_chunk_size,
        help="CSV chunk size when streaming n_cap=all.",
    )
    parser.add_argument(
        "--refit-uc",
        action="store_true",
        help="Force refit of UC model and assignments if cached UC artifacts exist.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    args.cache_parquet = (
        args.cache_parquet
        if args.cache_parquet is not None
        else _default_cache_parquet_from_config(repo_root, args.config)
    )
    try:
        n_cap = parse_n_cap(args.n_cap)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.n_uc <= 0:
        raise SystemExit("--n-uc must be positive.")
    if args.n_clusters <= 1:
        raise SystemExit("--n-clusters must be > 1.")
    if args.pca_components is not None and args.pca_components <= 0:
        raise SystemExit("--pca-components must be positive when provided.")
    if not (0.0 < args.pca_variance <= 1.0):
        raise SystemExit("--pca-variance must be in (0, 1].")
    if args.clr_pseudocount <= 0:
        raise SystemExit("--clr-pseudocount must be positive.")
    if args.batch_size <= 0 or args.max_iter <= 0 or args.chunk_size <= 0:
        raise SystemExit("--batch-size, --max-iter, and --chunk-size must be positive.")
    if not args.cache_parquet.is_file():
        raise SystemExit(f"Cache parquet not found: {args.cache_parquet}")

    start = time.perf_counter()
    transform = SequenceTransform(
        normalize=not args.no_seq_normalize,
        log1p=args.seq_log1p,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_needed_from_cache = max(args.n_uc, n_cap if isinstance(n_cap, int) else args.n_uc)
    print(
        f"Loading cache rows with sequence_index <= {n_needed_from_cache}",
        flush=True,
    )
    cache_df = load_cache_subset(args.cache_parquet, n_needed_from_cache)
    if cache_df.empty:
        raise SystemExit("No sequence rows found in cache for requested settings.")
    if len(cache_df.columns) != 259:
        raise SystemExit("Cache schema mismatch: expected 259 columns.")
    metadata = build_run_metadata(config_path=args.config).loc[
        :, ["cancer_type", "study_name", "Run", "sample_label"]
    ]
    metadata = metadata.drop_duplicates(subset=["study_name", "Run"])
    run_split_map = load_run_split_map(config_path=args.config)
    train_runs = {run for run, split in run_split_map.items() if split == "train"}

    uc_dir = args.out_dir / f"uc{args.n_uc}_k{args.n_clusters}"
    uc_dir.mkdir(parents=True, exist_ok=True)
    uc_assign_out = uc_dir / "uc_assignments.csv"
    model_out = uc_dir / "uc_model.pkl"
    reuse_uc = (not args.refit_uc) and uc_assign_out.is_file() and model_out.is_file()

    if reuse_uc:
        print("Reusing existing UC model and assignments", flush=True)
        with open(model_out, "rb") as f:
            model_payload = pickle.load(f)
        km = model_payload["kmeans"]
        pca = model_payload["pca"]
        saved_train_runs = model_payload.get("split_train_runs")
        if saved_train_runs is None:
            raise SystemExit(
                "Existing UC model was fit without shared train-only runs. "
                "Use --refit-uc to rebuild."
            )
        if set(saved_train_runs) != train_runs:
            raise SystemExit(
                "Existing UC model train split does not match current shared split. "
                "Use --refit-uc to rebuild."
            )
        saved_transform = model_payload.get("transform", {})
        if (
            saved_transform.get("normalize") != transform.normalize
            or saved_transform.get("log1p") != transform.log1p
        ):
            raise SystemExit(
                "Existing UC model transform settings do not match current sequence "
                "transform flags. Use --refit-uc to rebuild."
            )
        uc_df = pd.read_csv(uc_assign_out)
    else:
        uc_df = cache_df[
            (cache_df["sequence_index"] <= args.n_uc) & (cache_df["Run"].isin(train_runs))
        ]
        if uc_df.empty:
            raise SystemExit(
                f"No UC rows found for training runs with --n-uc={args.n_uc}"
            )
        print(
            f"UC fit set (train runs only): {len(uc_df)} sequences, "
            f"{uc_df[['study_name', 'Run']].drop_duplicates().shape[0]} runs",
            flush=True,
        )
        km, pca, uc_labels = fit_uc_model(
            uc_df,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            transform=transform,
            pca_components=args.pca_components,
            pca_variance=args.pca_variance,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
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
            n_clusters=args.n_clusters,
        )
    else:
        run_cluster_counts = stream_cap_all_sequences(
            outputs_dir=args.outputs_dir,
            transform=transform,
            pca=pca,
            km=km,
            n_clusters=args.n_clusters,
            chunk_size=args.chunk_size,
        )

    cap_df = make_cap_dataframe(
        run_cluster_counts,
        n_clusters=args.n_clusters,
        cap_transform=args.cap_transform,
        clr_pseudocount=args.clr_pseudocount,
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
        f"cap{run_tag}"
        if args.cap_transform == "none"
        else f"cap{run_tag}_{args.cap_transform}"
    )
    cap_out = uc_dir / f"{cap_name}.csv"
    config_out = uc_dir / f"{cap_name}.json"

    cap_df.to_csv(cap_out, index=False)
    split_counts = cap_df["split"].value_counts().to_dict()
    paths_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["paths"]

    def _resolve_cfg_path(raw: object) -> Path:
        p = Path(str(raw).strip())
        return p if p.is_absolute() else repo_root / p

    datasets_csv_abs = _resolve_cfg_path(paths_cfg["datasets_csv"])
    data_dir_abs = _resolve_cfg_path(paths_cfg["data_dir"])
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_parquet": str(args.cache_parquet),
                "outputs_dir": str(args.outputs_dir),
                "datasets_csv": str(datasets_csv_abs),
                "data_dir": str(data_dir_abs),
                "n_uc": args.n_uc,
                "n_cap": n_cap,
                "n_clusters": args.n_clusters,
                "random_state": args.random_state,
                "pca_components": args.pca_components,
                "pca_variance": args.pca_variance,
                "pca_components_used": pca_components_used,
                "seq_normalize": transform.normalize,
                "seq_log1p": transform.log1p,
                "cap_transform": args.cap_transform,
                "clr_pseudocount": args.clr_pseudocount,
                "batch_size": args.batch_size,
                "max_iter": args.max_iter,
                "chunk_size": args.chunk_size,
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
            },
            f,
            indent=2,
        )

    elapsed = time.perf_counter() - start
    print(
        f"UC clusters used: {uc_unique_clusters}/{args.n_clusters}",
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


if __name__ == "__main__":
    raise SystemExit(main())
