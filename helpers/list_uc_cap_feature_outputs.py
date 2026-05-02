#!/usr/bin/env python3
"""List CAP CSV paths for UC/CAP Makefile rules (baseline or experiment feature sets)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def merge_run_uc_cap_baseline(defaults_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Shallow-merge ``defaults.yaml`` ``run_uc_cap_pipeline`` list (same as no ``--feat``)."""
    baseline = defaults_cfg["run_uc_cap_pipeline"]
    if not isinstance(baseline, list):
        raise SystemExit("defaults.yaml run_uc_cap_pipeline must be a list")
    merged: Dict[str, Any] = {}
    for frag in baseline:
        if not isinstance(frag, dict):
            raise SystemExit("defaults.yaml run_uc_cap_pipeline entries must be mappings")
        merged = {**merged, **frag}
    return merged


def cap_csv_path(repo_root: Path, defaults_cfg: Mapping[str, Any], merged: Mapping[str, Any]) -> str:
    """Return repo-relative POSIX path to the CAP CSV for ``merged`` UC/CAP parameters."""
    uc_root = str(defaults_cfg["paths"]["uc_cap_root"]).strip()
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
    path = repo_root / uc_root / f"uc{n_uc}_k{n_clusters}" / f"{stem}.csv"
    # No .resolve() so Make's $(ROOT)/... matches this target string when the repo is symlinked.
    return path.as_posix()


def _load_defaults(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "defaults.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_experiments(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "experiments.yaml"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print CAP CSV paths under the repo: all experiment rows (default), "
            "or a single baseline path with --baseline."
        )
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="Repository root (directory containing defaults.yaml).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Print only the baseline CAP path (defaults.yaml merge; no experiments row).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root
    defaults_cfg = _load_defaults(repo_root)
    base = merge_run_uc_cap_baseline(defaults_cfg)

    if args.baseline:
        print(cap_csv_path(repo_root, defaults_cfg, base))
        return 0

    experiments_cfg = _load_experiments(repo_root)
    rows = experiments_cfg.get("run_uc_cap_pipeline") or []
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("experiments.yaml run_uc_cap_pipeline entries must be mappings")
        merged = {**base, **row}
        paths.append(cap_csv_path(repo_root, defaults_cfg, merged))
    print(" ".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
