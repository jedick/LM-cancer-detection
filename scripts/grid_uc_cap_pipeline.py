#!/usr/bin/env python3
"""Run run_uc_cap_pipeline.py for each uc_cap_pipeline_grid entry in defaults.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    print(f"PyYAML is required ({exc})", file=sys.stderr)
    raise SystemExit(1) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=root / "defaults.yaml",
        help="Path to defaults.yaml (default: <repo>/defaults.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    grid: List[Any] = cfg.get("uc_cap_pipeline_grid") or []
    if not grid:
        print("uc_cap_pipeline_grid is empty in config.", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "run_uc_cap_pipeline.py"
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        return 1

    py = sys.executable
    n_total = len(grid)
    for i, row in enumerate(grid, start=1):
        if not isinstance(row, dict):
            print(f"Bad grid row {i}: expected mapping, got {type(row)}", file=sys.stderr)
            return 1
        n_uc = int(row["n_uc"])
        n_clusters = int(row["n_clusters"])
        n_cap = int(row["n_cap"])
        cmd = [
            py,
            str(script),
            "--n-uc",
            str(n_uc),
            "--n-clusters",
            str(n_clusters),
            "--n-cap",
            str(n_cap),
        ]
        print(
            f"[{i}/{n_total}] run_uc_cap_pipeline.py --n-uc {n_uc} "
            f"--n-clusters {n_clusters} --n-cap {n_cap}",
            flush=True,
        )
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=str(root), check=False)
        if proc.returncode != 0:
            return int(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
