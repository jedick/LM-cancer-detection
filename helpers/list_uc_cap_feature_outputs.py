#!/usr/bin/env python3
"""Print space-resolved CAP CSV paths for Makefile (FEAT=0 / per-feature-set rules)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: helpers/list_uc_cap_feature_outputs.py <repo_root>",
            file=sys.stderr,
        )
        return 2
    r = Path(sys.argv[1])
    d = yaml.safe_load((r / "defaults.yaml").read_text(encoding="utf-8"))
    e = yaml.safe_load((r / "experiments.yaml").read_text(encoding="utf-8"))
    baseline = d["run_uc_cap_pipeline"]
    if not isinstance(baseline, list):
        raise SystemExit("defaults.yaml run_uc_cap_pipeline must be a list")
    base: dict = {}
    for frag in baseline:
        if not isinstance(frag, dict):
            raise SystemExit("defaults.yaml run_uc_cap_pipeline entries must be mappings")
        base = {**base, **frag}
    rows = e.get("run_uc_cap_pipeline") or []
    uc_root = str(d["paths"]["uc_cap_root"]).strip()
    paths: list[str] = []
    for row in rows:
        m = {**base, **row}
        nu, nk = int(m["n_uc"]), int(m["n_clusters"])
        nc = m["n_cap"]
        tag = (
            "all"
            if isinstance(nc, str) and str(nc).strip().lower() == "all"
            else str(int(nc))
        )
        ct = str(m["cap_transform"]).strip()
        stem = f"cap{tag}" if ct == "none" else f"cap{tag}_{ct}"
        paths.append(str(r / uc_root / f"uc{nu}_k{nk}" / f"{stem}.csv"))
    print(" ".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
