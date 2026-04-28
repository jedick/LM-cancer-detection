#!/usr/bin/env python3
"""Read configs/pipeline.yaml for the root Makefile.

Modes:
  pipeline_emit_mk.py <config.yaml>
      Emit GNU Make assignments (one per line) for $(eval); not safe with $(shell),
      which collapses newlines — use --get instead.

  pipeline_emit_mk.py <config.yaml> --get dotted.key
      Print a single scalar (for use in Makefile $(shell ...)).

Examples: --get paths.data_dir   --get tetramer.default_task
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    print(f"pipeline_emit_mk.py: PyYAML is required ({exc})", file=sys.stderr)
    sys.exit(1)


def _escape_make_rhs(value: str) -> str:
    return value.replace("\\", "\\\\").replace("#", "\\#").replace("$", "$$")


def _get_dotted(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def _emit_all_mk(cfg: dict[str, Any]) -> None:
    for key, raw in cfg["paths"].items():
        if raw is None:
            continue
        var = f"PIPE_{key.upper()}"
        val = _escape_make_rhs(str(raw).strip())
        print(f"{var} := {val}")
    tet = cfg.get("tetramer") or {}
    print(f"PIPE_DEFAULT_TASK := {tet.get('default_task', 'cancer_diagnosis')}")
    classifiers = cfg.get("uc_cap_classifiers") or ["random_forest"]
    print(f"PIPE_DEFAULT_UC_CAP_CLASSIFIER := {classifiers[0]}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, help="Path to pipeline.yaml")
    p.add_argument(
        "--get",
        metavar="DOTTED.KEY",
        dest="dotted_key",
        default=None,
        help="Print one value for Make $(shell); trailing newline is stripped by Make.",
    )
    p.add_argument(
        "--render-cap-csv",
        action="store_true",
        help="Render paths.cap_csv_pattern with --n-uc/--n-clusters/--n-cap.",
    )
    p.add_argument("--n-uc", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--n-clusters", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--n-cap", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        print("pipeline_emit_mk.py: invalid YAML root", file=sys.stderr)
        return 1

    if args.render_cap_csv:
        if args.n_uc is None or args.n_clusters is None or args.n_cap is None:
            print(
                "pipeline_emit_mk.py: --render-cap-csv requires --n-uc --n-clusters --n-cap",
                file=sys.stderr,
            )
            return 1
        try:
            pattern = str(_get_dotted(cfg, "paths.cap_csv_pattern"))
            rendered = pattern.format(
                n_uc=int(args.n_uc),
                n_clusters=int(args.n_clusters),
                n_cap=int(args.n_cap),
            )
        except (KeyError, ValueError, TypeError) as exc:
            print(f"pipeline_emit_mk.py: cannot render cap_csv_pattern: {exc}", file=sys.stderr)
            return 1
        print(rendered)
        return 0

    if args.dotted_key:
        try:
            val = _get_dotted(cfg, args.dotted_key)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            print(f"pipeline_emit_mk.py: --get {args.dotted_key!r}: {exc}", file=sys.stderr)
            return 1
        # Keep empty string for nulls so Make can conditionally add args.
        if val is None:
            print("")
            return 0
        if isinstance(val, bool):
            print("true" if val else "false")
            return 0
        print(val)
        return 0

    if "paths" not in cfg:
        print("pipeline_emit_mk.py: missing paths", file=sys.stderr)
        return 1
    _emit_all_mk(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
