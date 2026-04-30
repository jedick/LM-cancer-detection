#!/usr/bin/env python3
"""Summarize make --trace output as an indented dependency tree with mtimes."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TRACE_RE = re.compile(r"^[^:]+:\d+: update target '(.+)' due to: (.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run `make -n --trace <target>` and print a compact dependency/mtime summary."
        )
    )
    parser.add_argument("target", help="make target to explain")
    parser.add_argument(
        "--make-bin",
        default="make",
        help="make executable to use (default: make)",
    )
    parser.add_argument(
        "--show-raw-trace",
        action="store_true",
        help="print the raw make trace before the compact summary",
    )
    return parser.parse_args()


def run_make_trace(make_bin: str, target: str, cwd: Path) -> str:
    result = subprocess.run(
        [make_bin, "-n", "--trace", target],
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )
    trace = result.stdout
    if result.returncode != 0:
        if trace:
            sys.stderr.write(trace)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return trace


def maybe_path_tokens(reason: str) -> list[str]:
    # make --trace gives the specific prerequisites that triggered rebuild.
    # We keep path-like tokens and drop explanatory text (e.g. "target is .PHONY").
    if reason.startswith("target is .PHONY"):
        return []
    tokens: list[str] = []
    for token in reason.split():
        cleaned = token.strip(",")
        if cleaned.startswith("/") or cleaned.startswith("./") or cleaned.startswith("../"):
            tokens.append(cleaned)
    return tokens


def fmt_target(path_like: str, repo_root: Path) -> str:
    p = Path(path_like)
    try:
        if p.is_absolute():
            return str(p.relative_to(repo_root))
    except ValueError:
        return str(p)
    return str(p)


def fmt_mtime(path_like: str, repo_root: Path) -> str:
    p = Path(path_like)
    if not p.is_absolute():
        p = repo_root / p
    if not p.exists():
        return "missing"
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def is_newer(child: str, parent: str, repo_root: Path) -> bool | None:
    c = Path(child)
    p = Path(parent)
    if not c.is_absolute():
        c = repo_root / c
    if not p.is_absolute():
        p = repo_root / p
    if not c.exists() or not p.exists():
        return None
    return c.stat().st_mtime > p.stat().st_mtime


def build_summary(trace: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    edges: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    for line in trace.splitlines():
        match = TRACE_RE.match(line.strip())
        if not match:
            continue
        target, reason = match.groups()
        reasons[target] = reason
        deps = maybe_path_tokens(reason)
        if deps:
            edges[target] = deps
        else:
            edges.setdefault(target, [])
    return edges, reasons


def print_tree(
    node: str,
    edges: dict[str, list[str]],
    reasons: dict[str, str],
    repo_root: Path,
    level: int = 0,
    parent: str | None = None,
    seen: set[str] | None = None,
) -> None:
    if seen is None:
        seen = set()

    indent = "  " * level
    label = fmt_target(node, repo_root)
    mtime = fmt_mtime(node, repo_root)

    relation = ""
    if parent is not None:
        newer = is_newer(node, parent, repo_root)
        if newer is True:
            relation = " (newer than parent)"
        elif newer is False:
            relation = " (older than parent)"

    print(f"{indent}{label} -> {mtime}{relation}")

    if node in seen:
        return
    seen.add(node)

    deps = edges.get(node, [])
    if not deps and node in reasons and reasons[node].startswith("target is .PHONY"):
        print(f"{indent}  [reason] target is .PHONY")
        return

    for dep in deps:
        print_tree(dep, edges, reasons, repo_root, level + 1, parent=node, seen=seen)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    trace = run_make_trace(args.make_bin, args.target, repo_root)
    if args.show_raw_trace:
        print(trace.rstrip())
        print()

    edges, reasons = build_summary(trace)

    requested = args.target
    requested_abs = str((repo_root / requested).resolve())

    # If the requested target name appears in trace, use it. Otherwise fallback to
    # resolved absolute path (useful when make normalizes to absolute paths).
    if not edges:
        print("No `update target ... due to ...` lines found in make trace.")
        return

    if requested in edges:
        root = requested
    elif requested_abs in edges:
        root = requested_abs
    else:
        root = next(iter(edges.keys()))

    # For phony roots, pivot to concrete rebuilt targets so the recursive chain is visible.
    if reasons.get(root, "").startswith("target is .PHONY") and not edges.get(root):
        dep_nodes = {dep for dep_list in edges.values() for dep in dep_list}
        concrete_roots = [
            t for t in edges.keys() if t not in dep_nodes and not reasons.get(t, "").startswith("target is .PHONY")
        ]
        if concrete_roots:
            for idx, concrete_root in enumerate(concrete_roots):
                if idx:
                    print()
                print_tree(concrete_root, edges, reasons, repo_root)
            return

    print_tree(root, edges, reasons, repo_root)


if __name__ == "__main__":
    main()
