#!/usr/bin/env python3
"""
Build Table 4 (HyenaDNA) from JSON metrics under results/hyenadna/<cache>/.

Expected rows (cache max_length, model max_length):
  1k-1k, 2k-1k, 2k-2k, 4k-1k, 4k-2k, 4k-4k

For each row, this script loads one cancer_diagnosis result and one cancer_type
result, then prints an HTML table with nested headers:
  - max_length: cache, model
  - cancer_diagnosis: test, holdout
  - cancer_type: test, holdout
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TASKS = ("cancer_diagnosis", "cancer_type")
ROW_ORDER: Tuple[Tuple[int, int], ...] = (
    (1024, 1024),
    (2048, 1024),
    (2048, 2048),
    (4096, 1024),
    (4096, 2048),
    (4096, 4096),
)


def _fmt_cell(value: Optional[float], *, decimals: int) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def _fmt_len(value: int) -> str:
    return f"{int(value) // 1024}k"


def _load_candidate_paths(hyenadna_dir: Path) -> List[Path]:
    return sorted(
        [p for p in hyenadna_dir.glob("*/*.json") if p.is_file()],
        key=lambda p: (p.parent.name, p.name),
    )


def _read_result_payload(path: Path) -> Optional[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("config"), dict):
        return None
    if not isinstance(data.get("data"), dict):
        return None
    if not isinstance(data.get("metrics"), dict):
        return None
    return data


def _extract_lengths_and_task(path: Path, data: dict) -> Tuple[int, int, str]:
    task = data.get("config", {}).get("task")
    if task not in TASKS:
        raise SystemExit(f"{path}: expected config.task in {TASKS}, got {task!r}")

    cache_blob = data.get("data", {}).get("cache", {})
    cache_max_len = cache_blob.get("run_tensors_max_length")
    if cache_max_len is None:
        raise SystemExit(
            f"{path}: missing data.cache.run_tensors_max_length "
            "(required to map table rows)."
        )

    model_max_len = data.get("config", {}).get("max_length")
    if model_max_len is None:
        raise SystemExit(
            f"{path}: missing config.max_length "
            "(required to map table rows)."
        )

    return int(cache_max_len), int(model_max_len), str(task)


def _extract_auc_pair(path: Path, data: dict) -> Tuple[Optional[float], Optional[float]]:
    metrics = data.get("metrics") or {}
    for split in ("test", "holdout"):
        blob = metrics.get(split)
        if not isinstance(blob, dict):
            raise SystemExit(
                f"{path}: expected metrics['{split}'] to be an object with 'roc_auc'."
            )
    test_v = metrics["test"].get("roc_auc")
    hold_v = metrics["holdout"].get("roc_auc")
    return (
        float(test_v) if test_v is not None else None,
        float(hold_v) if hold_v is not None else None,
    )


def _collect_table_metrics(
    hyenadna_dir: Path,
) -> Dict[Tuple[int, int], Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """
    Map (cache_max_length, model_max_length) ->
      {task: (test_auc, holdout_auc)} for task in TASKS.
    """
    candidates = _load_candidate_paths(hyenadna_dir)
    if not candidates:
        raise SystemExit(f"No JSON files found under: {hyenadna_dir}")

    grouped: Dict[Tuple[int, int, str], List[Path]] = {}
    for p in candidates:
        data = _read_result_payload(p)
        if data is None:
            continue
        cache_len, model_len, task = _extract_lengths_and_task(p, data)
        key = (cache_len, model_len, task)
        grouped.setdefault(key, []).append(p)

    out: Dict[Tuple[int, int], Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    missing: List[str] = []
    for cache_len, model_len in ROW_ORDER:
        row_key = (cache_len, model_len)
        task_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for task in TASKS:
            key = (cache_len, model_len, task)
            paths = grouped.get(key, [])
            if not paths:
                missing.append(f"{_fmt_len(cache_len)}-{_fmt_len(model_len)}:{task}")
                continue
            if len(paths) > 1:
                listed = ", ".join(str(p) for p in paths)
                raise SystemExit(
                    "Multiple JSON files match one Table 4 cell; "
                    "keep one file per (cache max_length, model max_length, task): "
                    f"{key} -> [{listed}]"
                )
            data = _read_result_payload(paths[0])
            if data is None:
                raise SystemExit(f"{paths[0]}: expected HyenaDNA metrics JSON object.")
            task_map[task] = _extract_auc_pair(paths[0], data)
        out[row_key] = task_map

    if missing:
        miss = "\n".join(f"  - {m}" for m in missing)
        raise SystemExit(
            "Missing HyenaDNA results for Table 4 combinations:\n"
            f"{miss}\n"
            f"Searched under: {hyenadna_dir}"
        )

    return out


def format_table_html(
    rows: Dict[Tuple[int, int], Dict[str, Tuple[Optional[float], Optional[float]]]],
    *,
    decimals: int,
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        '<th colspan="2">max_length</th>\n'
        '<th colspan="2">Cancer diagnosis AUC</th>\n'
        '<th colspan="2">Cancer type AUC</th>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>cache</th><th>model</th>"
        "<th>Test</th><th>Holdout</th>"
        "<th>Test</th><th>Holdout</th>\n"
        "</tr>\n"
        "</thead>\n"
    )

    body_rows: List[str] = []
    for cache_len, model_len in ROW_ORDER:
        vals: List[str] = [_fmt_len(cache_len), _fmt_len(model_len)]
        for task in TASKS:
            test_v, hold_v = rows[(cache_len, model_len)][task]
            vals.append(_fmt_cell(test_v, decimals=decimals))
            vals.append(_fmt_cell(hold_v, decimals=decimals))
        tds = "".join(f"<td>{html.escape(v)}</td>" for v in vals)
        body_rows.append(f"<tr>\n{tds}\n</tr>")
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hyenadna-dir",
        type=Path,
        default=root / "results" / "hyenadna",
        help="Directory with per-cache subdirectories of JSON files (default: results/hyenadna).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for AUC values (default: 3).",
    )
    args = p.parse_args()
    hyenadna_dir = args.hyenadna_dir.expanduser()
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    rows = _collect_table_metrics(hyenadna_dir)
    print(format_table_html(rows, decimals=args.decimals), end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
