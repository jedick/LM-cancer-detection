#!/usr/bin/env python3
"""
Build Table 1 from tetramer classifier JSON metrics under results/tetramer/.

Expects six files named {task}_{model}.json (e.g. cancer_diagnosis_knn.json),
as written by scripts/fit_classifier.py: tasks cancer_diagnosis and
cancer_type; models baseline, knn, and random_forest. Each file must have
``metrics.test.roc_auc`` and ``metrics.holdout.roc_auc``.

By default prints an HTML table with nested headers (task × test/holdout).
Use --markdown for a GitHub-flavored pipe table with a two-line header.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

TASKS = ("cancer_diagnosis", "cancer_type")
MODELS = ("baseline", "knn", "random_forest")

ROW_LABELS: Dict[str, str] = {
    "baseline": "Majority class",
    "knn": "KNN",
    "random_forest": "Random Forest",
}

TASK_HEADER = {
    "cancer_diagnosis": "Cancer diagnosis AUC",
    "cancer_type": "Cancer type AUC",
}


def _load_metrics(
    tetramer_dir: Path,
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Map (task, model) -> (test_roc_auc, holdout_roc_auc). None if JSON null."""
    out: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    for task in TASKS:
        for model in MODELS:
            path = tetramer_dir / f"{task}_{model}.json"
            if not path.is_file():
                raise SystemExit(
                    f"Missing expected JSON: {path}\n"
                    f"Required files: {{task}}_{{model}}.json for "
                    f"task in {TASKS}, model in {MODELS}."
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            file_task = data.get("task")
            file_model = data.get("model")
            if file_task != task:
                raise SystemExit(f"{path}: expected task {task!r}, got {file_task!r}")
            if file_model != model:
                raise SystemExit(f"{path}: expected model {model!r}, got {file_model!r}")
            metrics = data.get("metrics") or {}
            for split in ("test", "holdout"):
                blob = metrics.get(split)
                if not isinstance(blob, dict):
                    raise SystemExit(
                        f"{path}: expected metrics['{split}'] to be an object with "
                        f"'roc_auc' (scripts/fit_classifier.py output layout)."
                    )
            test_v = metrics["test"].get("roc_auc")
            hold_v = metrics["holdout"].get("roc_auc")
            out[(task, model)] = (
                float(test_v) if test_v is not None else None,
                float(hold_v) if hold_v is not None else None,
            )
    return out


def _fmt_cell(value: Optional[float], *, decimals: int) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def format_table_html(
    metrics: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
    *,
    decimals: int,
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        '<th rowspan="2">Model</th>\n'
        f'<th colspan="2">{html.escape(TASK_HEADER["cancer_diagnosis"])}</th>\n'
        f'<th colspan="2">{html.escape(TASK_HEADER["cancer_type"])}</th>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>Test</th><th>Holdout</th>"
        "<th>Test</th><th>Holdout</th>\n"
        "</tr>\n"
        "</thead>\n"
    )
    body_rows = []
    for model in MODELS:
        label = html.escape(ROW_LABELS[model])
        cells = []
        for task in TASKS:
            test_v, hold_v = metrics[(task, model)]
            cells.append(_fmt_cell(test_v, decimals=decimals))
            cells.append(_fmt_cell(hold_v, decimals=decimals))
        tds = "".join(f"<td>{html.escape(c)}</td>" for c in cells)
        body_rows.append(f"<tr>\n<td>{label}</td>{tds}\n</tr>")
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def format_table_markdown(
    metrics: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
    *,
    decimals: int,
) -> str:
    """Pipe table: duplicate task labels on row 1 (no colspan in GFM)."""
    d1 = TASK_HEADER["cancer_diagnosis"]
    d2 = TASK_HEADER["cancer_type"]
    row1 = f"| | {d1} | {d1} | {d2} | {d2} |"
    row2 = "| Model | Test | Holdout | Test | Holdout |"
    sep = "| :--- | ---: | ---: | ---: | ---: |"
    lines = [row1, row2, sep]
    for model in MODELS:
        label = ROW_LABELS[model]
        vals = []
        for task in TASKS:
            test_v, hold_v = metrics[(task, model)]
            vals.append(_fmt_cell(test_v, decimals=decimals))
            vals.append(_fmt_cell(hold_v, decimals=decimals))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tetramer-dir",
        type=Path,
        default=root / "results" / "tetramer",
        help="Directory with {task}_{model}.json files (default: results/tetramer).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for AUC values (default: 3).",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a pipe Markdown table instead of HTML.",
    )
    args = p.parse_args()
    tetramer_dir: Path = args.tetramer_dir.expanduser()
    if not tetramer_dir.is_dir():
        raise SystemExit(f"Not a directory: {tetramer_dir}")

    metrics = _load_metrics(tetramer_dir)
    if args.markdown:
        text = format_table_markdown(metrics, decimals=args.decimals)
    else:
        text = format_table_html(metrics, decimals=args.decimals)
    print(text, end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
