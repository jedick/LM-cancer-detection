#!/usr/bin/env python3
"""
Build Table 2 (UC/CAP classifiers) from JSON metrics under results/uc_cap/<feat>/.

Reads experiments.yaml ``run_uc_cap_pipeline`` rows merged over defaults.yaml
( same ordering as ``helpers/list_uc_cap_feature_outputs.py`` and ``FEAT=1..N`` ),
and for each feature index loads:

  cancer_diagnosis_{knn,svm,random_forest}.json
  cancer_type_{knn,svm,random_forest}.json

using ``metrics.test.roc_auc`` only. Prints an HTML table to stdout (like
``helpers/table1_from_classifier.py``).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from list_uc_cap_feature_outputs import merge_run_uc_cap_baseline

TASKS = ("cancer_diagnosis", "cancer_type")
MODELS = ("knn", "svm", "random_forest")
MODEL_HEADER = {"knn": "KNN", "svm": "SVM", "random_forest": "RF"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_feature_parameter_rows(repo_root: Path) -> List[Tuple[int, int, Union[int, str]]]:
    """One (n_uc, n_clusters, n_cap_display) per ``FEAT`` index (1-based order in experiments.yaml)."""
    defaults_cfg = _load_yaml(repo_root / "defaults.yaml")
    experiments_cfg = _load_yaml(repo_root / "experiments.yaml")
    base = merge_run_uc_cap_baseline(defaults_cfg)
    rows = experiments_cfg.get("run_uc_cap_pipeline") or []
    out: List[Tuple[int, int, Union[int, str]]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("experiments.yaml run_uc_cap_pipeline entries must be mappings")
        merged: Dict[str, Any] = {**base, **row}
        n_cap = merged["n_cap"]
        if isinstance(n_cap, str) and str(n_cap).strip().lower() == "all":
            cap_disp: Union[int, str] = "all"
        else:
            cap_disp = int(n_cap)
        out.append((int(merged["n_uc"]), int(merged["n_clusters"]), cap_disp))
    return out


def _fmt_cell(value: Optional[float], *, decimals: int) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def _fmt_param_cell(value: Union[int, str]) -> str:
    if isinstance(value, str):
        return html.escape(value, quote=True)
    return str(int(value))


def _validate_and_collect(
    uc_cap_dir: Path,
    param_rows: List[Tuple[int, int, Union[int, str]]],
) -> List[Tuple[Tuple[int, int, Union[int, str]], Dict[Tuple[str, str], Optional[float]]]]:
    """Load metrics; validate task/model once per file."""
    grid: List[
        Tuple[Tuple[int, int, Union[int, str]], Dict[Tuple[str, str], Optional[float]]]
    ] = []
    for feat_idx, params in enumerate(param_rows, start=1):
        sub = uc_cap_dir / str(feat_idx)
        scores: Dict[Tuple[str, str], Optional[float]] = {}
        for task in TASKS:
            for model in MODELS:
                path = sub / f"{task}_{model}.json"
                if not path.is_file():
                    raise SystemExit(
                        f"Missing expected JSON: {path}\n"
                        f"Required under {sub}/: {{task}}_{{model}}.json for "
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
                test = metrics.get("test")
                if not isinstance(test, dict):
                    raise SystemExit(
                        f"{path}: expected metrics['test'] to be an object with 'roc_auc'."
                    )
                v = test.get("roc_auc")
                auc: Optional[float] = None if v is None else float(v)
                scores[(task, model)] = auc
        grid.append((params, scores))
    return grid


def format_table_html(
    rows: List[Tuple[Tuple[int, int, Union[int, str]], Dict[Tuple[str, str], Optional[float]]]],
    *,
    decimals: int,
) -> str:
    """Nested headers: parameter columns + cancer diagnosis (3 models) + cancer type (3 models)."""
    n_uc_h = "<i>n</i><sub>UC</sub>"
    k_h = "<i>K</i>"
    n_cap_h = "<i>n</i><sub>CAP</sub>"
    thead = (
        "<thead>\n"
        "<tr>\n"
        f'<th rowspan="2">{n_uc_h}</th>\n'
        f'<th rowspan="2">{k_h}</th>\n'
        f'<th rowspan="2">{n_cap_h}</th>\n'
        '<th colspan="3">Cancer diagnosis AUC (test)</th>\n'
        '<th colspan="3">Cancer type AUC (test)</th>\n'
        "</tr>\n"
        "<tr>\n"
        + "".join(f"<th>{html.escape(MODEL_HEADER[m])}</th>\n" for m in MODELS)
        + "".join(f"<th>{html.escape(MODEL_HEADER[m])}</th>\n" for m in MODELS)
        + "</tr>\n"
        "</thead>\n"
    )
    body_lines: List[str] = []
    for params, scores in rows:
        n_uc, n_clusters, n_cap = params
        cells = [
            _fmt_param_cell(n_uc),
            _fmt_param_cell(n_clusters),
            _fmt_param_cell(n_cap),
        ]
        for task in TASKS:
            for model in MODELS:
                cells.append(_fmt_cell(scores[(task, model)], decimals=decimals))
        tds = "".join(f"<td>{c}</td>" for c in cells)
        body_lines.append(f"<tr>\n{tds}\n</tr>")
    tbody = "<tbody>\n" + "\n".join(body_lines) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--uc-cap-dir",
        type=Path,
        default=root / "results" / "uc_cap",
        help="Parent directory with 1..N subdirs of JSON (default: results/uc_cap).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for AUC values (default: 3, same as Table 1).",
    )
    args = p.parse_args()
    uc_cap_dir = args.uc_cap_dir.expanduser()
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    param_rows = load_feature_parameter_rows(root)
    if not param_rows:
        raise SystemExit("No run_uc_cap_pipeline rows in experiments.yaml.")

    rows = _validate_and_collect(uc_cap_dir, param_rows)
    print(format_table_html(rows, decimals=args.decimals), end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
