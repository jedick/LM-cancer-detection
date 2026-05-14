#!/usr/bin/env python3
"""
Build Table 5 (HyenaDNA ablations) as HTML under manuscript/table5_hyenadna.html.

Reads ``experiments.yaml`` ``train_hyenadna.experiments`` (row order preserved)
and aggregates
``metrics.test.roc_auc`` / ``metrics.holdout.roc_auc`` from per-seed files:
``{task_abbrv}_{name}_<L>k_s<seed>.json`` under ``results/hyenadna/``.

Run from the repository root: ``python helpers/table5_hyenadna.py``
"""

from __future__ import annotations

import html
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


# Plain-English row labels keyed by ``train_hyenadna.experiments[].name``.
# Row order follows ``experiments.yaml``. Descriptions may contain HTML (e.g.
# <sup>) and are intentionally not escaped when rendered into the <td>.
ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "best_recipe": "Best recipe (baseline)",
    "no_study_balanced": "Random training sampler (no study-balanced sampling)",
    "no_class_weight": "No class weighting",
    "high_lr": "Higher learning rate (10<sup>\u22124</sup> instead of 10<sup>\u22125</sup>)",
    "hi_adv_weight": "Higher study adversarial weight (0.6 instead of 0.3)",
    "no_dann": "No domain adversarial training",
}
TASK_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("cd", "Cancer diagnosis"),
    ("ct", "Cancer type"),
)

DECIMALS = 3
OUTPUT_REL = Path("manuscript") / "table5_hyenadna.html"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _train_hyenadna_base(repo_root: Path) -> dict:
    cfg = _load_yaml(repo_root / "defaults.yaml")
    sec = cfg.get("train_hyenadna")
    if not isinstance(sec, dict):
        raise SystemExit("defaults.yaml must define train_hyenadna as a mapping.")
    return dict(sec)


def _parse_seed_spec(raw: object) -> Set[int]:
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    txt = str(raw).strip()
    if not txt:
        return set()
    seeds: Set[int] = set()
    for token in txt.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seeds.add(int(token))
        except ValueError as exc:
            raise SystemExit(f"random_seed contains non-integer value {token!r}: {exc}")
    return seeds


def _experiments(repo_root: Path) -> List[Dict[str, object]]:
    """Return metadata for HyenaDNA ablations in experiments.yaml order."""
    base = _train_hyenadna_base(repo_root)
    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_hyenadna") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit(
            "experiments.yaml train_hyenadna.experiments must be a non-empty list."
        )
    out: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"experiments.yaml row {idx + 1} is not a mapping.")
        name = str(row.get("name") or "").strip()
        if name not in ABLATION_DESCRIPTIONS:
            raise SystemExit(
                f"experiments.yaml row {idx + 1}: unknown experiment name {name!r}; "
                "add a description to ABLATION_DESCRIPTIONS in helpers/table5_hyenadna.py."
            )
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit(f"{name}: overrides must be a mapping.")
        merged = {**base, **overrides}
        try:
            max_length = int(merged["max_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{name}: cannot resolve max_length from defaults+overrides ({exc})."
            )
        expected_seeds = _parse_seed_spec(merged.get("random_seed"))
        out.append(
            {
                "name": name,
                "max_length": max_length,
                "expected_seeds": expected_seeds,
            }
        )
    return out


def _len_token(max_length: int) -> str:
    if max_length <= 0 or max_length % 1024 != 0:
        raise SystemExit(
            f"max_length {max_length} is not a positive multiple of 1024; "
            "cannot reconstruct results JSON filename."
        )
    return f"{max_length // 1024}k"


_SEED_RE = re.compile(r"_s(\d+)\.json$")


def _seed_files(
    hyenadna_dir: Path, task_abbrv: str, name: str, max_length: int
) -> List[Tuple[int, Path]]:
    tok = _len_token(max_length)
    candidates = sorted(hyenadna_dir.glob(f"{task_abbrv}_{name}_{tok}_s*.json"))
    out: List[Tuple[int, Path]] = []
    for p in candidates:
        m = _SEED_RE.search(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _read_metrics(path: Path) -> Tuple[int, float, float]:
    """Return (best_epoch, test_roc_auc, holdout_roc_auc) from one JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")

    tuning = data.get("tuning") or {}
    best_epoch = tuning.get("best_epoch")
    if not isinstance(best_epoch, int):
        raise SystemExit(f"{path}: missing or non-integer tuning.best_epoch.")

    metrics = data.get("metrics") or {}
    for split in ("test", "holdout"):
        blob = metrics.get(split)
        if not isinstance(blob, dict):
            raise SystemExit(
                f"{path}: expected metrics['{split}'] to be an object with 'roc_auc'."
            )
    test_v = metrics["test"].get("roc_auc")
    hold_v = metrics["holdout"].get("roc_auc")
    if test_v is None or hold_v is None:
        raise SystemExit(f"{path}: missing test or holdout roc_auc.")
    t = float(test_v)
    h = float(hold_v)
    if not math.isfinite(t) or not math.isfinite(h):
        raise SystemExit(f"{path}: non-finite test/holdout roc_auc ({t}, {h}).")
    return int(best_epoch), t, h


def _mean_std(values: List[float]) -> Tuple[float, Optional[float]]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def _fmt_mean_std(values: List[float], *, decimals: int) -> str:
    if not values:
        return "\u2014"  # em dash
    mean, std = _mean_std(values)
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


def _fmt_median_epoch(values: List[int]) -> str:
    if not values:
        return "\u2014"  # em dash
    med = statistics.median(values)
    if float(med).is_integer():
        return str(int(med))
    return f"{med:.1f}"


def collect_rows(
    hyenadna_dir: Path,
    experiments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for exp in experiments:
        name = str(exp["name"])
        max_length = int(exp["max_length"])
        expected_seeds = set(exp["expected_seeds"])
        task_metrics: Dict[str, Dict[str, List[float]]] = {}
        for task_abbrv, _task_heading in TASK_COLUMNS:
            files = _seed_files(hyenadna_dir, task_abbrv, name, max_length)
            if not files:
                missing.append(f"{task_abbrv}_{name} ({_len_token(max_length)})")
                continue
            found_seeds = {seed for seed, _path in files}
            if expected_seeds and found_seeds != expected_seeds:
                raise SystemExit(
                    f"{task_abbrv}_{name}: expected seeds {sorted(expected_seeds)} "
                    f"from experiments.yaml, found {sorted(found_seeds)}."
                )
            best_epochs: List[int] = []
            test_aucs: List[float] = []
            hold_aucs: List[float] = []
            for _seed, path in files:
                ep, t, h = _read_metrics(path)
                best_epochs.append(ep)
                test_aucs.append(t)
                hold_aucs.append(h)
            task_metrics[task_abbrv] = {
                "best_epochs": best_epochs,
                "test_aucs": test_aucs,
                "hold_aucs": hold_aucs,
            }
        rows.append(
            {
                "name": name,
                "description": ABLATION_DESCRIPTIONS[name],
                "task_metrics": task_metrics,
            }
        )
    if missing:
        miss = "\n".join(f"  - {m}" for m in missing)
        raise SystemExit(
            "Missing HyenaDNA results for ablations:\n"
            f"{miss}\n"
            f"Searched under: {hyenadna_dir}"
        )
    return rows


def format_table_html(
    rows: List[Dict[str, object]],
    *,
    decimals: int,
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        "<th rowspan='2'>Ablation</th>"
        "<th colspan='3'>Cancer diagnosis</th>"
        "<th colspan='3'>Cancer type</th>\n"
        "</tr>\n"
        "<tr>\n"
        "<th>Median best epoch</th>"
        "<th>Test AUC</th>"
        "<th>Holdout AUC</th>"
        "<th>Median best epoch</th>"
        "<th>Test AUC</th>"
        "<th>Holdout AUC</th>\n"
        "</tr>\n"
        "</thead>\n"
    )
    body_rows: List[str] = []
    for row in rows:
        desc = row["description"]  # raw HTML allowed (e.g. <sup>)
        task_metrics = row["task_metrics"]
        cd_med_ep = html.escape(_fmt_median_epoch(task_metrics["cd"]["best_epochs"]))
        cd_test = html.escape(
            _fmt_mean_std(task_metrics["cd"]["test_aucs"], decimals=decimals)
        )
        cd_hold = html.escape(
            _fmt_mean_std(task_metrics["cd"]["hold_aucs"], decimals=decimals)
        )
        ct_med_ep = html.escape(_fmt_median_epoch(task_metrics["ct"]["best_epochs"]))
        ct_test = html.escape(
            _fmt_mean_std(task_metrics["ct"]["test_aucs"], decimals=decimals)
        )
        ct_hold = html.escape(
            _fmt_mean_std(task_metrics["ct"]["hold_aucs"], decimals=decimals)
        )
        body_rows.append(
            f"<tr>\n<td>{desc}</td>"
            f"<td>{cd_med_ep}</td>"
            f"<td>{cd_test}</td>"
            f"<td>{cd_hold}</td>"
            f"<td>{ct_med_ep}</td>"
            f"<td>{ct_test}</td>"
            f"<td>{ct_hold}</td>\n</tr>"
        )
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hyenadna_dir = repo_root / "results" / "hyenadna"
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    experiments = _experiments(repo_root)
    rows = collect_rows(hyenadna_dir, experiments)
    text = format_table_html(rows, decimals=DECIMALS)
    out_path = repo_root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
