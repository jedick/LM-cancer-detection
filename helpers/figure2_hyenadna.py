#!/usr/bin/env python3
"""
Build Figure 2: HyenaDNA AUC vs sequence length per set (test vs holdout).

Layout:
- 2x2 subplots
- rows = task (cancer diagnosis, cancer type)
- columns = num_sets (5 sets, 10 sets)
- x-axis = length per set (1k, 2k, 4k, 8k, 16k) from experiment max_length
- y-axis = ROC AUC

Each subplot draws:
- test split: thin dashed line with markers
- holdout split: solid bold line with markers

JSON paths follow experiments.yaml train_hyenadna.results_json_template
(default results/hyenadna/{name}.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


TASKS: List[Tuple[str, str]] = [
    ("cancer_diagnosis", "Cancer diagnosis"),
    ("cancer_type", "Cancer type"),
]
SET_COLUMNS: List[Tuple[int, str]] = [
    (5, "5 sets"),
    (10, "10 sets"),
]

LENGTHS_BP: Tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)
X_LABELS: Tuple[str, ...] = ("1k", "2k", "4k", "8k", "16k")


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _train_hyenadna_base(repo_root: Path) -> dict:
    cfg = _load_yaml(repo_root / "defaults.yaml")
    sec = cfg.get("train_hyenadna")
    if not isinstance(sec, dict):
        raise SystemExit("defaults.yaml must define train_hyenadna as a mapping.")
    return dict(sec)


def _experiment_grid(repo_root: Path) -> Dict[Tuple[str, int, int], str]:
    """Map (task, num_sets, max_length) -> results JSON stem (experiment name)."""
    base = _train_hyenadna_base(repo_root)
    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_hyenadna") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("experiments.yaml must list train_hyenadna.experiments.")

    out: Dict[Tuple[str, int, int], str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("train_hyenadna experiment entry must be a mapping.")
        name = str(row.get("name") or "").strip()
        if not name:
            raise SystemExit("Each train_hyenadna experiment must have a non-empty name.")
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit("experiment overrides must be a mapping.")
        merged = {**base, **overrides}

        task = merged.get("task")
        if task not in ("cancer_diagnosis", "cancer_type"):
            raise SystemExit(f"{name}: expected task cancer_diagnosis or cancer_type, got {task!r}")

        num_sets = int(merged["num_sets"])
        max_length = int(merged["max_length"])

        key = (str(task), num_sets, max_length)
        if key in out:
            raise SystemExit(
                f"Duplicate experiment for task={task!r}, num_sets={num_sets}, "
                f"max_length={max_length}: {out[key]!r} and {name!r}."
            )
        out[key] = name

    return out


def _grid_complete(grid: Dict[Tuple[str, int, int], str]) -> None:
    for task, _ in TASKS:
        for num_sets, _ in SET_COLUMNS:
            for L in LENGTHS_BP:
                key = (task, num_sets, L)
                if key not in grid:
                    raise SystemExit(
                        f"Missing train_hyenadna experiment for "
                        f"task={task!r}, num_sets={num_sets}, max_length={L}."
                    )


def _load_metrics(json_path: Path) -> Tuple[float, float]:
    if not json_path.is_file():
        raise SystemExit(f"Missing expected JSON: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    test = metrics.get("test") or {}
    holdout = metrics.get("holdout") or {}
    t_auc = test.get("roc_auc")
    h_auc = holdout.get("roc_auc")
    if t_auc is None or h_auc is None:
        raise SystemExit(
            f"{json_path}: need finite metrics.test.roc_auc and metrics.holdout.roc_auc "
            f"(got {t_auc!r}, {h_auc!r})."
        )
    return float(t_auc), float(h_auc)


def collect_series(
    hyenadna_dir: Path,
    grid: Dict[Tuple[str, int, int], str],
) -> Dict[Tuple[str, int], Tuple[List[float], List[float]]]:
    out: Dict[Tuple[str, int], Tuple[List[float], List[float]]] = {}
    for task, _ in TASKS:
        for num_sets, _ in SET_COLUMNS:
            test_vals: List[float] = []
            holdout_vals: List[float] = []
            for L in LENGTHS_BP:
                name = grid[(task, num_sets, L)]
                json_path = hyenadna_dir / f"{name}.json"
                test_auc, holdout_auc = _load_metrics(json_path)
                test_vals.append(test_auc)
                holdout_vals.append(holdout_auc)
            out[(task, num_sets)] = (test_vals, holdout_vals)
    return out


def build_plot(
    series: Dict[Tuple[str, int], Tuple[List[float], List[float]]],
    output_path: Path,
) -> None:
    x = list(range(len(LENGTHS_BP)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for row, (task_key, task_label) in enumerate(TASKS):
        for col, (num_sets, col_title) in enumerate(SET_COLUMNS):
            ax = axes[row, col]
            test_vals, holdout_vals = series[(task_key, num_sets)]

            ax.plot(
                x,
                test_vals,
                linestyle="--",
                linewidth=1.2,
                marker="o",
                markersize=4,
                label="Test",
                color="#4C78A8",
            )
            ax.plot(
                x,
                holdout_vals,
                linestyle="-",
                linewidth=2.8,
                marker="o",
                markersize=4.5,
                label="Holdout",
                color="#F58518",
            )

            if row == 0:
                ax.set_title(col_title)
            if col == 0:
                ax.set_ylabel(f"{task_label}\nROC AUC")

            ax.set_xticks(x)
            ax.set_xticklabels(X_LABELS)
            ax.set_ylim(0.5, 1.02)
            ax.grid(alpha=0.25, linewidth=0.7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Length per set")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hyenadna-dir",
        type=Path,
        default=repo_root / "results" / "hyenadna",
        help="Directory with HyenaDNA result JSON files (default: results/hyenadna).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "figures" / "figure2_hyenadna.png",
        help="Output PNG path (default: figures/figure2_hyenadna.png).",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=repo_root / "figures" / "figure2_hyenadna.svg",
        help="Output SVG path (default: figures/figure2_hyenadna.svg).",
    )
    args = parser.parse_args()

    hyenadna_dir = args.hyenadna_dir.expanduser().resolve()
    output_path = args.output.expanduser()
    svg_output_path = args.svg_output.expanduser()
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    grid = _experiment_grid(repo_root)
    _grid_complete(grid)

    series = collect_series(hyenadna_dir, grid)
    build_plot(series, output_path=output_path)
    build_plot(series, output_path=svg_output_path)
    print(output_path)
    print(svg_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
