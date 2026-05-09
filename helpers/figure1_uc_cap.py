#!/usr/bin/env python3
"""
Build Figure 1: UC/CAP feature-set stability across test vs holdout AUC.

Layout:
- 2x2 subplots
- rows = task (cancer diagnosis, cancer type)
- columns = model (SVM, Random Forest)
- x-axis = UC/CAP feature set index (1..N from experiments.yaml run_uc_cap_pipeline)
- y-axis = ROC AUC

Each subplot draws:
- test split: thin dashed line with markers
- holdout split: solid bold line with markers
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
MODELS: List[Tuple[str, str]] = [
    ("svm", "SVM"),
    ("random_forest", "Random Forest"),
]


def _load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _feature_count(repo_root: Path) -> int:
    cfg = _load_yaml(repo_root / "experiments.yaml")
    rows = cfg.get("run_uc_cap_pipeline") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No run_uc_cap_pipeline rows found in experiments.yaml.")
    return len(rows)


def _load_metrics(json_path: Path) -> Tuple[float, float]:
    if not json_path.is_file():
        raise SystemExit(f"Missing expected JSON: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    test = metrics.get("test") or {}
    holdout = metrics.get("holdout") or {}
    if "roc_auc" not in test or "roc_auc" not in holdout:
        raise SystemExit(
            f"{json_path}: expected metrics.test.roc_auc and metrics.holdout.roc_auc."
        )
    return float(test["roc_auc"]), float(holdout["roc_auc"])


def collect_series(
    uc_cap_dir: Path, n_features: int
) -> Dict[Tuple[str, str], Tuple[List[float], List[float]]]:
    out: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    for task, _ in TASKS:
        for model, _ in MODELS:
            test_vals: List[float] = []
            holdout_vals: List[float] = []
            for feat_idx in range(1, n_features + 1):
                json_path = uc_cap_dir / str(feat_idx) / f"{task}_{model}.json"
                test_auc, holdout_auc = _load_metrics(json_path)
                test_vals.append(test_auc)
                holdout_vals.append(holdout_auc)
            out[(task, model)] = (test_vals, holdout_vals)
    return out


def build_plot(
    series: Dict[Tuple[str, str], Tuple[List[float], List[float]]],
    n_features: int,
    output_path: Path,
) -> None:
    x = list(range(1, n_features + 1))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for row, (task_key, task_label) in enumerate(TASKS):
        for col, (model_key, model_label) in enumerate(MODELS):
            ax = axes[row, col]
            test_vals, holdout_vals = series[(task_key, model_key)]

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
                ax.set_title(model_label)
            if col == 0:
                ax.set_ylabel(f"{task_label}\nROC AUC")

            ax.set_xticks(x)
            ax.set_ylim(0.5, 1.02)
            ax.grid(alpha=0.25, linewidth=0.7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Feature set index")

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
        "--uc-cap-dir",
        type=Path,
        default=repo_root / "results" / "uc_cap",
        help="Directory containing FEAT-indexed UC/CAP result subdirectories (default: results/uc_cap).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "figures" / "figure1_uc_cap.png",
        help="Output figure path (default: figures/figure1_uc_cap.png).",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=repo_root / "figures" / "figure1_uc_cap.svg",
        help="Optional SVG output path (default: figures/figure1_uc_cap.svg).",
    )
    args = parser.parse_args()

    uc_cap_dir = args.uc_cap_dir.expanduser()
    output_path = args.output.expanduser()
    svg_output_path = args.svg_output.expanduser()
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    n_features = _feature_count(repo_root)
    series = collect_series(uc_cap_dir, n_features=n_features)
    build_plot(series, n_features=n_features, output_path=output_path)
    build_plot(series, n_features=n_features, output_path=svg_output_path)
    print(output_path)
    print(svg_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
