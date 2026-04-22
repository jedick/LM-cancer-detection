#!/usr/bin/env python3
"""Run UC/CAP classifier grid and write markdown result tables.

For each UC/CAP parameter combination from scripts/run_uc_cap_pipeline.sh:
  - n_uc in {1000, 2000}
  - K in {2000, 5000}
  - n_cap in {5000, 10000}

This script evaluates all classifiers:
  - random_forest
  - logistic_regression
  - svm

for both tasks:
  - cancer_diagnosis
  - cancer_type

and writes two markdown tables:
  - results/uc_cap_cancer_diagnosis.md
  - results/uc_cap_cancer_type.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


TASKS: Tuple[str, str] = ("cancer_diagnosis", "cancer_type")
MODELS: Tuple[str, str, str] = ("random_forest", "logistic_regression", "svm")
GRID: Tuple[Tuple[int, int, int], ...] = (
    (1000, 2000, 5000),
    (2000, 2000, 5000),
    (1000, 5000, 5000),
    (2000, 5000, 5000),
    (1000, 2000, 10000),
    (2000, 2000, 10000),
    (1000, 5000, 10000),
    (2000, 5000, 10000),
)

ROC_AUC_PATTERN = re.compile(r"^Test ROC AUC:\s+([0-9]*\.?[0-9]+|nan)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Combo:
    n_uc: int
    n_clusters: int
    n_cap: int

    @property
    def csv_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        return (
            root
            / "outputs"
            / "uc_cap"
            / f"uc{self.n_uc}_k{self.n_clusters}"
            / f"cap{self.n_cap}.csv"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-script",
        type=Path,
        default=root / "scripts" / "fit_uc_cap_classifier.py",
        help="Path to fit_uc_cap_classifier.py.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used to run classifier script.",
    )
    parser.add_argument(
        "--run-metadata-csv",
        type=Path,
        default=root / "outputs" / "tetranucleotide_frequencies.csv",
        help="Run metadata CSV for shared split reconstruction.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results",
        help="Directory where markdown files are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_one(
    *,
    combo: Combo,
    task: str,
    model: str,
    classifier_script: Path,
    python_exe: str,
    run_metadata_csv: Path,
) -> str:
    cmd = [
        python_exe,
        str(classifier_script),
        "--csv",
        str(combo.csv_path),
        "--task",
        task,
        "--classifier",
        model,
        "--run-metadata-csv",
        str(run_metadata_csv),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (
            f"Command failed for task={task}, model={model}, "
            f"n_uc={combo.n_uc}, K={combo.n_clusters}, n_cap={combo.n_cap}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
        raise SystemExit(msg)
    match = ROC_AUC_PATTERN.search(proc.stdout)
    if match is None:
        raise SystemExit(
            "Could not parse Test ROC AUC from classifier output for "
            f"task={task}, model={model}, n_uc={combo.n_uc}, "
            f"K={combo.n_clusters}, n_cap={combo.n_cap}.\nOutput:\n{proc.stdout}"
        )
    return match.group(1)


def markdown_table(task: str, combos: List[Combo], values: Dict[Tuple[Combo, str], str]) -> str:
    lines = [
        f"# UC/CAP {task}",
        "",
        "| n_uc | K | n_cap | random_forest | logistic_regression | svm |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for combo in combos:
        rf = values[(combo, "random_forest")]
        lr = values[(combo, "logistic_regression")]
        svm = values[(combo, "svm")]
        lines.append(
            f"| {combo.n_uc} | {combo.n_clusters} | {combo.n_cap} | {rf} | {lr} | {svm} |"
        )
    lines.append("")
    lines.append("Values are test-set ROC AUC reported by `fit_uc_cap_classifier.py`.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    combos = [Combo(*triple) for triple in GRID]
    for combo in combos:
        if not combo.csv_path.is_file():
            raise SystemExit(f"Missing UC/CAP feature CSV: {combo.csv_path}")
    if not args.classifier_script.is_file():
        raise SystemExit(f"Classifier script not found: {args.classifier_script}")
    if not args.run_metadata_csv.is_file():
        raise SystemExit(f"Run metadata CSV not found: {args.run_metadata_csv}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        scores: Dict[Tuple[Combo, str], str] = {}
        for combo in combos:
            for model in MODELS:
                cmd_preview = (
                    f"{args.python} {args.classifier_script} --csv {combo.csv_path} "
                    f"--task {task} --classifier {model} "
                    f"--run-metadata-csv {args.run_metadata_csv}"
                )
                print(cmd_preview, flush=True)
                if args.dry_run:
                    scores[(combo, model)] = "NA"
                    continue
                score = run_one(
                    combo=combo,
                    task=task,
                    model=model,
                    classifier_script=args.classifier_script,
                    python_exe=args.python,
                    run_metadata_csv=args.run_metadata_csv,
                )
                scores[(combo, model)] = score
        out_path = args.results_dir / f"uc_cap_{task}.md"
        out_path.write_text(markdown_table(task, combos, scores), encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
