#!/usr/bin/env python3
"""
Read binary-task classifier logs and print a Markdown table (majority-class + KNN AUC).

Default inputs (repo-root relative): outputs/cancer_diagnosis_results.txt and
outputs/cancer_type_results.txt.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW_RE = re.compile(r"^\s*(KNN|Majority class):\s*ROC AUC = ([0-9.]+|nan)\s*$")


def parse_aucs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def format_table(
    diagnosis_aucs: dict[str, str],
    type_aucs: dict[str, str],
    *,
    decimals: int,
) -> str:
    def fmt_one(x: str) -> str:
        if x.lower() == "nan":
            return "nan"
        return f"{float(x):.{decimals}f}"

    need = ("Majority class", "KNN")
    missing_diagnosis = [k for k in need if k not in diagnosis_aucs]
    missing_type = [k for k in need if k not in type_aucs]
    if missing_diagnosis or missing_type:
        parts = []
        if missing_diagnosis:
            parts.append(
                "cancer_diagnosis missing "
                + ", ".join(missing_diagnosis)
            )
        if missing_type:
            parts.append("cancer_type missing " + ", ".join(missing_type))
        raise SystemExit(
            "Missing rows in input logs: "
            + "; ".join(parts)
            + ". Expected lines like '  KNN: ROC AUC = ...'."
        )

    rows = []
    for label in need:
        diagnosis_auc = fmt_one(diagnosis_aucs[label])
        type_auc = fmt_one(type_aucs[label])
        rows.append(f"| {label} | {diagnosis_auc} | {type_auc} |")

    header = "| Model | Cancer diagnosis AUC | Cancer type AUC |"
    sep = "| :--- | ---: | ---: |"
    return "\n".join([header, sep] + rows)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--diagnosis-input",
        type=Path,
        default=root / "outputs" / "cancer_diagnosis_results.txt",
        help="Path to cancer_diagnosis classifier log (default: outputs/cancer_diagnosis_results.txt).",
    )
    p.add_argument(
        "--type-input",
        type=Path,
        default=root / "outputs" / "cancer_type_results.txt",
        help="Path to cancer_type classifier log (default: outputs/cancer_type_results.txt).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for AUC values (default: 3).",
    )
    args = p.parse_args()
    diagnosis_path: Path = args.diagnosis_input
    type_path: Path = args.type_input
    if not diagnosis_path.is_file():
        raise SystemExit(f"Input not found: {diagnosis_path}")
    if not type_path.is_file():
        raise SystemExit(f"Input not found: {type_path}")

    diagnosis_text = diagnosis_path.read_text(encoding="utf-8")
    type_text = type_path.read_text(encoding="utf-8")
    diagnosis_aucs = parse_aucs(diagnosis_text)
    type_aucs = parse_aucs(type_text)
    print(
        format_table(diagnosis_aucs, type_aucs, decimals=args.decimals),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
