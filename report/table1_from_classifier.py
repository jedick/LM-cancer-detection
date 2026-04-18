#!/usr/bin/env python3
"""
Read classifier stdout log and print a Markdown table (majority-class + KNN, macro/micro AUC).

Default input: outputs/classifier_results.txt (repo root relative to this file's parent).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^\s*(KNN|Majority class):\s*macro-averaged ROC AUC = ([0-9.]+|nan),\s*"
    r"micro-averaged ROC AUC = ([0-9.]+|nan)\s*$"
)


def parse_aucs(text: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def format_table(aucs: dict[str, tuple[str, str]], *, decimals: int) -> str:
    def fmt_pair(mac: str, mic: str) -> tuple[str, str]:
        def one(x: str) -> str:
            if x.lower() == "nan":
                return "nan"
            return f"{float(x):.{decimals}f}"

        return one(mac), one(mic)

    need = ("Majority class", "KNN")
    missing = [k for k in need if k not in aucs]
    if missing:
        raise SystemExit(
            f"Missing rows in input ({', '.join(missing)}). "
            "Expected lines like '  KNN: macro-averaged ROC AUC = …'."
        )

    rows = []
    for label in need:
        mac, mic = fmt_pair(*aucs[label])
        rows.append(f"| {label} | {mac} | {mic} |")

    header = "| Model | Macro AUC | Micro AUC |"
    sep = "| :--- | ---: | ---: |"
    return "\n".join([header, sep] + rows)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=root / "outputs" / "classifier_results.txt",
        help="Path to classifier log (default: outputs/classifier_results.txt).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for AUC values (default: 3).",
    )
    args = p.parse_args()
    path: Path = args.input
    if not path.is_file():
        raise SystemExit(f"Input not found: {path}")

    text = path.read_text(encoding="utf-8")
    aucs = parse_aucs(text)
    print(format_table(aucs, decimals=args.decimals), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
