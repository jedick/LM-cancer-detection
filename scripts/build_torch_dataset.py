#!/usr/bin/env python3
"""
Build a disk cache of tokenized HyenaDNA tensors per sequencing run (FASTA → .pt).

Reads run-level labels/splits from study CSV metadata via shared utilities (not from
tetramer_frequencies.csv). Skips runs with missing FASTA or no tokenizable sequence
content. Reuse the cache with scripts/train_hyenadna.py.

Config: defaults.yaml (train_hyenadna + paths) with optional experiments.yaml (--expt).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
from hyenadna_fasta_data import (  # noqa: E402
    cache_slug,
    fasta_path_for_run,
    iter_fasta_sequences,
    merge_train_hyenadna_config,
    make_character_tokenizer,
    model_max_length,
    resolve_repo_path,
    run_to_tensors,
)
from shared_utilities import require_binary_classes, run_task_table_from_study_csvs

DATASET_SCHEMA = "hyenadna_torch_dataset_v1"
DATASET_VERSION = 1


def _paths_cfg(defaults_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    return paths


def _norm_label_arg(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help="Optional 1-based train_hyenadna experiment index from experiments.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing cache directory for this configuration before rebuilding.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    expt = int(args.expt) if args.expt is not None else 0
    if expt < 0:
        raise SystemExit("--expt must be >= 0 (0 = defaults.yaml only).")

    root = _REPO_ROOT
    defaults_path = root / "defaults.yaml"
    experiments_path = root / "experiments.yaml"
    merged, _name, _tpl = merge_train_hyenadna_config(
        defaults_path, experiments_path, expt=expt
    )

    paths_cfg = _paths_cfg(defaults_path)
    fasta_dir_key = str(paths_cfg["fasta_dir"]).strip()
    torch_root = resolve_repo_path(
        root, str(paths_cfg.get("torch_dataset_dir", "outputs/torch_dataset")).strip()
    )

    task = str(merged["task"]).strip()
    model_name = str(merged["model"]).strip()
    num_sets = int(merged["num_sets"])
    max_length = model_max_length(model_name, merged.get("max_length"))
    label_arg = _norm_label_arg(merged.get("label_column"))

    slug = cache_slug(model_name, task, num_sets, max_length)
    out_dir = torch_root / slug
    meta_path = out_dir / "meta.json"
    runs_dir = out_dir / "runs"

    if meta_path.is_file() and not args.force:
        print(
            f"\nUsing existing torch dataset (use --force to rebuild): {out_dir}",
            flush=True,
        )
        return 0

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)

    runs_dir.mkdir(parents=True, exist_ok=True)

    task_df, label_column = run_task_table_from_study_csvs(
        config_path=defaults_path,
        task=task,
        label_column=label_arg,
    )
    y_train = task_df.loc[task_df["split"] == "train", "task_label"].to_numpy(dtype=object)
    y_val = task_df.loc[task_df["split"] == "val", "task_label"].to_numpy(dtype=object)
    y_test = task_df.loc[task_df["split"] == "test", "task_label"].to_numpy(dtype=object)
    require_binary_classes(y_train, split_name="train split", task=task)
    require_binary_classes(y_val, split_name="validation split", task=task)
    require_binary_classes(y_test, split_name="test split", task=task)

    label_enc = LabelEncoder()
    label_enc.fit(y_train)
    class_names = [str(x) for x in label_enc.classes_.tolist()]

    tokenizer = make_character_tokenizer(max_length)

    runs_frame = task_df[["Run", "study_name", "split", "task_label"]].drop_duplicates(
        subset=["Run"]
    )

    run_records: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    n_total = len(runs_frame)

    print(
        f"\nBuilding HyenaDNA torch dataset ({n_total} runs) → {out_dir}",
        flush=True,
    )

    for _, row in tqdm(
        runs_frame.iterrows(),
        total=n_total,
        desc="FASTA → tensors",
        unit="run",
    ):
        run = str(row["Run"]).strip()
        study_name = str(row["study_name"]).strip()
        split_name = str(row["split"]).strip()
        task_label = str(row["task_label"])

        fasta_gz = fasta_path_for_run(root, fasta_dir_key, study_name, run)
        if not fasta_gz.is_file():
            skipped.append({"run": run, "reason": f"missing_fasta:{fasta_gz}"})
            continue

        sequences = list(iter_fasta_sequences(fasta_gz))
        input_ids, attention_mask, n_valid = run_to_tensors(
            sequences,
            tokenizer=tokenizer,
            max_length=max_length,
            num_sets=num_sets,
        )
        if input_ids is None or n_valid == 0:
            skipped.append({"run": run, "reason": "no_tokenized_sequence_content"})
            continue

        y_int = int(label_enc.transform([task_label])[0])
        rel_file = f"runs/{run}.pt"
        torch.save(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "n_sets": int(n_valid),
            },
            runs_dir / f"{run}.pt",
        )
        run_records.append(
            {
                "run": run,
                "study_name": study_name,
                "split": split_name,
                "label": y_int,
                "task_label": task_label,
                "n_sets": int(n_valid),
                "file": rel_file,
            }
        )

    meta = {
        "schema": DATASET_SCHEMA,
        "version": DATASET_VERSION,
        "model": model_name,
        "max_length": max_length,
        "task": task,
        "num_sets": num_sets,
        "classes": class_names,
        "label_column": label_column,
        "slug": slug,
        "runs": run_records,
        "skipped": skipped,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nWrote {len(run_records)} run tensors; skipped {len(skipped)} "
        f"(see meta.skipped). Meta: {meta_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
