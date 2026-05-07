#!/usr/bin/env python3
"""
Fine-tune HyenaDNA with the classification head (hyenadna/standalone_hyenadna.py).

Uses a pre-built feature-only per-run tensor cache from scripts/build_run_tensors.py,
joins labels/splits from shared metadata at runtime, and reports run-level ROC-AUC on
the test and holdout splits (one score per Run).

Config: defaults.yaml (train_hyenadna + paths) with optional experiments.yaml (--expt).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hyenadna import HyenaDNAPreTrainedModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from shared_utilities import (
    binary_roc_auc_from_scores,
    build_run_task_table,
)
from hyenadna_fasta_data import (
    merge_train_hyenadna_config,
    model_max_length,
    resolve_repo_path,
)

HEAD_MODES = ("last", "first", "pool", "sum")
AGGREGATES = ("mean", "max")


@dataclass(frozen=True)
class RunRecord:
    run: str
    split: str
    label: int
    task_label: str
    n_sets: int
    file: Path


def _paths_cfg(defaults_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    return paths


def _results_json_out_path(
    repo_root: Path,
    raw: Optional[object],
    *,
    task: str,
) -> Optional[Path]:
    if raw is None:
        return None
    if raw in ("", "null"):
        defaults_path = repo_root / "defaults.yaml"
        try:
            paths_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))["paths"]
            scratch_key = paths_cfg["results_scratch_dir"]
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise SystemExit(
                f"Cannot read paths.results_scratch_dir from {defaults_path}: {exc}"
            ) from exc
        scratch_base = resolve_repo_path(repo_root, scratch_key)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return scratch_base / f"train_hyenadna_{task}_{ts}.json"
    p = Path(str(raw).strip()).expanduser()
    return p if p.is_absolute() else repo_root / p


def _load_build_run_tensors_cfg(defaults_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    section = cfg.get("build_run_tensors")
    if not isinstance(section, dict):
        raise SystemExit(f"{defaults_path} must define build_run_tensors as a mapping.")
    return section


def _build_records(
    run_task_df,
    cache_root: Path,
    *,
    label_encoder: LabelEncoder,
    requested_num_sets: int,
) -> List[RunRecord]:
    rows = (
        run_task_df.loc[:, ["Run", "split", "task_label"]]
        .drop_duplicates(subset=["Run"])
        .reset_index(drop=True)
    )
    out: List[RunRecord] = []
    missing_files: List[str] = []

    for _, row in rows.iterrows():
        run = str(row["Run"]).strip()
        pt_path = cache_root / f"{run}.pt"
        if not pt_path.is_file():
            missing_files.append(run)
            continue
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(pt_path, map_location="cpu")
        n_sets = int(blob.get("n_sets", 0))
        if n_sets <= 0:
            continue
        out.append(
            RunRecord(
                run=run,
                split=str(row["split"]),
                label=int(label_encoder.transform([str(row["task_label"])])[0]),
                task_label=str(row["task_label"]),
                n_sets=min(n_sets, requested_num_sets),
                file=pt_path,
            )
        )

    if missing_files:
        ex = ", ".join(sorted(missing_files)[:5])
        raise SystemExit(
            "Missing run tensor files in torch cache. "
            f"Example runs: {ex}. Run: python scripts/build_run_tensors.py"
        )
    return out


class RunTensorDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[RunRecord],
        *,
        requested_num_sets: int,
        requested_max_len: int,
        cache_num_sets: int,
        cache_max_len: int,
    ):
        self.entries = list(entries)
        self.requested_num_sets = requested_num_sets
        self.requested_max_len = requested_max_len
        self.cache_num_sets = cache_num_sets
        self.cache_max_len = cache_max_len

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        e = self.entries[idx]
        path = e.file
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        n_sets = min(int(blob["n_sets"]), self.requested_num_sets, self.cache_num_sets)
        start = self.cache_max_len - self.requested_max_len
        if start < 0:
            raise SystemExit(
                "train_hyenadna.max_length exceeds build_run_tensors.max_length. "
                "Increase build_run_tensors.max_length and rebuild run tensors."
            )
        input_ids = blob["input_ids"][: self.requested_num_sets, start:self.cache_max_len]
        attention_mask = blob["attention_mask"][: self.requested_num_sets, start:self.cache_max_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_sets": n_sets,
            "label": e.label,
            "run": e.run,
        }


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    n_sets = torch.tensor([b["n_sets"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    runs = [str(b["run"]) for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "n_sets": n_sets,
        "label": labels,
        "run": runs,
    }


def training_loss(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    bsz, n_set, seq_len = input_ids.shape
    logits = model(input_ids.view(bsz * n_set, seq_len))
    logits = logits.view(bsz, n_set, -1)
    nv = batch["n_sets"]
    labels = batch["label"].to(device)
    mask = torch.arange(n_set, device=device).unsqueeze(0) < nv.to(device).unsqueeze(1)
    flat_logits = logits[mask]
    flat_y = labels.unsqueeze(1).expand(bsz, n_set)[mask]
    return F.cross_entropy(flat_logits, flat_y)


def run_level_scores(
    model: torch.nn.Module,
    entries: Sequence[RunRecord],
    device: torch.device,
    *,
    aggregate: str,
    pos_class_index: int,
    requested_num_sets: int,
    requested_max_len: int,
    cache_num_sets: int,
    cache_max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[str] = []
    y_score: List[float] = []
    with torch.no_grad():
        for e in entries:
            path = e.file
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(path, map_location="cpu")
            start = cache_max_len - requested_max_len
            if start < 0:
                raise SystemExit(
                    "train_hyenadna.max_length exceeds build_run_tensors.max_length. "
                    "Increase build_run_tensors.max_length and rebuild run tensors."
                )
            x = blob["input_ids"][:requested_num_sets, start:cache_max_len].to(device)
            nv = min(int(blob["n_sets"]), requested_num_sets, cache_num_sets)
            if nv <= 0:
                continue
            logits = model(x[:nv])
            if aggregate == "mean":
                agg = logits.mean(dim=0)
            elif aggregate == "max":
                agg = logits.max(dim=0)[0]
            else:
                raise SystemExit(f"Unknown run_logits_aggregate: {aggregate!r}")
            prob = torch.softmax(agg, dim=-1)[pos_class_index].item()
            y_true.append(e.task_label)
            y_score.append(float(prob))
    return np.asarray(y_true, dtype=object), np.asarray(y_score, dtype=np.float64)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        loss = training_loss(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help="Optional train_hyenadna experiment index from experiments.yaml (1-based). "
        "Omit or use 0 for defaults.yaml only.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="?",
        const="",
        default=argparse.SUPPRESS,
        help=(
            "Override results JSON path. With no path, writes under results/scratch/ "
            "as train_hyenadna_<task>_<utc>.json. Omit entirely to use YAML only."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.expt is not None and args.expt < 0:
        raise SystemExit("--expt must be >= 0.")
    return args


def _write_results_json(
    path: Path,
    *,
    merged: Mapping[str, Any],
    cache_info: Mapping[str, Any],
    test_auc: float,
    holdout_auc: float,
    last_train_loss: float,
    epochs_ran: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": merged["task"],
        "model": merged["model"],
        "results_json": str(path.resolve()),
        "config": dict(merged),
        "data": {
            "cache": dict(cache_info),
        },
        "metrics": {
            "test": {"roc_auc": float(test_auc) if test_auc == test_auc else None},
            "holdout": {"roc_auc": float(holdout_auc) if holdout_auc == holdout_auc else None},
        },
        "training": {
            "epochs": int(epochs_ran),
            "final_train_loss": float(last_train_loss),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = _parse_argv(argv)
    expt = int(cli.expt) if cli.expt is not None else 0

    repo_root = Path(__file__).resolve().parent.parent

    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"
    merged, _exp_name, _tpl = merge_train_hyenadna_config(
        defaults_path, experiments_path, expt=expt
    )

    if hasattr(cli, "results_json"):
        rj = cli.results_json
        if rj == "":
            merged = {**merged, "results_json": ""}
        else:
            merged = {**merged, "results_json": rj}

    paths_cfg = _paths_cfg(defaults_path)
    run_tensors_root = resolve_repo_path(
        repo_root, str(paths_cfg.get("run_tensors_dir", "outputs/run_tensors")).strip()
    )
    build_cfg = _load_build_run_tensors_cfg(defaults_path)
    cache_num_sets = int(build_cfg["num_sets"])
    cache_max_len = int(build_cfg["max_length"])

    task = str(merged["task"]).strip()
    model_name = str(merged["model"]).strip()
    num_sets = int(merged["num_sets"])
    max_len = model_max_length(model_name, merged.get("max_length"))
    head_mode = str(merged["head_pooling_mode"]).strip()
    aggregate = str(merged["run_logits_aggregate"]).strip().lower()
    if head_mode not in HEAD_MODES:
        raise SystemExit(f"head_pooling_mode must be one of {HEAD_MODES}; got {head_mode!r}.")
    if aggregate not in AGGREGATES:
        raise SystemExit(f"run_logits_aggregate must be one of {AGGREGATES}; got {aggregate!r}.")
    if num_sets > cache_num_sets:
        raise SystemExit(
            f"train_hyenadna.num_sets ({num_sets}) exceeds build_run_tensors.num_sets "
            f"({cache_num_sets}). Rebuild run tensors with a larger cache."
        )
    if max_len > cache_max_len:
        raise SystemExit(
            f"Resolved max_length ({max_len}) exceeds build_run_tensors.max_length "
            f"({cache_max_len}). Rebuild run tensors with a larger cache."
        )
    if not run_tensors_root.is_dir():
        raise SystemExit(
            f"Missing run tensors directory: {run_tensors_root}. "
            "Run: python scripts/build_run_tensors.py"
        )

    run_task_df = build_run_task_table(task, config_path=defaults_path)
    enc = LabelEncoder()
    y_train = run_task_df.loc[run_task_df["split"] == "train", "task_label"].to_numpy(
        dtype=object
    )
    if y_train.size == 0:
        raise SystemExit("No training runs found after shared split assignment.")
    enc.fit(y_train)
    class_names = [str(x) for x in enc.classes_.tolist()]

    pos_class_index = 1
    pos_label = class_names[pos_class_index]
    print(
        f"\nHyenaDNA train | task={task} model={model_name} | "
        f"positive_class={pos_label!r} (index {pos_class_index}) | "
        f"cache={run_tensors_root}",
        flush=True,
    )

    all_records = _build_records(
        run_task_df,
        run_tensors_root,
        label_encoder=enc,
        requested_num_sets=num_sets,
    )
    by_split: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in all_records:
        by_split[r.split].append(r)

    train_entries = by_split["train"]
    test_entries = by_split["test"]
    holdout_entries = by_split["holdout"]
    if not train_entries:
        raise SystemExit("No training runs in cache (split=train). Build dataset first.")
    if not test_entries:
        raise SystemExit("No test runs in cache.")

    seed = int(merged["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_s = str(merged.get("device") or "").strip().lower()
    if not device_s or device_s == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_s)

    train_ds = RunTensorDataset(
        train_entries,
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(merged["batch_size"]),
        shuffle=True,
        num_workers=int(merged["num_workers"]),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    ckpt_dir = resolve_repo_path(repo_root, str(merged["checkpoint_dir"]).strip())
    model = HyenaDNAPreTrainedModel.from_pretrained(
        str(ckpt_dir),
        model_name,
        download=bool(merged["download_pretrained"]),
        config=None,
        device=str(device),
        use_head=True,
        n_classes=len(class_names),
        head_pooling_mode=head_mode,
    )
    model = model.to(device)

    if bool(merged["freeze_backbone"]):
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(merged["learning_rate"]),
        weight_decay=float(merged["weight_decay"]),
    )

    epochs = int(merged["epochs"])
    last_loss = 0.0
    for ep in range(1, epochs + 1):
        print(f"\n--- Epoch {ep}/{epochs} ---", flush=True)
        last_loss = train_epoch(model, train_loader, opt, device)
        test_auc = binary_roc_auc_from_scores(
            *run_level_scores(
                model,
                test_entries,
                device,
                aggregate=aggregate,
                pos_class_index=pos_class_index,
                requested_num_sets=num_sets,
                requested_max_len=max_len,
                cache_num_sets=cache_num_sets,
                cache_max_len=cache_max_len,
            )
        )
        hold_auc = float("nan")
        if holdout_entries:
            hold_auc = binary_roc_auc_from_scores(
                *run_level_scores(
                    model,
                    holdout_entries,
                    device,
                    aggregate=aggregate,
                    pos_class_index=pos_class_index,
                    requested_num_sets=num_sets,
                    requested_max_len=max_len,
                    cache_num_sets=cache_num_sets,
                    cache_max_len=cache_max_len,
                )
            )
        print(
            f"train_loss={last_loss:.6f}  test_roc_auc={test_auc:.6f}  "
            f"holdout_roc_auc={hold_auc:.6f}  (run-level; logits={aggregate})",
            flush=True,
        )

    results_path = _results_json_out_path(
        repo_root,
        merged.get("results_json"),
        task=task,
    )
    if results_path is not None:
        _write_results_json(
            results_path,
            merged=merged,
            cache_info={
                "dir": str(run_tensors_root),
                "build_run_tensors_num_sets": cache_num_sets,
                "build_run_tensors_max_length": cache_max_len,
                "train_num_sets": num_sets,
                "train_max_length": max_len,
                "n_cached_runs": len(all_records),
            },
            test_auc=test_auc,
            holdout_auc=hold_auc,
            last_train_loss=last_loss,
            epochs_ran=epochs,
        )
        print(f"\nWrote results JSON: {results_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
