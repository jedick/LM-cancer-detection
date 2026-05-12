#!/usr/bin/env python3
"""
Fine-tune HyenaDNA with the classification head (hyenadna/standalone_hyenadna.py).

Uses a pre-built feature-only per-run tensor cache under paths.run_tensors_dir/ from
scripts/build_run_tensors.py, joins labels/splits from shared metadata at runtime, and reports
run-level ROC-AUC on the test and holdout splits (one score per Run).

Training logs (`*_training.json`) record validation ROC-AUC, validation study-macro ROC-AUC,
validation CE loss, a fixed train-subset ROC-AUC, gradient norms (when clipping is enabled),
optional per-study AUC on validation/holdout, and score moments by label. Checkpoints are
selected by ``train_hyenadna.tuning_metric`` (default ``val_roc_auc``).

When ``study_adv`` is true, the model adds a study (domain) classifier with optional
``study_adv_delay_epochs`` (task-only), ``study_discriminator_warmup`` plus
``study_disc_warmup_epochs`` (train study head on detached backbone features), then
DANN-style gradient reversal with ``study_adv_weight`` as both loss scale and GRL lambda.
Logs include train/val study cross-entropy and val study accuracy (train-split study labels
only; unknown studies use ``ignore_index``).

Config: defaults.yaml (train_hyenadna + run_tensors + paths) with optional
experiments.yaml train_hyenadna overrides (--expt).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hyenadna import HyenaDNAPreTrainedModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
HALF_BATCH_SIZE = 0.5
SET_SPLIT_INDEX = 5
STUDY_IGNORE_INDEX = -100


def _resolve_study_adv_phase(
    epoch_1based: int,
    *,
    study_adv_enabled: bool,
    delay_epochs: int,
    disc_warmup_enabled: bool,
    disc_warmup_epochs: int,
) -> str:
    if not study_adv_enabled:
        return "off"
    if epoch_1based <= int(delay_epochs):
        return "delay"
    wu = int(disc_warmup_epochs) if disc_warmup_enabled else 0
    if wu > 0 and epoch_1based <= int(delay_epochs) + wu:
        return "warmup"
    return "dann"


@dataclass(frozen=True)
class RunRecord:
    run: str
    split: str
    label: int
    task_label: str
    study_name: str
    study_id: int
    n_sets: int
    file: Path


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


def _build_records(
    run_task_df,
    cache_root: Path,
    *,
    label_encoder: LabelEncoder,
    requested_num_sets: int,
    study_name_to_id: Optional[Mapping[str, int]] = None,
) -> Tuple[List[RunRecord], int, Tuple[str, ...]]:
    rows = (
        run_task_df.loc[:, ["Run", "split", "task_label", "study_name"]]
        .drop_duplicates(subset=["Run"])
        .reset_index(drop=True)
    )
    out: List[RunRecord] = []
    missing_runs: List[str] = []

    for _, row in rows.iterrows():
        run = str(row["Run"]).strip()
        pt_path = cache_root / f"{run}.pt"
        if not pt_path.is_file():
            missing_runs.append(run)
            continue
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(pt_path, map_location="cpu")
        n_sets = int(blob.get("n_sets", 0))
        if n_sets <= 0:
            continue
        study_name = str(row["study_name"]).strip()
        if study_name_to_id is None:
            study_id = 0
        else:
            study_id = int(study_name_to_id.get(study_name, -1))
        out.append(
            RunRecord(
                run=run,
                split=str(row["split"]),
                label=int(label_encoder.transform([str(row["task_label"])])[0]),
                task_label=str(row["task_label"]),
                study_name=study_name,
                study_id=study_id,
                n_sets=min(n_sets, requested_num_sets),
                file=pt_path,
            )
        )

    n_skipped_missing = len(missing_runs)
    examples = tuple(sorted(missing_runs)[:5])
    return out, n_skipped_missing, examples


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
                "train_hyenadna.max_length exceeds run_tensors.max_length. "
                "Increase run_tensors.max_length and rebuild run tensors."
            )
        input_ids = blob["input_ids"][: self.requested_num_sets, start:self.cache_max_len]
        attention_mask = blob["attention_mask"][: self.requested_num_sets, start:self.cache_max_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_sets": n_sets,
            "label": e.label,
            "study_id": int(e.study_id),
            "run": e.run,
        }


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    n_sets = torch.tensor([b["n_sets"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    study_ids = torch.tensor([int(b["study_id"]) for b in batch], dtype=torch.long)
    runs = [str(b["run"]) for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "n_sets": n_sets,
        "label": labels,
        "study_id": study_ids,
        "run": runs,
    }


def training_loss(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ce_weight: Optional[torch.Tensor],
    label_smoothing: float,
    study_train_kw: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    loss_sum, denom, _ = training_loss_sum_and_count(
        model,
        batch,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        ce_weight=ce_weight,
        label_smoothing=label_smoothing,
        study_train_kw=study_train_kw,
    )
    if denom <= 0:
        raise SystemExit("Training batch has zero valid sets after masking.")
    return loss_sum / float(denom)


def training_loss_sum_and_count(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ce_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    study_train_kw: Optional[Mapping[str, object]] = None,
) -> Tuple[torch.Tensor, int, Optional[Dict[str, float]]]:
    input_ids = batch["input_ids"].to(device)
    bsz, n_set, seq_len = input_ids.shape
    flat_in = input_ids.view(bsz * n_set, seq_len)
    nv = batch["n_sets"]
    labels = batch["label"].to(device)
    mask = torch.arange(n_set, device=device).unsqueeze(0) < nv.to(device).unsqueeze(1)
    study_ids = batch.get("study_id")
    if study_ids is None:
        raise SystemExit("Batch missing study_id (internal error).")
    study_ids_t = study_ids.to(device)

    phase = (
        str(study_train_kw.get("phase", "off")).strip().lower()
        if study_train_kw is not None
        else "off"
    )
    use_study = bool(getattr(model, "use_study_adv", False)) and phase in ("warmup", "dann")

    with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        if use_study:
            assert study_train_kw is not None
            logits, study_logits = model(
                flat_in,
                return_study_logits=True,
                study_grl_lambda=float(study_train_kw.get("study_grl_lambda", 0.0)),
                study_discriminator_detach=(phase == "warmup"),
            )
            logits = logits.view(bsz, n_set, -1)
            study_logits = study_logits.view(bsz, n_set, -1)
        else:
            logits = model(flat_in)
            logits = logits.view(bsz, n_set, -1)
            study_logits = None

        flat_logits = logits[mask]
        flat_y = labels.unsqueeze(1).expand(bsz, n_set)[mask]
    if flat_logits.numel() == 0:
        return torch.zeros((), device=device), 0, None
    ce_kw: Dict[str, object] = {"reduction": "sum"}
    if ce_weight is not None:
        # AMP can produce bf16/fp16 logits; class-weight tensor must match.
        ce_kw["weight"] = ce_weight.to(
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        )
    if label_smoothing > 0.0:
        ce_kw["label_smoothing"] = float(label_smoothing)
    task_ce = F.cross_entropy(flat_logits, flat_y, **ce_kw)
    n_task = int(flat_y.shape[0])

    if (
        study_logits is None
        or phase in ("off", "delay")
        or not getattr(model, "use_study_adv", False)
    ):
        return task_ce, n_task, None

    flat_study_logits = study_logits[mask]
    flat_study_y = study_ids_t.unsqueeze(1).expand(bsz, n_set)[mask].long()
    study_ce = F.cross_entropy(
        flat_study_logits,
        flat_study_y,
        ignore_index=STUDY_IGNORE_INDEX,
        reduction="sum",
    )
    n_study_valid = int((flat_study_y != STUDY_IGNORE_INDEX).sum().item())
    if n_study_valid <= 0:
        return task_ce, n_task, None

    w = float(study_train_kw.get("study_adv_weight", 0.1))
    scale = float(n_task) / float(max(n_study_valid, 1))
    total = task_ce + w * study_ce * scale
    extra = {
        "study_ce_sum": float(study_ce.detach().float().item()),
        "study_n": float(n_study_valid),
    }
    return total, n_task, extra


def _slice_batch_sets(
    batch: Mapping[str, object],
    *,
    start: int,
    end: int,
) -> Dict[str, object]:
    batch_input_ids = batch["input_ids"]
    batch_attention_mask = batch["attention_mask"]
    batch_n_sets = batch["n_sets"]
    sub_n_sets = torch.clamp(batch_n_sets - start, min=0, max=max(end - start, 0))
    return {
        "input_ids": batch_input_ids[:, start:end, :],
        "attention_mask": batch_attention_mask[:, start:end, :],
        "n_sets": sub_n_sets,
        "label": batch["label"],
        "study_id": batch["study_id"],
        "run": batch["run"],
    }


def _count_valid_sets(batch: Mapping[str, object]) -> int:
    n_sets = batch["n_sets"]
    if not isinstance(n_sets, torch.Tensor):
        raise SystemExit("Expected batch n_sets to be a torch.Tensor.")
    return int(n_sets.sum().item())


def _resolve_train_batch_size(raw: object) -> Tuple[float, int, bool]:
    value = float(raw)
    if value == HALF_BATCH_SIZE:
        return value, 1, True
    if not value.is_integer() or value < 1:
        raise SystemExit(
            "train_hyenadna.batch_size must be a positive integer, or exactly 0.5 "
            "to enable split-set microbatch mode."
        )
    return value, int(value), False


def _resolve_amp_config(
    merged: Mapping[str, object],
    device: torch.device,
) -> Tuple[bool, torch.dtype, str, bool]:
    amp_requested = bool(merged.get("amp", False))
    amp_dtype_raw = str(merged.get("amp_dtype", "float16")).strip().lower()
    if amp_dtype_raw == "float16":
        amp_dtype = torch.float16
        use_grad_scaler = True
    elif amp_dtype_raw == "bfloat16":
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    else:
        raise SystemExit("train_hyenadna.amp_dtype must be 'float16' or 'bfloat16'.")

    if not amp_requested:
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    if device.type != "cuda":
        print("AMP requested but CUDA is unavailable; running in float32.", flush=True)
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    return True, amp_dtype, amp_dtype_raw, use_grad_scaler


def _amp_autocast(
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> ContextManager[object]:
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


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
    amp_eval_enabled: bool,
    amp_dtype: torch.dtype,
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
                    "train_hyenadna.max_length exceeds run_tensors.max_length. "
                    "Increase run_tensors.max_length and rebuild run tensors."
                )
            x = blob["input_ids"][:requested_num_sets, start:cache_max_len].to(device)
            nv = min(int(blob["n_sets"]), requested_num_sets, cache_num_sets)
            if nv <= 0:
                continue
            with _amp_autocast(device, amp_enabled=amp_eval_enabled, amp_dtype=amp_dtype):
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


def run_level_prediction_rows(
    entries: Sequence[RunRecord],
    y_score: np.ndarray,
    *,
    negative_label: str,
    positive_label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if len(entries) != int(y_score.shape[0]):
        raise SystemExit(
            "Prediction row count mismatch while writing HyenaDNA split predictions."
        )
    for e, score in zip(entries, y_score):
        pred = positive_label if float(score) >= 0.5 else negative_label
        rows.append(
            {
                "Run": str(e.run),
                "task_label": str(e.task_label),
                "predicted_label": str(pred),
                "positive_score": float(score),
            }
        )
    return rows


def _write_predictions_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Run", "task_label", "predicted_label", "positive_score"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Run": str(row["Run"]),
                    "task_label": str(row["task_label"]),
                    "predicted_label": str(row["predicted_label"]),
                    "positive_score": f"{float(row['positive_score']):.10f}",
                }
            )


def run_level_weighted_f1_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    negative_label: str,
    positive_label: str,
) -> float:
    if y_true.size == 0:
        return float("nan")
    y_pred = np.where(y_score >= 0.5, positive_label, negative_label)
    return float(f1_score(y_true, y_pred, average="weighted"))


def _study_sampler_weights(entries: Sequence[RunRecord]) -> torch.Tensor:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e.study_name] += 1
    w = [1.0 / counts[e.study_name] for e in entries]
    return torch.tensor(w, dtype=torch.double)


def _compute_ce_weight_tensor(
    train_entries: Sequence[RunRecord],
    *,
    mode: str,
    n_classes: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    m = str(mode or "none").strip().lower()
    if m in ("none", "null", ""):
        return None
    if m in ("balanced", "balanced_sqrt"):
        y = np.array([e.label for e in train_entries], dtype=np.int64)
        classes = np.arange(n_classes, dtype=np.int64)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        if m == "balanced_sqrt":
            cw = np.sqrt(cw)
            cw = cw / np.mean(cw)
        t = torch.zeros(n_classes, dtype=torch.float32, device=device)
        for i in range(n_classes):
            t[i] = float(cw[i])
        return t
    raise SystemExit(
        f"Unknown class_weight {mode!r} (use none, balanced, or balanced_sqrt)."
    )


def _eval_mean_ce_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ce_weight: Optional[torch.Tensor],
    label_smoothing: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss_sum, n, _ = training_loss_sum_and_count(
                model,
                batch,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                ce_weight=ce_weight,
                label_smoothing=label_smoothing,
                study_train_kw=None,
            )
            total += float(loss_sum.item())
            count += int(n)
    return total / max(count, 1)


def _eval_val_study_branch_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float, int]:
    """Mean study CE and accuracy on validation (only rows with train-mapped study_id)."""
    if not getattr(model, "use_study_adv", False) or not hasattr(model, "eval_study_logits"):
        return float("nan"), float("nan"), 0
    model.eval()
    ce_sum = 0.0
    correct = 0
    n_labeled = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            bsz, n_set, seq_len = input_ids.shape
            nv = batch["n_sets"].to(device)
            study_ids = batch["study_id"].to(device)
            mask = torch.arange(n_set, device=device).unsqueeze(0) < nv.unsqueeze(1)
            flat_in = input_ids.view(bsz * n_set, seq_len)
            flat_study_y = study_ids.unsqueeze(1).expand(bsz, n_set)[mask]
            with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                slogits = model.eval_study_logits(flat_in)
            slogits = slogits.view(bsz, n_set, -1)
            flat_slogits = slogits[mask]
            labeled = flat_study_y != STUDY_IGNORE_INDEX
            n_lab = int(labeled.sum().item())
            if n_lab <= 0:
                continue
            ce = F.cross_entropy(
                flat_slogits[labeled],
                flat_study_y[labeled].long(),
                reduction="sum",
            )
            pred = flat_slogits[labeled].argmax(dim=-1)
            ce_sum += float(ce.item())
            correct += int((pred == flat_study_y[labeled]).sum().item())
            n_labeled += n_lab
    if n_labeled <= 0:
        return float("nan"), float("nan"), 0
    return ce_sum / n_labeled, correct / float(n_labeled), n_labeled


def _make_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: Optional[float],
) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if getattr(model, "study_head", None) is not None:
        head_params.extend([p for p in model.study_head.parameters() if p.requires_grad])
    groups: List[Dict[str, object]] = []
    if backbone_params:
        blr = lr * float(backbone_lr_mult) if backbone_lr_mult is not None else lr
        groups.append({"params": backbone_params, "lr": blr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if not groups:
        raise SystemExit("No trainable parameters for optimizer.")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def _set_backbone_requires_grad(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    for p in model.head.parameters():
        p.requires_grad = True
    if getattr(model, "study_head", None) is not None:
        for p in model.study_head.parameters():
            p.requires_grad = True


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr_ratio: float,
) -> Optional[Any]:
    sched = str(schedule or "none").strip().lower()
    if sched == "none":
        return None
    if sched == "warmup_cosine":
        wu = max(int(warmup_epochs), 0)
        if wu >= epochs:
            raise SystemExit("warmup_epochs must be < epochs when using warmup_cosine.")
        min_lr = float(base_lr) * float(min_lr_ratio)
        rem = max(epochs - wu, 1)
        cosine = CosineAnnealingLR(optimizer, T_max=rem, eta_min=min_lr)
        if wu == 0:
            return cosine
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=wu)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[wu])
    raise SystemExit(f"Unknown lr_schedule {schedule!r} (use none or warmup_cosine).")


def _per_study_roc_dict(
    entries: Sequence[RunRecord],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Optional[float]]:
    by_study: Dict[str, List[Tuple[object, float]]] = defaultdict(list)
    for e, yt, ys in zip(entries, y_true, y_score):
        by_study[e.study_name].append((yt, float(ys)))
    out: Dict[str, Optional[float]] = {}
    for study, pairs in by_study.items():
        yt = np.array([p[0] for p in pairs], dtype=object)
        ys = np.array([p[1] for p in pairs], dtype=np.float64)
        auc = binary_roc_auc_from_scores(yt, ys)
        out[study] = float(auc) if auc == auc else None
    return out


def _per_study_score_diagnostics(
    entries: Sequence[RunRecord],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    negative_label: str,
    positive_label: str,
) -> Dict[str, Dict[str, object]]:
    by_study: Dict[str, List[Tuple[object, float]]] = defaultdict(list)
    for e, yt, ys in zip(entries, y_true, y_score):
        by_study[e.study_name].append((yt, float(ys)))
    out: Dict[str, Dict[str, object]] = {}
    for study, pairs in by_study.items():
        yt = np.array([p[0] for p in pairs], dtype=object)
        ys = np.array([p[1] for p in pairs], dtype=np.float64)
        y_pred = np.where(ys >= 0.5, positive_label, negative_label)
        true_counts = {
            str(lbl): int(np.sum(yt == lbl)) for lbl in (negative_label, positive_label)
        }
        pred_counts = {
            str(lbl): int(np.sum(y_pred == lbl)) for lbl in (negative_label, positive_label)
        }
        out[study] = {
            "n_runs": int(yt.size),
            "true_label_counts": true_counts,
            "predicted_label_counts": pred_counts,
            "predicted_positive_rate": float(np.mean(y_pred == positive_label)),
            "score_mean": float(np.mean(ys)),
            "score_std": float(np.std(ys)),
            "accuracy": float(np.mean(y_pred == yt)),
        }
    return out


def _label_score_moments(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    neg_label: str,
    pos_label: str,
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=object)
    y_score = np.asarray(y_score, dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in (
        (str(neg_label), y_true == neg_label),
        (str(pos_label), y_true == pos_label),
    ):
        s = y_score[mask]
        if s.size == 0:
            out[name] = {"mean": float("nan"), "std": float("nan"), "n": 0.0}
        else:
            out[name] = {
                "mean": float(np.mean(s)),
                "std": float(np.std(s)),
                "n": float(s.size),
            }
    return out


def _study_macro_roc_auc(
    per_study_auc: Mapping[str, Optional[float]],
) -> Tuple[float, int]:
    vals = [float(v) for v in per_study_auc.values() if v is not None and float(v) == float(v)]
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), int(len(vals))


def _tuning_score(
    val_f1: float,
    val_auc: float,
    val_study_macro_auc: float,
    val_study_n_roc_eligible: int,
    metric: str,
) -> float:
    m = str(metric).strip().lower()
    if m in ("val_roc_auc", "roc_auc", "val_auc"):
        v = float(val_auc)
    elif m in (
        "val_study_macro_roc_auc",
        "study_macro_roc_auc",
        "val_study_auc",
    ):
        # Some tasks/splits can have only single-class studies in validation, making
        # per-study ROC undefined. Fall back to run-level val ROC in that case.
        if int(val_study_n_roc_eligible) > 0 and float(val_study_macro_auc) == float(
            val_study_macro_auc
        ):
            v = float(val_study_macro_auc)
        else:
            v = float(val_auc)
    elif m in ("val_f1_weighted", "f1", "f1_weighted"):
        v = float(val_f1)
    else:
        raise SystemExit(
            "Unknown tuning_metric "
            f"{metric!r} (use val_roc_auc, val_study_macro_roc_auc, or val_f1_weighted)."
        )
    return v if v == v else float("-inf")


def _average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not states:
        raise SystemExit("SWA requested but no weight snapshots were collected.")
    keys = list(states[0].keys())
    out: Dict[str, torch.Tensor] = {}
    n = float(len(states))
    for k in keys:
        acc: Optional[torch.Tensor] = None
        for sd in states:
            t = sd[k].float()
            acc = t if acc is None else acc + t
        assert acc is not None
        out[k] = (acc / n).to(states[0][k].dtype)
    return out


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip_norm: Optional[float],
) -> float:
    grad_norm = float("nan")
    if grad_clip_norm is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        )
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return grad_norm


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    split_set_microbatch: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: Optional[torch.amp.GradScaler],
    ce_weight: Optional[torch.Tensor],
    label_smoothing: float,
    grad_clip_norm: Optional[float],
    study_train_kw: Optional[Mapping[str, object]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    model.train()
    total = 0.0
    n_batches = 0
    grad_norms: List[float] = []
    study_ce_num = 0.0
    study_ce_den = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        if split_set_microbatch:
            first_half = _slice_batch_sets(batch, start=0, end=SET_SPLIT_INDEX)
            second_half = _slice_batch_sets(batch, start=SET_SPLIT_INDEX, end=10)
            denom_1 = _count_valid_sets(first_half)
            denom_2 = _count_valid_sets(second_half)
            denom = denom_1 + denom_2
            if denom <= 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            if denom_1 > 0:
                loss_sum_1, _, extra1 = training_loss_sum_and_count(
                    model,
                    first_half,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    ce_weight=ce_weight,
                    label_smoothing=label_smoothing,
                    study_train_kw=study_train_kw,
                )
                scaled_loss_1 = loss_sum_1 / float(denom)
                if scaler is not None:
                    scaler.scale(scaled_loss_1).backward()
                else:
                    scaled_loss_1.backward()
                if extra1 is not None:
                    study_ce_num += float(extra1["study_ce_sum"])
                    study_ce_den += float(extra1["study_n"])
            else:
                loss_sum_1 = torch.zeros((), device=device)
            if denom_2 > 0:
                loss_sum_2, _, extra2 = training_loss_sum_and_count(
                    model,
                    second_half,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    ce_weight=ce_weight,
                    label_smoothing=label_smoothing,
                    study_train_kw=study_train_kw,
                )
                scaled_loss_2 = loss_sum_2 / float(denom)
                if scaler is not None:
                    scaler.scale(scaled_loss_2).backward()
                else:
                    scaled_loss_2.backward()
                if extra2 is not None:
                    study_ce_num += float(extra2["study_ce_sum"])
                    study_ce_den += float(extra2["study_n"])
            else:
                loss_sum_2 = torch.zeros((), device=device)
            gn = _optimizer_step(model, optimizer, scaler, grad_clip_norm)
            grad_norms.append(gn)
            total += float((loss_sum_1 + loss_sum_2).item() / float(denom))
        else:
            loss_sum, denom, extra_full = training_loss_sum_and_count(
                model,
                batch,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                ce_weight=ce_weight,
                label_smoothing=label_smoothing,
                study_train_kw=study_train_kw,
            )
            if denom <= 0:
                raise SystemExit("Training batch has zero valid sets after masking.")
            loss = loss_sum / float(denom)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            gn = _optimizer_step(model, optimizer, scaler, grad_clip_norm)
            grad_norms.append(gn)
            total += float(loss.item())
            if extra_full is not None:
                study_ce_num += float(extra_full["study_ce_sum"])
                study_ce_den += float(extra_full["study_n"])
        n_batches += 1
    mean_loss = total / max(n_batches, 1)
    mean_gn = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
    train_study_ce = (
        float(study_ce_num / study_ce_den) if study_ce_den > 0.0 else float("nan")
    )
    return mean_loss, mean_gn, {"train_study_ce": train_study_ce}


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
    best_epoch: int,
    tuning_metric: str,
    best_tuning_score: float,
    best_val_f1_weighted: float,
    best_val_roc_auc: float,
    best_val_study_macro_roc_auc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dict(merged),
        "data": {
            "cache": dict(cache_info),
        },
        "tuning": {
            "split": "validation",
            "metric": str(tuning_metric),
            "score": _float_or_none(float(best_tuning_score)),
            "best_epoch": int(best_epoch),
            "val_f1_weighted_at_best": _float_or_none(float(best_val_f1_weighted)),
            "val_roc_auc_at_best": _float_or_none(float(best_val_roc_auc)),
            "val_study_macro_roc_auc_at_best": _float_or_none(
                float(best_val_study_macro_roc_auc)
            ),
        },
        "metrics": {
            "test": {"roc_auc": float(test_auc) if test_auc == test_auc else None},
            "holdout": {"roc_auc": float(holdout_auc) if holdout_auc == holdout_auc else None},
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _training_log_path_from_results(path: Path) -> Path:
    return path.with_name(f"{path.stem}_training.json")


def _float_or_none(x: float) -> Optional[float]:
    return float(x) if x == x else None


def _write_training_log(path: Path, epoch_rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, object]] = []
    for row in epoch_rows:
        payload.append(
            {
                "epoch": int(row["epoch"]),
                "learning_rate": float(row["learning_rate"]),
                "train_loss": float(row["train_loss"]),
                "grad_norm_clip": _float_or_none(float(row["grad_norm_clip"])),
                "val_loss": _float_or_none(float(row["val_loss"])),
                "val_f1_weighted": _float_or_none(float(row["val_f1_weighted"])),
                "val_roc_auc": _float_or_none(float(row["val_roc_auc"])),
                "val_study_macro_roc_auc": _float_or_none(
                    float(row["val_study_macro_roc_auc"])
                ),
                "val_study_n_roc_eligible": int(row["val_study_n_roc_eligible"]),
                "val_per_study_roc_auc": dict(row["val_per_study_roc_auc"]),
                "train_roc_auc": _float_or_none(float(row["train_roc_auc"])),
                "test_roc_auc": _float_or_none(float(row["test_roc_auc"])),
                "holdout_roc_auc": _float_or_none(float(row["holdout_roc_auc"])),
                "holdout_per_study_roc_auc": dict(row["holdout_per_study_roc_auc"]),
                "holdout_per_study_score_diagnostics": dict(
                    row["holdout_per_study_score_diagnostics"]
                ),
                "val_score_by_label": dict(row["val_score_by_label"]),
                "holdout_score_by_label": dict(row["holdout_score_by_label"]),
                "study_adv_phase": str(row["study_adv_phase"]),
                "study_grl_lambda": _float_or_none(float(row["study_grl_lambda"])),
                "train_study_ce": _float_or_none(float(row["train_study_ce"])),
                "val_study_ce": _float_or_none(float(row["val_study_ce"])),
                "val_study_accuracy": _float_or_none(float(row["val_study_accuracy"])),
                "val_study_n_labeled": int(row["val_study_n_labeled"]),
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = _parse_argv(argv)
    expt = int(cli.expt) if cli.expt is not None else 0

    repo_root = Path(__file__).resolve().parent.parent

    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"

    merged, exp_name, _tpl = merge_train_hyenadna_config(
        defaults_path,
        experiments_path,
        expt=expt,
    )

    if hasattr(cli, "results_json"):
        rj = cli.results_json
        if rj == "":
            merged = {**merged, "results_json": ""}
        else:
            merged = {**merged, "results_json": rj}

    defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    if not isinstance(defaults_cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    paths_cfg = defaults_cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    run_tensors_cfg = defaults_cfg.get("run_tensors")
    if not isinstance(run_tensors_cfg, dict):
        raise SystemExit(f"{defaults_path} must define run_tensors as a mapping.")
    run_tensors_root = resolve_repo_path(
        repo_root,
        str(paths_cfg.get("run_tensors_dir", "outputs/run_tensors")).strip(),
    )

    cache_num_sets = int(run_tensors_cfg["num_sets"])
    cache_max_len = int(run_tensors_cfg["max_length"])
    cache_seq_offset = int(run_tensors_cfg.get("seq_offset", 0))
    cache_min_seqs = int(run_tensors_cfg.get("min_seqs", 0))

    task = str(merged["task"]).strip()
    model_name = str(merged["model"]).strip()
    num_sets = int(merged["num_sets"])
    raw_batch_size = merged["batch_size"]
    _batch_size_cfg, loader_batch_size, split_set_microbatch = _resolve_train_batch_size(
        raw_batch_size
    )
    max_len = model_max_length(model_name, merged.get("max_length"))
    head_mode = str(merged["head_pooling_mode"]).strip()
    aggregate = str(merged["run_logits_aggregate"]).strip().lower()
    if head_mode not in HEAD_MODES:
        raise SystemExit(f"head_pooling_mode must be one of {HEAD_MODES}; got {head_mode!r}.")
    if aggregate not in AGGREGATES:
        raise SystemExit(f"run_logits_aggregate must be one of {AGGREGATES}; got {aggregate!r}.")
    if num_sets > cache_num_sets:
        raise SystemExit(
            f"train_hyenadna.num_sets ({num_sets}) exceeds run_tensors.num_sets "
            f"({cache_num_sets}). Rebuild run tensors with a larger cache."
        )
    if max_len > cache_max_len:
        raise SystemExit(
            f"Resolved max_length ({max_len}) exceeds run_tensors.max_length "
            f"({cache_max_len}). Rebuild run tensors with a larger cache."
        )
    if not run_tensors_root.is_dir():
        raise SystemExit(
            f"Missing run tensors directory: {run_tensors_root}. "
            "Run: python scripts/build_run_tensors.py"
        )
    if split_set_microbatch and num_sets != 10:
        raise SystemExit(
            "train_hyenadna.batch_size=0.5 requires train_hyenadna.num_sets=10 "
            "to split each run into sets 0-4 and 5-9."
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
    if len(class_names) < 2:
        raise SystemExit("HyenaDNA training expects at least 2 task classes.")

    neg_class_index = 0
    pos_class_index = 1
    neg_label = class_names[neg_class_index]
    pos_label = class_names[pos_class_index]

    study_adv_enabled = bool(merged.get("study_adv", False))
    study_name_to_id: Optional[Dict[str, int]] = None
    if study_adv_enabled:
        tr_st = run_task_df.loc[run_task_df["split"] == "train", "study_name"]
        uniq_studies = sorted({str(s).strip() for s in tr_st.unique()})
        if len(uniq_studies) < 2:
            raise SystemExit(
                "train_hyenadna.study_adv requires at least 2 distinct study_name values "
                "in the train split."
            )
        study_name_to_id = {name: i for i, name in enumerate(uniq_studies)}

    all_records, n_skipped_missing_cache, missing_cache_examples = _build_records(
        run_task_df,
        run_tensors_root,
        label_encoder=enc,
        requested_num_sets=num_sets,
        study_name_to_id=study_name_to_id,
    )
    if n_skipped_missing_cache:
        ex = ", ".join(missing_cache_examples)
        print(
            f"train_hyenadna: skipping {n_skipped_missing_cache} metadata run(s) with no "
            f"{run_tensors_root.name}/*.pt tensor (examples: {ex}).",
            flush=True,
        )
    by_split: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in all_records:
        by_split[r.split].append(r)

    train_entries = by_split["train"]
    val_entries = by_split["val"]
    test_entries = by_split["test"]
    holdout_entries = by_split["holdout"]
    if not train_entries:
        raise SystemExit("No training runs in cache (split=train). Build dataset first.")
    if not val_entries:
        raise SystemExit("No validation runs in cache.")
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
    amp_enabled, amp_dtype, amp_dtype_name, amp_use_grad_scaler = _resolve_amp_config(
        merged, device
    )
    amp_eval_enabled = amp_enabled
    exp_label = str(exp_name).strip() if exp_name is not None else ""
    if expt > 0 and exp_label:
        print(f"\nExperiment: EXPT={expt} name={exp_label}", flush=True)
    elif expt > 0:
        print(f"\nExperiment: EXPT={expt}", flush=True)
    else:
        print("\nExperiment: EXPT=0 (defaults.yaml config)", flush=True)
    print(
        f"\nHyenaDNA train | task={task} model={model_name} | "
        f"positive_class={pos_label!r} (index {pos_class_index}) | "
        f"precision={'amp(' + amp_dtype_name + ')' if amp_enabled else 'fp32'} | "
        f"study_adv={'on' if study_adv_enabled else 'off'}",
        flush=True,
    )

    lr = float(merged["learning_rate"])
    wd = float(merged["weight_decay"])
    label_smoothing = float(merged.get("label_smoothing") or 0.0)
    raw_blm = merged.get("backbone_lr_mult")
    backbone_lr_mult: Optional[float] = (
        None if raw_blm is None else float(raw_blm)
    )
    raw_gc = merged.get("grad_clip_norm")
    grad_clip_norm: Optional[float] = None if raw_gc is None else float(raw_gc)
    train_sampler = str(merged.get("train_sampler") or "random").strip().lower()
    tuning_metric = str(merged.get("tuning_metric") or "val_roc_auc").strip()
    ce_weight = _compute_ce_weight_tensor(
        train_entries,
        mode=str(merged.get("class_weight") or "none"),
        n_classes=len(class_names),
        device=device,
    )
    freeze_full = bool(merged.get("freeze_backbone"))
    transition_n = int(merged.get("freeze_backbone_epochs") or 0)
    swa_enabled = bool(merged.get("swa"))
    swa_keep = int(merged.get("swa_epochs") or 0)
    train_subset_n = min(
        int(merged.get("train_eval_subset_size") or 200),
        len(train_entries),
    )
    rng_idx = np.random.RandomState(seed).permutation(len(train_entries))[
        :train_subset_n
    ]
    train_eval_entries = [train_entries[int(i)] for i in rng_idx]

    head_arch = str(merged.get("head_arch") or "linear").strip().lower()
    raw_hh = merged.get("head_hidden")
    head_hidden = int(raw_hh) if raw_hh is not None else None
    head_dropout = float(merged.get("head_dropout") or 0.0)

    train_ds = RunTensorDataset(
        train_entries,
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
    )
    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True
    if train_sampler == "study_balanced":
        sampler = WeightedRandomSampler(
            _study_sampler_weights(train_entries),
            num_samples=len(train_entries),
            replacement=True,
        )
        shuffle = False
    elif train_sampler != "random":
        raise SystemExit(
            f"Unknown train_sampler {train_sampler!r} (use random or study_balanced)."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=loader_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(merged["num_workers"]),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    val_ds = RunTensorDataset(
        val_entries,
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=int(merged["num_workers"]),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    ckpt_dir = resolve_repo_path(repo_root, str(merged["checkpoint_dir"]).strip())
    n_study_classes = len(study_name_to_id) if study_name_to_id is not None else 0
    study_head_hidden = int(merged.get("study_head_hidden") or 256)
    study_head_dropout = float(merged.get("study_head_dropout") or 0.0)
    model = HyenaDNAPreTrainedModel.from_pretrained(
        str(ckpt_dir),
        model_name,
        download=bool(merged["download_pretrained"]),
        config=None,
        device=str(device),
        use_head=True,
        n_classes=len(class_names),
        head_pooling_mode=head_mode,
        head_arch=head_arch,
        head_hidden=head_hidden if head_arch == "mlp" else None,
        head_dropout=head_dropout,
        use_study_adv=study_adv_enabled,
        n_study_classes=n_study_classes,
        study_head_hidden=study_head_hidden,
        study_head_dropout=study_head_dropout,
    )
    model = model.to(device)

    scaler: Optional[torch.amp.GradScaler] = None
    if amp_enabled and amp_use_grad_scaler:
        scaler = torch.amp.GradScaler("cuda")

    epochs = int(merged["epochs"])
    epoch_log: List[Dict[str, object]] = []
    best_tuning_score = float("-inf")
    best_epoch = 0
    best_val_f1 = float("nan")
    best_val_roc_auc = float("nan")
    best_val_study_macro_roc_auc = float("nan")
    best_test_auc = float("nan")
    best_holdout_auc = float("nan")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    swa_snapshots: List[Dict[str, torch.Tensor]] = []

    opt: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None

    results_path = _results_json_out_path(
        repo_root,
        merged.get("results_json"),
        task=task,
    )
    training_log_path: Optional[Path] = None
    if results_path is not None:
        training_log_path = _training_log_path_from_results(results_path)

    schedule_key = str(merged.get("lr_schedule") or "none").strip().lower()
    warmup_epochs_cfg = int(merged.get("warmup_epochs") or 0)
    min_lr_ratio_cfg = float(merged.get("min_lr_ratio") or 0.1)
    tuning_metric_key = tuning_metric.strip().lower()
    warned_study_macro_fallback = False

    for ep in range(1, epochs + 1):
        print(f"\n--- Epoch {ep}/{epochs} ---", flush=True)

        if freeze_full:
            _set_backbone_requires_grad(model, False)
        elif transition_n > 0:
            _set_backbone_requires_grad(model, ep > transition_n)
        else:
            _set_backbone_requires_grad(model, True)

        if ep == 1 or (transition_n > 0 and ep == transition_n + 1):
            if freeze_full:
                opt = _make_optimizer(
                    model, lr=lr, weight_decay=wd, backbone_lr_mult=None
                )
                scheduler = None
            elif transition_n > 0 and ep <= transition_n:
                opt = _make_optimizer(
                    model, lr=lr, weight_decay=wd, backbone_lr_mult=None
                )
                scheduler = None
            elif transition_n > 0 and ep == transition_n + 1:
                opt = _make_optimizer(
                    model,
                    lr=lr,
                    weight_decay=wd,
                    backbone_lr_mult=backbone_lr_mult,
                )
                rem_epochs = max(epochs - transition_n, 1)
                scheduler = _make_lr_scheduler(
                    opt,
                    schedule=schedule_key,
                    epochs=rem_epochs,
                    warmup_epochs=min(warmup_epochs_cfg, max(rem_epochs - 1, 0)),
                    base_lr=lr,
                    min_lr_ratio=min_lr_ratio_cfg,
                )
            else:
                opt = _make_optimizer(
                    model,
                    lr=lr,
                    weight_decay=wd,
                    backbone_lr_mult=backbone_lr_mult,
                )
                scheduler = _make_lr_scheduler(
                    opt,
                    schedule=schedule_key,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs_cfg,
                    base_lr=lr,
                    min_lr_ratio=min_lr_ratio_cfg,
                )

        assert opt is not None
        study_phase = _resolve_study_adv_phase(
            ep,
            study_adv_enabled=study_adv_enabled,
            delay_epochs=int(merged.get("study_adv_delay_epochs") or 0),
            disc_warmup_enabled=bool(merged.get("study_discriminator_warmup", False)),
            disc_warmup_epochs=int(merged.get("study_disc_warmup_epochs") or 0),
        )
        study_train_kw: Optional[Dict[str, object]] = None
        if study_adv_enabled:
            saw = float(merged.get("study_adv_weight") or 0.1)
            study_train_kw = {
                "phase": study_phase,
                "study_adv_weight": saw,
                "study_grl_lambda": float(saw) if study_phase == "dann" else 0.0,
            }
            print(
                f"study_adv phase={study_phase}  study_grl_lambda={study_train_kw['study_grl_lambda']}",
                flush=True,
            )

        last_loss, batch_grad_norm, train_study_stats = train_epoch(
            model,
            train_loader,
            opt,
            device,
            split_set_microbatch=split_set_microbatch,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            ce_weight=ce_weight,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
            study_train_kw=study_train_kw,
        )
        train_study_ce = float(train_study_stats["train_study_ce"])

        val_loss = _eval_mean_ce_loss(
            model,
            val_loader,
            device,
            amp_enabled=amp_eval_enabled,
            amp_dtype=amp_dtype,
            ce_weight=ce_weight,
            label_smoothing=label_smoothing,
        )

        val_study_ce = float("nan")
        val_study_acc = float("nan")
        val_study_n_labeled = 0
        if study_adv_enabled:
            val_study_ce, val_study_acc, val_study_n_labeled = _eval_val_study_branch_metrics(
                model,
                val_loader,
                device,
                amp_enabled=amp_eval_enabled,
                amp_dtype=amp_dtype,
            )

        val_true, val_score = run_level_scores(
            model,
            val_entries,
            device,
            aggregate=aggregate,
            pos_class_index=pos_class_index,
            requested_num_sets=num_sets,
            requested_max_len=max_len,
            cache_num_sets=cache_num_sets,
            cache_max_len=cache_max_len,
            amp_eval_enabled=amp_eval_enabled,
            amp_dtype=amp_dtype,
        )
        val_f1 = run_level_weighted_f1_from_scores(
            val_true,
            val_score,
            negative_label=neg_label,
            positive_label=pos_label,
        )
        val_auc = binary_roc_auc_from_scores(val_true, val_score)
        val_ps = _per_study_roc_dict(val_entries, val_true, val_score)
        val_study_macro_auc, val_study_n_roc_eligible = _study_macro_roc_auc(val_ps)

        train_true, train_score = run_level_scores(
            model,
            train_eval_entries,
            device,
            aggregate=aggregate,
            pos_class_index=pos_class_index,
            requested_num_sets=num_sets,
            requested_max_len=max_len,
            cache_num_sets=cache_num_sets,
            cache_max_len=cache_max_len,
            amp_eval_enabled=amp_eval_enabled,
            amp_dtype=amp_dtype,
        )
        train_auc = binary_roc_auc_from_scores(train_true, train_score)

        test_true, test_score_m = run_level_scores(
            model,
            test_entries,
            device,
            aggregate=aggregate,
            pos_class_index=pos_class_index,
            requested_num_sets=num_sets,
            requested_max_len=max_len,
            cache_num_sets=cache_num_sets,
            cache_max_len=cache_max_len,
            amp_eval_enabled=amp_eval_enabled,
            amp_dtype=amp_dtype,
        )
        test_auc = binary_roc_auc_from_scores(test_true, test_score_m)

        hold_true: np.ndarray = np.asarray([], dtype=object)
        hold_score_m = np.asarray([], dtype=np.float64)
        hold_auc = float("nan")
        if holdout_entries:
            hold_true, hold_score_m = run_level_scores(
                model,
                holdout_entries,
                device,
                aggregate=aggregate,
                pos_class_index=pos_class_index,
                requested_num_sets=num_sets,
                requested_max_len=max_len,
                cache_num_sets=cache_num_sets,
                cache_max_len=cache_max_len,
                amp_eval_enabled=amp_eval_enabled,
                amp_dtype=amp_dtype,
            )
            hold_auc = binary_roc_auc_from_scores(hold_true, hold_score_m)

        hold_ps = _per_study_roc_dict(holdout_entries, hold_true, hold_score_m)
        hold_ps_diag = _per_study_score_diagnostics(
            holdout_entries,
            hold_true,
            hold_score_m,
            negative_label=neg_label,
            positive_label=pos_label,
        )
        val_by_label = _label_score_moments(
            val_true,
            val_score,
            neg_label=neg_label,
            pos_label=pos_label,
        )
        hold_by_label = _label_score_moments(
            hold_true,
            hold_score_m,
            neg_label=neg_label,
            pos_label=pos_label,
        )

        lr_log = float(opt.param_groups[0]["lr"])
        if (
            tuning_metric_key
            in ("val_study_macro_roc_auc", "study_macro_roc_auc", "val_study_auc")
            and val_study_n_roc_eligible == 0
            and not warned_study_macro_fallback
        ):
            print(
                "tuning_metric=val_study_macro_roc_auc has zero ROC-eligible validation "
                "studies this run; falling back to val_roc_auc for checkpoint selection.",
                flush=True,
            )
            warned_study_macro_fallback = True
        ts = _tuning_score(
            val_f1,
            val_auc,
            val_study_macro_auc,
            val_study_n_roc_eligible,
            tuning_metric,
        )
        epoch_log.append(
            {
                "epoch": int(ep),
                "learning_rate": lr_log,
                "train_loss": float(last_loss),
                "grad_norm_clip": float(batch_grad_norm),
                "val_loss": float(val_loss),
                "val_f1_weighted": float(val_f1),
                "val_roc_auc": float(val_auc),
                "val_study_macro_roc_auc": float(val_study_macro_auc),
                "val_study_n_roc_eligible": int(val_study_n_roc_eligible),
                "val_per_study_roc_auc": val_ps,
                "train_roc_auc": float(train_auc),
                "test_roc_auc": float(test_auc),
                "holdout_roc_auc": float(hold_auc),
                "holdout_per_study_roc_auc": hold_ps,
                "holdout_per_study_score_diagnostics": hold_ps_diag,
                "val_score_by_label": val_by_label,
                "holdout_score_by_label": hold_by_label,
                "study_adv_phase": str(study_phase),
                "study_grl_lambda": float(study_train_kw["study_grl_lambda"]) if study_train_kw else 0.0,
                "train_study_ce": float(train_study_ce),
                "val_study_ce": float(val_study_ce),
                "val_study_accuracy": float(val_study_acc),
                "val_study_n_labeled": int(val_study_n_labeled),
            }
        )
        if training_log_path is not None:
            _write_training_log(training_log_path, epoch_log)

        if ts > best_tuning_score:
            best_tuning_score = float(ts)
            best_epoch = int(ep)
            best_val_f1 = float(val_f1)
            best_val_roc_auc = float(val_auc)
            best_val_study_macro_roc_auc = float(val_study_macro_auc)
            best_test_auc = float(test_auc)
            best_holdout_auc = float(hold_auc)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if swa_enabled and swa_keep > 0 and ep > epochs - swa_keep:
            swa_snapshots.append(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

        if scheduler is not None:
            scheduler.step()

        print(
            f"train_loss={last_loss:.6f}  val_loss={val_loss:.6f}  val_roc_auc={val_auc:.6f}  "
            f"test_roc_auc={test_auc:.6f}  holdout_roc_auc={hold_auc:.6f}  "
            f"grad_norm={batch_grad_norm:.6f}  (run-level; logits={aggregate})"
            + (
                f"  train_study_ce={train_study_ce:.4f}  val_study_ce={val_study_ce:.4f}  "
                f"val_study_acc={val_study_acc:.4f}"
                if study_adv_enabled
                else ""
            ),
            flush=True,
        )

    use_swa = swa_enabled and bool(swa_snapshots)
    if use_swa:
        model.load_state_dict(_average_state_dicts(swa_snapshots))
    elif best_state is not None:
        model.load_state_dict(best_state)

    if results_path is not None:
        test_true_final, test_score_best = run_level_scores(
            model,
            test_entries,
            device,
            aggregate=aggregate,
            pos_class_index=pos_class_index,
            requested_num_sets=num_sets,
            requested_max_len=max_len,
            cache_num_sets=cache_num_sets,
            cache_max_len=cache_max_len,
            amp_eval_enabled=amp_eval_enabled,
            amp_dtype=amp_dtype,
        )
        final_test_auc = binary_roc_auc_from_scores(test_true_final, test_score_best)
        hold_score_best = np.asarray([], dtype=np.float64)
        final_holdout_auc = float("nan")
        if holdout_entries:
            hold_true_final, hold_score_best = run_level_scores(
                model,
                holdout_entries,
                device,
                aggregate=aggregate,
                pos_class_index=pos_class_index,
                requested_num_sets=num_sets,
                requested_max_len=max_len,
                cache_num_sets=cache_num_sets,
                cache_max_len=cache_max_len,
                amp_eval_enabled=amp_eval_enabled,
                amp_dtype=amp_dtype,
            )
            final_holdout_auc = binary_roc_auc_from_scores(
                hold_true_final,
                hold_score_best,
            )

        test_pred_rows = run_level_prediction_rows(
            test_entries,
            test_score_best,
            negative_label=neg_label,
            positive_label=pos_label,
        )
        holdout_pred_rows = run_level_prediction_rows(
            holdout_entries[: len(hold_score_best)],
            hold_score_best,
            negative_label=neg_label,
            positive_label=pos_label,
        )
        test_pred_path = results_path.with_name(f"{results_path.stem}_test.csv")
        holdout_pred_path = results_path.with_name(f"{results_path.stem}_holdout.csv")
        _write_predictions_csv(test_pred_path, test_pred_rows)
        _write_predictions_csv(holdout_pred_path, holdout_pred_rows)

        _write_results_json(
            results_path,
            merged=merged,
            cache_info={
                "dir": str(run_tensors_root),
                "run_tensors_num_sets": cache_num_sets,
                "run_tensors_max_length": cache_max_len,
                "run_tensors_seq_offset": cache_seq_offset,
                "run_tensors_min_seqs": cache_min_seqs,
                "n_cached_runs": len(all_records),
            },
            test_auc=final_test_auc,
            holdout_auc=final_holdout_auc,
            best_epoch=best_epoch,
            tuning_metric=tuning_metric,
            best_tuning_score=best_tuning_score,
            best_val_f1_weighted=best_val_f1,
            best_val_roc_auc=best_val_roc_auc,
            best_val_study_macro_roc_auc=best_val_study_macro_roc_auc,
        )
        tail = (
            f"SWA last {swa_keep} epochs"
            if use_swa
            else f"best epoch by {tuning_metric}"
        )
        print(
            f"Final eval ({tail}): best_epoch={best_epoch} "
            f"test_roc_auc={final_test_auc:.6f} holdout_roc_auc={final_holdout_auc:.6f}\n",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
