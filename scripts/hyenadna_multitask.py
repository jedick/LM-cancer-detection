#!/usr/bin/env python3
"""Multitask training, evaluation, and artifact helpers for scripts/train_hyenadna.py."""

from __future__ import annotations

import csv
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from shared_utilities import binary_auroc_from_scores, build_multitask_run_table

MULTITASK_TASK = "multitask"
MT_LABEL_IGNORE = -100
MT_HEAD_CD = 0
MT_HEAD_CT = 1
METRIC_KEY_CD = "cancer_diagnosis"
METRIC_KEY_CT = "cancer_type"
HEAD_CD = "cd"
HEAD_CT = "ct"


def is_multitask(task: str) -> bool:
    return str(task).strip() == MULTITASK_TASK


@dataclass(frozen=True)
class RunRecord:
    run: str
    split: str
    label: int
    task_label: str
    study_name: str
    n_sets: int
    file: Path
    sample_label: str = ""
    ct_label: int = MT_LABEL_IGNORE
    ct_task_label: str = ""
    is_cancer: bool = False


@dataclass(frozen=True)
class BinaryHeadSpec:
    """One binary classification head (single-task uses a single spec)."""

    name: str
    head_index: int
    pos_class_index: int
    neg_label: str
    pos_label: str
    label_attr: str = "task_label"
    cancer_only_metrics: bool = False
    metrics_json_key: Optional[str] = None


@dataclass(frozen=True)
class ScoreContext:
    requested_num_sets: int
    requested_max_len: int
    cache_num_sets: int
    cache_max_len: int


@dataclass(frozen=True)
class HeadScores:
    """Metrics use ``entries``/``y_true``/``y_score``; CSV uses ``all_*`` (one row per input run)."""

    entries: Tuple[RunRecord, ...]
    y_true: np.ndarray
    y_score: np.ndarray
    auroc: float
    f1: float
    all_entries: Tuple[RunRecord, ...] = ()
    all_scores: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))


@dataclass(frozen=True)
class SplitHeadScores:
    val: Dict[str, HeadScores]
    test: Dict[str, HeadScores]
    holdout: Dict[str, HeadScores]


@dataclass(frozen=True)
class TaskTrainingConfig:
    multitask: bool
    loss_ratio: float
    tuning_ratio: float
    heads: Tuple[BinaryHeadSpec, ...]
    primary_head: str
    class_names: Tuple[str, ...]
    ct_class_names: Tuple[str, ...]


def _float_or_none(x: float) -> Optional[float]:
    return float(x) if x == x else None


def _amp_autocast(
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _load_run_tensor(
    path: Path,
    *,
    requested_num_sets: int,
    requested_max_len: int,
    cache_num_sets: int,
    cache_max_len: int,
) -> Tuple[Optional[torch.Tensor], int]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    n_sets = int(blob.get("n_sets", 0))
    if n_sets <= 0:
        return None, 0
    start = cache_max_len - requested_max_len
    if start < 0:
        raise SystemExit(
            "train_hyenadna.max_length exceeds run_tensors.max_length. "
            "Increase run_tensors.max_length and rebuild run tensors."
        )
    x = blob["input_ids"][:requested_num_sets, start:cache_max_len]
    nv = min(n_sets, requested_num_sets, cache_num_sets)
    return x, nv


def build_run_records(
    run_df,
    cache_root: Path,
    *,
    requested_num_sets: int,
    label_col: str,
    label_encoder: LabelEncoder,
    extra_columns: Optional[Sequence[str]] = None,
    ct_encoder: Optional[LabelEncoder] = None,
    ct_label_col: str = "ct_task_label",
) -> Tuple[List[RunRecord], int, Tuple[str, ...]]:
    """Build RunRecords from a metadata frame and per-run tensor cache."""
    base_cols = ["Run", "split", label_col, "study_name"]
    cols = list(base_cols)
    if extra_columns:
        for c in extra_columns:
            if c not in cols:
                cols.append(c)
    rows = run_df.loc[:, cols].drop_duplicates(subset=["Run"]).reset_index(drop=True)
    multitask = ct_encoder is not None
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
        primary_label = str(row[label_col]).strip()
        sample_label = str(row.get("sample_label", "")).strip() if multitask else ""
        ct_task_label = ""
        ct_label = MT_LABEL_IGNORE
        is_cancer = False
        if multitask:
            ct_task_label = str(row[ct_label_col]).strip()
            is_cancer = bool(ct_task_label)
            if is_cancer:
                ct_label = int(ct_encoder.transform([ct_task_label])[0])

        out.append(
            RunRecord(
                run=run,
                split=str(row["split"]),
                label=int(label_encoder.transform([primary_label])[0]),
                task_label=primary_label,
                study_name=study_name,
                n_sets=min(n_sets, requested_num_sets),
                file=pt_path,
                sample_label=sample_label,
                ct_label=ct_label,
                ct_task_label=ct_task_label,
                is_cancer=is_cancer,
            )
        )

    return out, len(missing_runs), tuple(sorted(missing_runs)[:5])


def build_multitask_records(
    run_df,
    cache_root: Path,
    *,
    cd_encoder: LabelEncoder,
    ct_encoder: LabelEncoder,
    requested_num_sets: int,
) -> Tuple[List[RunRecord], int, Tuple[str, ...]]:
    return build_run_records(
        run_df,
        cache_root,
        requested_num_sets=requested_num_sets,
        label_col="cd_task_label",
        label_encoder=cd_encoder,
        extra_columns=["sample_label", "ct_task_label"],
        ct_encoder=ct_encoder,
        ct_label_col="ct_task_label",
    )


def build_single_task_records(
    run_task_df,
    cache_root: Path,
    *,
    label_encoder: LabelEncoder,
    requested_num_sets: int,
) -> Tuple[List[RunRecord], int, Tuple[str, ...]]:
    return build_run_records(
        run_task_df,
        cache_root,
        requested_num_sets=requested_num_sets,
        label_col="task_label",
        label_encoder=label_encoder,
    )


def _positive_class_score(logits: torch.Tensor, pos_class_index: int) -> float:
    """Run-level score: mean logits over sequence sets, then softmax positive class."""
    agg = logits.mean(dim=0)
    return float(torch.softmax(agg, dim=-1)[pos_class_index].item())


def _binary_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    head: BinaryHeadSpec,
) -> float:
    """Positive-class F1 at threshold 0.5 on the positive-class score."""
    if y_true.size == 0:
        return float("nan")
    y_pred = np.where(y_score >= 0.5, head.pos_label, head.neg_label)
    return float(
        f1_score(
            y_true,
            y_pred,
            pos_label=head.pos_label,
            average="binary",
            zero_division=0,
        )
    )


def _head_metrics(
    entries: Sequence[RunRecord],
    scores: np.ndarray,
    head: BinaryHeadSpec,
) -> HeadScores:
    scores = np.asarray(scores, dtype=np.float64)
    valid = scores == scores
    if head.cancer_only_metrics:
        pairs = [
            (e, float(s))
            for e, s, ok in zip(entries, scores, valid)
            if ok and e.is_cancer
        ]
        if not pairs:
            empty = np.asarray([], dtype=object)
            z = np.asarray([], dtype=np.float64)
            all_entries = tuple(entries)
            all_scores = np.asarray(scores, dtype=np.float64)
            return HeadScores(
                (), empty, z, float("nan"), float("nan"),
                all_entries=all_entries, all_scores=all_scores,
            )
        sub_entries, sub_scores = zip(*pairs)
        sub_entries_t = tuple(sub_entries)
        y_true = np.array([getattr(e, head.label_attr) for e in sub_entries_t], dtype=object)
        y_score = np.asarray(sub_scores, dtype=np.float64)
    else:
        pairs = [(e, float(s)) for e, s, ok in zip(entries, scores, valid) if ok]
        if not pairs:
            empty = np.asarray([], dtype=object)
            z = np.asarray([], dtype=np.float64)
            all_entries = tuple(entries)
            all_scores = np.asarray(scores, dtype=np.float64)
            return HeadScores(
                (), empty, z, float("nan"), float("nan"),
                all_entries=all_entries, all_scores=all_scores,
            )
        sub_entries, sub_scores = zip(*pairs)
        sub_entries_t = tuple(sub_entries)
        y_true = np.array([getattr(e, head.label_attr) for e in sub_entries_t], dtype=object)
        y_score = np.asarray(sub_scores, dtype=np.float64)

    auroc = binary_auroc_from_scores(y_true, y_score, positive_label=head.pos_label)
    f1 = _binary_f1(y_true, y_score, head)
    all_entries = tuple(entries)
    all_scores = np.asarray(scores, dtype=np.float64)
    return HeadScores(
        sub_entries_t,
        y_true,
        y_score,
        float(auroc),
        f1,
        all_entries=all_entries,
        all_scores=all_scores,
    )


def _forward_entry_scores(
    model: torch.nn.Module,
    entry: RunRecord,
    heads: Sequence[BinaryHeadSpec],
    ctx: ScoreContext,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    x, nv = _load_run_tensor(
        entry.file,
        requested_num_sets=ctx.requested_num_sets,
        requested_max_len=ctx.requested_max_len,
        cache_num_sets=ctx.cache_num_sets,
        cache_max_len=ctx.cache_max_len,
    )
    if x is None or nv <= 0:
        return {h.name: float("nan") for h in heads}

    multitask_mode = getattr(model, "multitask_mode", False)
    x = x.to(device)
    with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        logits_out = model(x[:nv])
    if multitask_mode:
        if not isinstance(logits_out, (list, tuple)):
            raise SystemExit("Multitask model must return a sequence of logits.")
        per_head = {i: logits_out[i] for i in range(len(logits_out))}
    else:
        if isinstance(logits_out, (list, tuple)):
            raise SystemExit("Single-task model must return a single logits tensor.")
        per_head = {heads[0].head_index: logits_out}
    return {
        h.name: _positive_class_score(per_head[h.head_index], h.pos_class_index)
        for h in heads
    }


def score_entries(
    model: torch.nn.Module,
    entries: Sequence[RunRecord],
    device: torch.device,
    heads: Sequence[BinaryHeadSpec],
    ctx: ScoreContext,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, HeadScores]:
    """Score all entries with one backbone forward per run (all heads)."""
    if not heads:
        raise SystemExit("score_entries requires at least one BinaryHeadSpec.")
    if not entries:
        empty = HeadScores(
            (),
            np.asarray([], dtype=object),
            np.asarray([], dtype=np.float64),
            float("nan"),
            float("nan"),
            all_entries=(),
            all_scores=np.asarray([], dtype=np.float64),
        )
        return {h.name: empty for h in heads}

    model.eval()
    raw: Dict[str, List[float]] = {h.name: [] for h in heads}
    with torch.no_grad():
        for e in entries:
            probs = _forward_entry_scores(
                model,
                e,
                heads,
                ctx,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            for h in heads:
                raw[h.name].append(probs[h.name])

    return {
        h.name: _head_metrics(entries, np.asarray(raw[h.name], dtype=np.float64), h)
        for h in heads
    }


def eval_splits(
    model: torch.nn.Module,
    *,
    val_entries: Sequence[RunRecord],
    test_entries: Sequence[RunRecord],
    holdout_entries: Sequence[RunRecord],
    heads: Sequence[BinaryHeadSpec],
    ctx: ScoreContext,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> SplitHeadScores:
    return SplitHeadScores(
        val=score_entries(
            model, val_entries, device, heads, ctx,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        ),
        test=score_entries(
            model, test_entries, device, heads, ctx,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        ),
        holdout=score_entries(
            model, holdout_entries, device, heads, ctx,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        ),
    )


def multitask_training_loss_sum_and_count(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ce_weight_cd: Optional[torch.Tensor],
    ce_weight_ct: Optional[torch.Tensor],
    loss_ratio: float,
) -> Tuple[torch.Tensor, int, None]:
    input_ids = batch["input_ids"].to(device)
    bsz, n_set, seq_len = input_ids.shape
    flat_in = input_ids.view(bsz * n_set, seq_len)
    nv = batch["n_sets"]
    cd_labels = batch["label"].to(device)
    ct_labels = batch.get("ct_label")
    if ct_labels is None:
        raise SystemExit("Multitask batch missing ct_label (internal error).")
    ct_labels_t = ct_labels.to(device)
    mask = torch.arange(n_set, device=device).unsqueeze(0) < nv.to(device).unsqueeze(1)
    alpha = float(loss_ratio)

    with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        logits_out = model(flat_in)
        if not isinstance(logits_out, (list, tuple)):
            raise SystemExit("Multitask model forward must return a sequence of logits.")
        logits_cd = logits_out[MT_HEAD_CD].view(bsz, n_set, -1)
        logits_ct = logits_out[MT_HEAD_CT].view(bsz, n_set, -1)

    flat_cd_logits = logits_cd[mask]
    flat_cd_y = cd_labels.unsqueeze(1).expand(bsz, n_set)[mask]
    if flat_cd_logits.numel() == 0:
        return torch.zeros((), device=device), 0, None

    ce_kw_cd: Dict[str, object] = {"reduction": "sum"}
    if ce_weight_cd is not None:
        ce_kw_cd["weight"] = ce_weight_cd.to(
            device=flat_cd_logits.device, dtype=flat_cd_logits.dtype
        )
    cd_ce_sum = F.cross_entropy(flat_cd_logits, flat_cd_y, **ce_kw_cd)
    n_cd = int(flat_cd_y.shape[0])

    flat_ct_logits = logits_ct[mask]
    flat_ct_y = ct_labels_t.unsqueeze(1).expand(bsz, n_set)[mask]
    cancer_mask = flat_ct_y != MT_LABEL_IGNORE
    n_ct = int(cancer_mask.sum().item())
    if n_ct > 0:
        ce_kw_ct: Dict[str, object] = {"reduction": "sum"}
        if ce_weight_ct is not None:
            ce_kw_ct["weight"] = ce_weight_ct.to(
                device=flat_ct_logits.device, dtype=flat_ct_logits.dtype
            )
        ct_ce_sum = F.cross_entropy(
            flat_ct_logits[cancer_mask],
            flat_ct_y[cancer_mask].long(),
            **ce_kw_ct,
        )
        mean_ct = ct_ce_sum / float(n_ct)
    else:
        mean_ct = torch.zeros((), device=device)

    mean_cd = cd_ce_sum / float(n_cd)
    combined = alpha * mean_cd + (1.0 - alpha) * mean_ct
    return combined * float(n_cd), n_cd, None


def tuning_score_single(f1: float, primary_curve: float, metric: str) -> float:
    m = str(metric).strip()
    if m == "auroc":
        v = float(primary_curve)
    elif m == "f1":
        v = float(f1)
    else:
        raise SystemExit(
            f"Unknown tuning_metric {metric!r} (use auroc or f1)."
        )
    return v if v == v else float("-inf")


def tuning_score_from_heads(
    val_scores: Mapping[str, HeadScores],
    *,
    tuning_metric: str,
    tuning_ratio: float,
    cd_name: str = HEAD_CD,
    ct_name: str = HEAD_CT,
) -> float:
    cd = val_scores[cd_name]
    ct = val_scores[ct_name]
    tr = float(tuning_ratio)
    if not 0.0 <= tr <= 1.0:
        raise SystemExit("train_hyenadna.tuning_ratio must be between 0 and 1.")
    if tr == 1.0:
        return tuning_score_single(cd.f1, cd.auroc, tuning_metric)
    if tr == 0.0:
        return tuning_score_single(ct.f1, ct.auroc, tuning_metric)
    s_cd = tuning_score_single(cd.f1, cd.auroc, tuning_metric)
    s_ct = tuning_score_single(ct.f1, ct.auroc, tuning_metric)
    w_cd = tr
    w_ct = 1.0 - tr
    num = 0.0
    den = 0.0
    if w_cd > 0.0 and math.isfinite(s_cd):
        num += w_cd * s_cd
        den += w_cd
    if w_ct > 0.0 and math.isfinite(s_ct):
        num += w_ct * s_ct
        den += w_ct
    if den <= 0.0:
        return float("-inf")
    return num / den


def prepare_multitask_config(
    defaults_path: Path,
) -> Tuple[Any, LabelEncoder, LabelEncoder, TaskTrainingConfig]:
    """Fit encoders and return metadata frame plus training config."""
    run_df = build_multitask_run_table(config_path=defaults_path)
    enc_cd = LabelEncoder()
    enc_ct = LabelEncoder()
    train_df = run_df.loc[run_df["split"] == "train"]
    y_train_cd = train_df["cd_task_label"].to_numpy(dtype=object)
    if y_train_cd.size == 0:
        raise SystemExit("No training runs found after shared split assignment.")
    enc_cd.fit(y_train_cd)
    cancer_train = train_df[train_df["cd_task_label"] == "cancer"]
    y_train_ct = cancer_train["ct_task_label"].to_numpy(dtype=object)
    if y_train_ct.size == 0:
        raise SystemExit("No cancer training runs found for multitask type head.")
    enc_ct.fit(y_train_ct)

    cd_class_names = [str(x) for x in enc_cd.classes_.tolist()]
    ct_class_names = [str(x) for x in enc_ct.classes_.tolist()]
    if len(cd_class_names) < 2 or len(ct_class_names) < 2:
        raise SystemExit("Multitask HyenaDNA training expects 2 classes per head.")

    cd_pos = cd_class_names.index("cancer")
    ct_pos = ct_class_names.index("breast_cancer")
    cd_neg = cd_class_names[1 - cd_pos]
    ct_neg = ct_class_names[1 - ct_pos]

    heads = (
        BinaryHeadSpec(
            name=HEAD_CD,
            head_index=MT_HEAD_CD,
            pos_class_index=cd_pos,
            neg_label=cd_neg,
            pos_label="cancer",
            label_attr="task_label",
            metrics_json_key=METRIC_KEY_CD,
        ),
        BinaryHeadSpec(
            name=HEAD_CT,
            head_index=MT_HEAD_CT,
            pos_class_index=ct_pos,
            neg_label=ct_neg,
            pos_label="breast_cancer",
            label_attr="ct_task_label",
            cancer_only_metrics=True,
            metrics_json_key=METRIC_KEY_CT,
        ),
    )
    cfg = TaskTrainingConfig(
        multitask=True,
        loss_ratio=0.7,
        tuning_ratio=0.5,
        heads=heads,
        primary_head=HEAD_CD,
        class_names=tuple(cd_class_names),
        ct_class_names=tuple(ct_class_names),
    )
    return run_df, enc_cd, enc_ct, cfg


def apply_task_training_overrides(
    cfg: TaskTrainingConfig,
    merged: Mapping[str, object],
) -> TaskTrainingConfig:
    loss_ratio = float(merged.get("loss_ratio", cfg.loss_ratio))
    if not 0.0 <= loss_ratio <= 1.0:
        raise SystemExit("train_hyenadna.loss_ratio must be between 0 and 1.")
    if cfg.multitask:
        tuning_ratio = float(merged.get("tuning_ratio", cfg.tuning_ratio))
        if not 0.0 <= tuning_ratio <= 1.0:
            raise SystemExit("train_hyenadna.tuning_ratio must be between 0 and 1.")
    else:
        tuning_ratio = cfg.tuning_ratio
    return TaskTrainingConfig(
        multitask=cfg.multitask,
        loss_ratio=loss_ratio,
        tuning_ratio=tuning_ratio,
        heads=cfg.heads,
        primary_head=cfg.primary_head,
        class_names=cfg.class_names,
        ct_class_names=cfg.ct_class_names,
    )


def prepare_single_task_config(
    task: str,
    defaults_path: Path,
) -> Tuple[Any, LabelEncoder, TaskTrainingConfig]:
    from shared_utilities import build_run_task_table

    run_df = build_run_task_table(task, config_path=defaults_path)
    enc = LabelEncoder()
    y_train = run_df.loc[run_df["split"] == "train", "task_label"].to_numpy(dtype=object)
    if y_train.size == 0:
        raise SystemExit("No training runs found after shared split assignment.")
    enc.fit(y_train)
    class_names = [str(x) for x in enc.classes_.tolist()]
    if len(class_names) < 2:
        raise SystemExit("HyenaDNA training expects at least 2 task classes.")
    neg_i, pos_i = 0, 1
    head = BinaryHeadSpec(
        name="task",
        head_index=0,
        pos_class_index=pos_i,
        neg_label=class_names[neg_i],
        pos_label=class_names[pos_i],
        label_attr="task_label",
    )
    cfg = TaskTrainingConfig(
        multitask=False,
        loss_ratio=1.0,
        tuning_ratio=1.0,
        heads=(head,),
        primary_head="task",
        class_names=tuple(class_names),
        ct_class_names=(),
    )
    return run_df, enc, cfg


_PREDICTION_CSV_SINGLE = (
    ["Run", "task_label", "predicted_label", "positive_score"],
    {
        "Run": "Run",
        "task_label": "task_label",
        "predicted_label": "predicted_label",
        "positive_score": "positive_score",
    },
)
_PREDICTION_CSV_MULTITASK = (
    [
        "Run",
        "sample_label",
        "cd_task_label",
        "cd_predicted_label",
        "cd_cancer_score",
        "ct_task_label",
        "ct_predicted_label",
        "ct_breast_score",
    ],
    {},
)


def _prediction_rows_single(
    entries: Sequence[RunRecord],
    scores: Mapping[str, HeadScores],
    head: BinaryHeadSpec,
) -> List[Dict[str, object]]:
    hs = scores[head.name]
    use_entries = hs.all_entries if hs.all_entries else tuple(entries)
    use_scores = hs.all_scores if hs.all_scores.size else hs.y_score
    if len(use_entries) != len(use_scores):
        raise SystemExit("Prediction row count mismatch for single-task CSV.")
    rows: List[Dict[str, object]] = []
    for e, score in zip(use_entries, use_scores):
        pred = head.pos_label if float(score) >= 0.5 else head.neg_label
        rows.append(
            {
                "Run": str(e.run),
                "task_label": str(e.task_label),
                "predicted_label": str(pred),
                "positive_score": float(score),
            }
        )
    return rows


def _prediction_rows_multitask(
    entries: Sequence[RunRecord],
    scores: Mapping[str, HeadScores],
    cd: BinaryHeadSpec,
    ct: BinaryHeadSpec,
) -> List[Dict[str, object]]:
    cd_hs = scores[cd.name]
    ct_hs = scores[ct.name]
    use_entries = cd_hs.all_entries if cd_hs.all_entries else tuple(entries)
    cd_scores = cd_hs.all_scores if cd_hs.all_scores.size else cd_hs.y_score
    ct_scores = ct_hs.all_scores if ct_hs.all_scores.size else ct_hs.y_score
    if len(use_entries) != len(cd_scores) or len(use_entries) != len(ct_scores):
        raise SystemExit("Prediction row count mismatch for multitask CSV.")
    rows: List[Dict[str, object]] = []
    for e, cd_s, ct_s in zip(use_entries, cd_scores, ct_scores):
        cd_pred = cd.pos_label if float(cd_s) >= 0.5 else cd.neg_label
        row: Dict[str, object] = {
            "Run": str(e.run),
            "sample_label": str(e.sample_label),
            "cd_task_label": str(e.task_label),
            "cd_predicted_label": str(cd_pred),
            "cd_cancer_score": float(cd_s),
            "ct_task_label": str(e.ct_task_label) if e.is_cancer else "",
            "ct_predicted_label": "",
            "ct_breast_score": float(ct_s),
        }
        if e.is_cancer:
            row["ct_predicted_label"] = (
                ct.pos_label if float(ct_s) >= 0.5 else ct.neg_label
            )
        rows.append(row)
    return rows


def write_predictions_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    multitask: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        _PREDICTION_CSV_MULTITASK[0] if multitask else _PREDICTION_CSV_SINGLE[0]
    )
    float_cols = (
        {"cd_cancer_score", "ct_breast_score"} if multitask else {"positive_score"}
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out_row: Dict[str, str] = {}
            for key in fieldnames:
                if key in float_cols:
                    out_row[key] = f"{float(row[key]):.10f}"
                else:
                    out_row[key] = str(row[key])
            writer.writerow(out_row)


def build_metrics_payload(
    test_scores: Mapping[str, HeadScores],
    holdout_scores: Mapping[str, HeadScores],
    heads: Sequence[BinaryHeadSpec],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for h in heads:
        key = h.metrics_json_key or h.name
        ts = test_scores[h.name]
        hs = holdout_scores[h.name]
        payload[key] = {
            "test": {"auroc": _float_or_none(ts.auroc)},
            "holdout": {"auroc": _float_or_none(hs.auroc)},
        }
    return payload


def build_flat_metrics_payload(
    test_scores: Mapping[str, HeadScores],
    holdout_scores: Mapping[str, HeadScores],
    head_name: str,
) -> Dict[str, Any]:
    ts = test_scores[head_name]
    hs = holdout_scores[head_name]
    return {
        "test": {"auroc": _float_or_none(ts.auroc)},
        "holdout": {"auroc": _float_or_none(hs.auroc)},
    }


def write_run_artifacts(
    results_path: Path,
    *,
    merged: Mapping[str, Any],
    cache_info: Mapping[str, Any],
    task_cfg: TaskTrainingConfig,
    test_entries: Sequence[RunRecord],
    holdout_entries: Sequence[RunRecord],
    test_scores: Mapping[str, HeadScores],
    holdout_scores: Mapping[str, HeadScores],
    best_epoch: int,
    tuning_metric: str,
    best_tuning_score: float,
    best_val_scores: Mapping[str, HeadScores],
    write_results_json: Callable[..., None],
) -> None:
    """Write results JSON and split prediction CSVs."""
    tuning_extra: Optional[Dict[str, Any]] = None
    if task_cfg.multitask:
        metrics_payload = build_metrics_payload(test_scores, holdout_scores, task_cfg.heads)
        cd = best_val_scores[HEAD_CD]
        ct = best_val_scores[HEAD_CT]
        tuning_extra = {
            "cancer_diagnosis_score": _float_or_none(
                tuning_score_single(cd.f1, cd.auroc, tuning_metric)
            ),
            "cancer_type_score": _float_or_none(
                tuning_score_single(ct.f1, ct.auroc, tuning_metric)
            ),
            "combined_score": _float_or_none(float(best_tuning_score)),
        }
        pred_rows_test = _prediction_rows_multitask(
            test_entries, test_scores, task_cfg.heads[0], task_cfg.heads[1]
        )
        pred_rows_hold = _prediction_rows_multitask(
            holdout_entries, holdout_scores, task_cfg.heads[0], task_cfg.heads[1]
        )
        final_test_cd_curve = test_scores[HEAD_CD].auroc
        final_holdout_cd_curve = holdout_scores[HEAD_CD].auroc
        final_test_ct_curve = test_scores[HEAD_CT].auroc
        final_holdout_ct_curve = holdout_scores[HEAD_CT].auroc
    else:
        head = task_cfg.heads[0]
        metrics_payload = build_flat_metrics_payload(test_scores, holdout_scores, head.name)
        hv = best_val_scores[head.name]
        task_key = str(merged.get("task") or "").strip()
        head_score = _float_or_none(
            tuning_score_single(hv.f1, hv.auroc, tuning_metric)
        )
        if task_key == "cancer_type":
            tuning_extra = {
                "cancer_diagnosis_score": None,
                "cancer_type_score": head_score,
                "combined_score": _float_or_none(float(best_tuning_score)),
            }
        else:
            tuning_extra = {
                "cancer_diagnosis_score": head_score,
                "cancer_type_score": None,
                "combined_score": _float_or_none(float(best_tuning_score)),
            }
        pred_rows_test = _prediction_rows_single(test_entries, test_scores, head)
        hold_entries = tuple(holdout_entries)[: len(holdout_scores[head.name].y_score)]
        pred_rows_hold = _prediction_rows_single(hold_entries, holdout_scores, head)
        final_test_cd_curve = test_scores[head.name].auroc
        final_holdout_cd_curve = holdout_scores[head.name].auroc
        final_test_ct_curve = float("nan")
        final_holdout_ct_curve = float("nan")

    test_pred_path = results_path.with_name(f"{results_path.stem}_test.csv")
    holdout_pred_path = results_path.with_name(f"{results_path.stem}_holdout.csv")
    write_predictions_csv(test_pred_path, pred_rows_test, multitask=task_cfg.multitask)
    write_predictions_csv(holdout_pred_path, pred_rows_hold, multitask=task_cfg.multitask)

    write_results_json(
        results_path,
        merged=merged,
        cache_info=cache_info,
        test_primary_curve=final_test_cd_curve,
        holdout_primary_curve=final_holdout_cd_curve,
        best_epoch=best_epoch,
        tuning_metric=tuning_metric,
        best_tuning_score=best_tuning_score,
        metrics=metrics_payload,
        tuning_extra=tuning_extra,
    )

    if task_cfg.multitask:
        print(
            f"Final eval (best epoch by tuning_ratio={task_cfg.tuning_ratio:.3f} "
            f"with {tuning_metric}): "
            f"best_epoch={best_epoch} "
            f"test_auroc_cd={final_test_cd_curve:.4f} "
            f"test_auroc_ct={final_test_ct_curve:.4f} "
            f"holdout_auroc_cd={final_holdout_cd_curve:.4f} "
            f"holdout_auroc_ct={final_holdout_ct_curve:.4f}\n",
            flush=True,
        )
    else:
        print(
            f"Final eval (best epoch by {tuning_metric}): best_epoch={best_epoch} "
            f"test_auroc={final_test_cd_curve:.4f} "
            f"holdout_auroc={final_holdout_cd_curve:.4f}\n",
            flush=True,
        )


def epoch_progress_fields(
    *,
    multitask: bool,
    split_eval: SplitHeadScores,
    train_loss: float,
    val_loss: float,
    best_epoch: int,
) -> List[str]:
    fields = [f"train_loss={train_loss:.4f}", f"val_loss={val_loss:.4f}"]
    if multitask:
        vcd = split_eval.val[HEAD_CD]
        vct = split_eval.val[HEAD_CT]
        tcd = split_eval.test[HEAD_CD]
        tct = split_eval.test[HEAD_CT]
        hcd = split_eval.holdout[HEAD_CD]
        hct = split_eval.holdout[HEAD_CT]
        fields.extend(
            [
                f"val_auroc_cd={vcd.auroc:.4f}",
                f"val_auroc_ct={vct.auroc:.4f}",
                f"test_auroc_cd={tcd.auroc:.4f}",
                f"test_auroc_ct={tct.auroc:.4f}",
                f"holdout_auroc_cd={hcd.auroc:.4f}",
                f"holdout_auroc_ct={hct.auroc:.4f}",
            ]
        )
    else:
        v = split_eval.val["task"]
        t = split_eval.test["task"]
        h = split_eval.holdout["task"]
        fields.extend(
            [
                f"val_auroc={v.auroc:.4f}",
                f"test_auroc={t.auroc:.4f}",
                f"holdout_auroc={h.auroc:.4f}",
            ]
        )
    fields.append(f"best_epoch={int(best_epoch)}")
    return fields
