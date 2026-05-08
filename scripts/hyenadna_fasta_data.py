#!/usr/bin/env python3
"""Shared FASTA → token batch helpers for HyenaDNA dataset build and training."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from hyenadna import CharacterTokenizer

# ----- Match scripts/train_hyenadna legacy MODEL_CONFIGS (max sequence positions) -----

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "hyenadna-tiny-1k-seqlen": {"max_length": 1024},
    "hyenadna-small-32k-seqlen": {"max_length": 32768},
    "hyenadna-medium-160k-seqlen": {"max_length": 160000},
    "hyenadna-medium-450k-seqlen": {"max_length": 450000},
    "hyenadna-large-1m-seqlen": {"max_length": 1_000_000},
}

SEP_TOKEN_ID = 1
PAD_TOKEN_ID = 4


def resolve_repo_path(repo_root: Path, raw: object) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


def load_train_hyenadna_section(defaults_path: Path) -> Dict[str, object]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    sec = cfg.get("train_hyenadna")
    if not isinstance(sec, dict):
        raise SystemExit(f"{defaults_path} must define a train_hyenadna mapping.")
    return dict(sec)


def merge_train_hyenadna_config(
    defaults_path: Path,
    experiments_path: Path,
    *,
    expt: int,
) -> Tuple[Dict[str, object], Optional[str], Optional[str]]:
    """Return merged train_hyenadna config, optional experiment name, results_json template.
    """
    defaults = load_train_hyenadna_section(defaults_path)
    experiment_name: Optional[str] = None
    template: Optional[str] = None
    experiments_cfg: Dict[str, object] = {}
    if experiments_path.is_file():
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8")) or {}
    section = experiments_cfg.get("train_hyenadna") or {}
    if section and not isinstance(section, dict):
        raise SystemExit("experiments.yaml train_hyenadna must be a mapping when present.")
    if isinstance(section, dict):
        raw_tpl = section.get("results_json_template")
        template = str(raw_tpl).strip() if isinstance(raw_tpl, str) else None
        experiments = section.get("experiments") or []
        if not isinstance(experiments, list):
            raise SystemExit("train_hyenadna.experiments must be a list when present.")
        if expt == 0:
            selected = {}
        else:
            if not experiments:
                raise SystemExit("No train_hyenadna.experiments in experiments.yaml.")
            if expt > len(experiments):
                raise SystemExit(
                    f"--expt {expt} out of range; experiments.yaml has {len(experiments)} rows."
                )
            row = experiments[expt - 1]
            if not isinstance(row, dict):
                raise SystemExit("train_hyenadna experiment entry must be a mapping.")
            experiment_name = str(row.get("name") or "").strip() or None
            overrides = row.get("overrides") or {}
            if not isinstance(overrides, dict):
                raise SystemExit("experiment overrides must be a mapping.")
            selected = dict(overrides)
        defaults = {**defaults, **selected}
    elif expt != 0:
        raise SystemExit(
            f"--expt {expt} requires experiments.yaml with a train_hyenadna.experiments list."
        )

    if (
        expt != 0
        and experiment_name
        and template
        and defaults.get("results_json") in (None, "null")
    ):
        defaults["results_json"] = template.format(
            name=experiment_name,
        )

    return defaults, experiment_name, template


def model_max_length(model_name: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    if model_name not in MODEL_CONFIGS:
        raise SystemExit(
            f"Unknown HyenaDNA model {model_name!r}. Choose from {sorted(MODEL_CONFIGS)} "
            "or set train_hyenadna.max_length."
        )
    return int(MODEL_CONFIGS[model_name]["max_length"])


def make_character_tokenizer(max_length: int) -> CharacterTokenizer:
    return CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=max_length + 2,
        padding_side="left",
    )


def iter_fasta_sequences(gzip_path: Path) -> Iterable[str]:
    """Yield sequence strings from a .fasta.gz (same convention as tetramer pipeline)."""
    with gzip.open(gzip_path, "rt", encoding="ascii", errors="replace") as handle:
        chunks: List[str] = []
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    yield "".join(chunks)
                    chunks = []
                continue
            chunks.append(line)
        if chunks:
            yield "".join(chunks)


def fasta_path_for_run(repo_root: Path, fasta_dir_key: str, study_name: str, run: str) -> Path:
    base = resolve_repo_path(repo_root, fasta_dir_key)
    return base / study_name / f"{run}.fasta.gz"


def split_sequences_into_sets(
    sequences: List[str],
    max_length: int,
    num_sets: int,
) -> List[List[str]]:
    """Split sequences into non-overlapping sets, each fitting within max_length (char budget)."""
    if not sequences:
        return [[] for _ in range(num_sets)]

    valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]
    if not valid_sequences:
        return [[] for _ in range(num_sets)]

    sets: List[List[str]] = [[] for _ in range(num_sets)]
    current_set_idx = 0
    sequence_idx = 0

    while sequence_idx < len(valid_sequences) and current_set_idx < num_sets:
        current_set = sets[current_set_idx]
        cumulative_length = sum(len(seq) for seq in current_set)

        while sequence_idx < len(valid_sequences):
            seq = valid_sequences[sequence_idx]
            seq_len = len(seq)
            tokens_needed = seq_len + (1 if current_set else 0)

            if cumulative_length + tokens_needed <= max_length:
                current_set.append(seq)
                cumulative_length += tokens_needed
                sequence_idx += 1
            else:
                break

        current_set_idx += 1
        if sequence_idx >= len(valid_sequences):
            break

    return sets


def concatenate_sequences(
    sequences: List[str],
    tokenizer: CharacterTokenizer,
    max_length: int,
    sep_token_id: int = SEP_TOKEN_ID,
) -> List[int]:
    """Concatenate DNA strings with [SEP]; return token ids of length <= max_length."""
    if not sequences:
        return [sep_token_id]

    valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]
    if not valid_sequences:
        return [sep_token_id]

    cumulative_length = 0
    sequences_to_use: List[str] = []

    for seq in valid_sequences:
        seq_len = len(seq)
        tokens_needed = seq_len + (1 if sequences_to_use else 0)
        if cumulative_length + tokens_needed <= max_length:
            sequences_to_use.append(seq)
            cumulative_length += tokens_needed
        else:
            break

    if not sequences_to_use:
        sequences_to_use = [valid_sequences[0]]

    tokenized_seqs: List[List[int]] = []
    for seq in sequences_to_use:
        tokens = tokenizer(seq)["input_ids"]
        dna_tokens = [t for t in tokens if t >= 7]
        if dna_tokens:
            tokenized_seqs.append(dna_tokens)

    if not tokenized_seqs:
        return [sep_token_id]

    result: List[int] = []
    for i, seq_tokens in enumerate(tokenized_seqs):
        if i > 0:
            result.append(sep_token_id)
        result.extend(seq_tokens)

    if len(result) > max_length:
        result = result[-max_length:]

    return result


def pad_left(token_ids: List[int], *, pad_id: int, target_len: int) -> Tuple[List[int], List[int]]:
    pad_len = target_len - len(token_ids)
    if pad_len < 0:
        raise ValueError("target_len shorter than sequence")
    padded = [pad_id] * pad_len + token_ids
    mask = [0] * pad_len + [1] * len(token_ids)
    return padded, mask


def run_to_tensors(
    sequences: List[str],
    *,
    tokenizer: CharacterTokenizer,
    max_length: int,
    num_sets: int,
) -> Tuple[Optional["object"], Optional["object"], int]:
    """Return (input_ids [num_sets,L], attention_mask [num_sets,L], n_valid_sets) or Nones if empty."""
    import torch

    sets = split_sequences_into_sets(sequences, max_length, num_sets)
    rows_ids: List[List[int]] = []
    for seq_set in sets:
        if not seq_set:
            continue
        tid = concatenate_sequences(seq_set, tokenizer, max_length, SEP_TOKEN_ID)
        rows_ids.append(tid)

    n_valid = len(rows_ids)
    if n_valid == 0:
        return None, None, 0

    target_len = max_length
    padded_ids: List[List[int]] = []
    padded_masks: List[List[int]] = []
    for tid in rows_ids:
        if len(tid) > max_length:
            tid = tid[-max_length:]
        p, m = pad_left(tid, pad_id=PAD_TOKEN_ID, target_len=target_len)
        padded_ids.append(p)
        padded_masks.append(m)

    while len(padded_ids) < num_sets:
        padded_ids.append([PAD_TOKEN_ID] * target_len)
        padded_masks.append([0] * target_len)

    return (
        torch.LongTensor(padded_ids),
        torch.LongTensor(padded_masks),
        n_valid,
    )
