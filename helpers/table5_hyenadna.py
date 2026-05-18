#!/usr/bin/env python3
"""
Build Table 5 (HyenaDNA ablations) as HTML under manuscript/table5_hyenadna.html.

Reads ``experiments.yaml`` ``train_hyenadna.experiments`` (row order preserved)
and aggregates tuning epoch plus ``metrics.test.auroc`` /
``metrics.holdout.auroc`` from per-seed JSON under ``results/hyenadna/``.

Single-task runs use ``cd_*`` / ``ct_*`` filename prefixes with flat
``metrics`` objects. Multitask runs use the ``mt_*`` prefix; each file nests
metrics under ``cancer_diagnosis`` and ``cancer_type``.

Consecutive YAML rows dedicated to diagnosis-only and type-only tasks are
paired into **one table row** (diagnosis metrics from the first available slot,
type metrics from the other). Multitask rows occupy a full row. A YAML row with
a YAML task list ``[cancer_diagnosis, cancer_type]`` (grid, not multitask)
also fills both sides from ``cd_*`` and ``ct_*`` files sharing the experiment
``name``.

YAML ``random_seed`` lists the intended seeds; partial result trees are aggregated
(no hard exit for missing seeds). Diagnostics use a fixed two-section layout
(seeds vs ablations) printed to stderr after a successful run when needed,
or raised as ``SystemExit`` when ablations are missing. Seed mismatches use a
``Warning:`` prefix; missing ablations use an ``Error:`` prefix.

Run from the repository root: ``python helpers/table5_hyenadna.py``
"""

from __future__ import annotations

import html
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

import yaml


# Plain-English row labels keyed by ``train_hyenadna.experiments[].name``.
# Row order follows ``experiments.yaml``. Descriptions may contain HTML (e.g.
# <sup>) and are intentionally not escaped when rendered into the <td>.
ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "best_recipe": "Best recipe (baseline)",
    "high_lr": "High learning rate (5e-4 instead of 2e-4)",
    "dropout_0.2": "Add dropout to MLP classification head (0.2)",
    "hidden_256": "MLP hidden layer width 256 (instead of 512)",
    "unfrozen_backbone": "Unfrozen backbone (lr 2e-4)",
    "unfrozen_backbone_low_lr": "Unfrozen backbone (lr 1e-5)",
    #"no_dropout": "No dropout in classification heads",
    #"hidden_64": "MLP with 64-width hidden layer",
    #"low_lr": "Low learning rate (1e-5 instead of 2e-4)",
    #"no_study_balanced": "Random training sampler (no study-balanced sampling)",
    #"class_balanced": "Balanced class weighting",
    #"head_linear": "Linear classification head instead of MLP",
}
TASK_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("cd", "Cancer diagnosis"),
    ("ct", "Cancer type"),
)

METRIC_KEY_CD = "cancer_diagnosis"
METRIC_KEY_CT = "cancer_type"
MULTITASK_TASK = "multitask"

DECIMALS = 3
OUTPUT_REL = Path("manuscript") / "table5_hyenadna.html"
MSG_SEEDS_HEADER = "Warning: Missing results for some seeds in experiments.yaml:"
MSG_ABLATIONS_HEADER = "Error: Missing results for ablations:"


@dataclass
class GapRecorder:
    """Collects YAML seed mismatches vs on-disk metrics JSON (filled during aggregation)."""

    gaps: List["SeedCoverageGap"] = field(default_factory=list)


@dataclass(frozen=True)
class SeedCoverageGap:
    label: str
    missing: Tuple[int, ...]
    stray_on_disk: Tuple[int, ...]


def _record_seed_coverage(rec: GapRecorder, gap: SeedCoverageGap) -> None:
    """Record discrepancies once per aggregated result group (caller filters trivial cases)."""
    if not gap.missing and not gap.stray_on_disk:
        return
    rec.gaps.append(gap)


def _gap_from_sets(
    label: str,
    expected_seeds: Set[int],
    *,
    found_on_disk: Set[int],
    seeds_used: Set[int],
) -> SeedCoverageGap:
    miss = tuple(sorted(expected_seeds - seeds_used))
    stray = tuple(sorted(found_on_disk - expected_seeds))
    return SeedCoverageGap(label=label, missing=miss, stray_on_disk=stray)


def _select_seed_files(
    label: str,
    files: List[Tuple[int, Path]],
    expected_seeds: Set[int],
    *,
    recorder: GapRecorder,
) -> List[Tuple[int, Path]]:
    """Keep YAML-listed seeds when configured; gaps go to recorder (no stderr here)."""
    if not files:
        return []
    found_on_disk = {seed for seed, _ in files}
    ordered = sorted(files, key=lambda x: x[0])
    if not expected_seeds:
        return ordered
    selected = [(s, p) for s, p in ordered if s in expected_seeds]
    if not selected:
        seeds_used = found_on_disk
        _record_seed_coverage(
            rec=recorder,
            gap=_gap_from_sets(label, expected_seeds, found_on_disk=found_on_disk, seeds_used=seeds_used),
        )
        return ordered
    seeds_used = {seed for seed, _ in selected}
    g = _gap_from_sets(label, expected_seeds, found_on_disk=found_on_disk, seeds_used=seeds_used)
    if g.missing or g.stray_on_disk:
        _record_seed_coverage(recorder, g)
    return selected


def _format_problem_report(recorder: GapRecorder, *, missing_ablations: List[str]) -> str:
    """Human-readable diagnostics (no trailing path breadcrumb)."""
    blocks: List[str] = []
    gap_lines = _seed_gap_summary_lines(recorder.gaps)
    if gap_lines:
        blocks.append(MSG_SEEDS_HEADER + "\n" + "\n".join(gap_lines))
    if missing_ablations:
        miss = "\n".join(f"  - {m}" for m in missing_ablations)
        blocks.append(MSG_ABLATIONS_HEADER + "\n" + miss)
    return "\n".join(blocks)


def _seed_gap_summary_lines(gaps: Sequence[SeedCoverageGap]) -> List[str]:
    """One bullet per aggregation label."""
    lines: List[str] = []
    seen: Set[str] = set()
    for g in gaps:
        key = g.label
        if key in seen:
            continue
        seen.add(key)
        parts: List[str] = []
        if g.missing:
            missing_list = "[" + ", ".join(str(i) for i in g.missing) + "]"
            parts.append(f"missing seeds {missing_list}")
        if g.stray_on_disk:
            stray_list = "[" + ", ".join(str(i) for i in g.stray_on_disk) + "]"
            parts.append(f"ignoring on-disk seeds {stray_list}")
        if not parts:
            continue
        lines.append(f"  - {g.label} ({'; '.join(parts)})")
    return lines


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _train_hyenadna_base(repo_root: Path) -> dict:
    cfg = _load_yaml(repo_root / "defaults.yaml")
    sec = cfg.get("train_hyenadna")
    if not isinstance(sec, dict):
        raise SystemExit("defaults.yaml must define train_hyenadna as a mapping.")
    return dict(sec)


def _parse_seed_spec(raw: object) -> Set[int]:
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    if isinstance(raw, (list, tuple)):
        try:
            return {int(x) for x in raw}
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"random_seed must be a YAML list of integers; got {raw!r}."
            ) from exc
    raise SystemExit(
        "random_seed must be an integer or a YAML list of integers; "
        f"got {type(raw).__name__}."
    )


def _experiment_task_mode(merged_task: object) -> str:
    """
    Return layout for one YAML experiment row:

    multitask_file   — mt_*.json, nested metrics keys
    dual_file        — cd_*.json + ct_*.json (comma-separated diagnosis+type grid)
    cancer_diagnosis — cd_* only
    cancer_type      — ct_* only
    """
    if isinstance(merged_task, (list, tuple)):
        tokens = [str(tok).strip().lower() for tok in merged_task if str(tok).strip()]
    else:
        task = str(merged_task or "").strip()
        if not task:
            tokens = []
        elif task.lower() == MULTITASK_TASK:
            tokens = [MULTITASK_TASK]
        else:
            tokens = [task.lower()]

    def _canonical_one(tok: str) -> Optional[str]:
        if tok == "multitask":
            return MULTITASK_TASK
        if tok == "cancer_diagnosis":
            return "cancer_diagnosis"
        if tok == "cancer_type":
            return "cancer_type"
        return None

    seen: Set[str] = set()
    for t in tokens:
        c = _canonical_one(t)
        if c is None:
            raise SystemExit(f"Unsupported train_hyenadna task token {t!r}.")
        if c == MULTITASK_TASK:
            if len(tokens) > 1:
                raise SystemExit(
                    "Multitask must not be combined with other tasks in one experiment row; "
                    "use task: multitask alone."
                )
            return "multitask_file"
        seen.add(c)
    uniq = tuple(sorted(seen))
    if not uniq:
        return "cancer_diagnosis"
    if len(uniq) == 1:
        return str(uniq[0])
    if uniq == ("cancer_diagnosis", "cancer_type"):
        return "dual_file"
    raise SystemExit(f"Unsupported combined task specification {tokens!r}.")


def _experiment_rows(repo_root: Path) -> List[Dict[str, object]]:
    """One dict per experiments.yaml train_hyenadna row (YAML order)."""
    base = _train_hyenadna_base(repo_root)
    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_hyenadna") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit(
            "experiments.yaml train_hyenadna.experiments must be a non-empty list."
        )
    out: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"experiments.yaml row {idx + 1} is not a mapping.")
        name = str(row.get("name") or "").strip()
        if "max_length" in name:
            continue
        if name not in ABLATION_DESCRIPTIONS:
            raise SystemExit(
                f"experiments.yaml row {idx + 1}: unknown experiment name {name!r}; "
                "add a description to ABLATION_DESCRIPTIONS in helpers/table5_hyenadna.py."
            )
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit(f"{name}: overrides must be a mapping.")
        merged = {**base, **overrides}
        try:
            max_length = int(merged["max_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{name}: cannot resolve max_length from defaults+overrides ({exc})."
            )
        expected_seeds = _parse_seed_spec(merged.get("random_seed"))
        task_mode = _experiment_task_mode(merged.get("task"))
        out.append(
            {
                "name": name,
                "description": ABLATION_DESCRIPTIONS[name],
                "task_mode": task_mode,
                "max_length": max_length,
                "expected_seeds": expected_seeds,
            }
        )
    return out


def _len_token(max_length: int) -> str:
    if max_length <= 0 or max_length % 1024 != 0:
        raise SystemExit(
            f"max_length {max_length} is not a positive multiple of 1024; "
            "cannot reconstruct results JSON filename."
        )
    return f"{max_length // 1024}k"


_SEED_RE = re.compile(r"_s(\d+)\.json$")


def _seed_files(
    hyenadna_dir: Path, task_abbrv: str, name: str, max_length: int
) -> List[Tuple[int, Path]]:
    tok = _len_token(max_length)
    candidates = sorted(hyenadna_dir.glob(f"{task_abbrv}_{name}_{tok}_s*.json"))
    out: List[Tuple[int, Path]] = []
    for p in candidates:
        if p.name.endswith("_training.json"):
            continue
        if "max_length" in p.name:
            continue
        m = _SEED_RE.search(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


TaskMetricsCols = Dict[str, Dict[str, List[float]]]


def _read_flat_metrics(path: Path) -> Tuple[int, float, float]:
    """Return (best_epoch, test_auroc, holdout_auroc) from one single-task JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")

    tuning = data.get("tuning") or {}
    best_epoch = tuning.get("best_epoch")
    if not isinstance(best_epoch, int):
        raise SystemExit(f"{path}: missing or non-integer tuning.best_epoch.")

    metrics = data.get("metrics") or {}
    for split in ("test", "holdout"):
        blob = metrics.get(split)
        if not isinstance(blob, dict):
            raise SystemExit(
                f"{path}: expected metrics['{split}'] to be an object with 'auroc'."
            )
    test_v = metrics["test"].get("auroc")
    hold_v = metrics["holdout"].get("auroc")
    if test_v is None or hold_v is None:
        raise SystemExit(f"{path}: missing test or holdout auroc.")
    t = float(test_v)
    h = float(hold_v)
    if not math.isfinite(t) or not math.isfinite(h):
        raise SystemExit(f"{path}: non-finite test/holdout auroc ({t}, {h}).")
    return int(best_epoch), t, h


def _read_multitask_metrics(path: Path) -> Tuple[int, Tuple[float, float], Tuple[float, float]]:
    """Return (epoch, (cd_test, cd_hold), (ct_test, ct_hold)) from multitask JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")
    tuning = data.get("tuning") or {}
    best_epoch = tuning.get("best_epoch")
    if not isinstance(best_epoch, int):
        raise SystemExit(f"{path}: missing or non-integer tuning.best_epoch.")
    metrics = data.get("metrics") or {}

    def _pair(task_key: str) -> Tuple[float, float]:
        blob = metrics.get(task_key)
        if not isinstance(blob, dict):
            raise SystemExit(
                f"{path}: missing metrics subtree {task_key!r} "
                "(expected multitask layout)."
            )
        for split in ("test", "holdout"):
            if not isinstance(blob.get(split), dict):
                raise SystemExit(f"{path}: metrics[{task_key!r}][{split!r}] not an object.")
        test_v = blob["test"].get("auroc")
        hold_v = blob["holdout"].get("auroc")
        if test_v is None or hold_v is None:
            raise SystemExit(f"{path}: missing auroc for {task_key}.")
        t, h = float(test_v), float(hold_v)
        if not math.isfinite(t) or not math.isfinite(h):
            raise SystemExit(f"{path}: non-finite AUROC under {task_key}.")
        return t, h

    cd_pair = _pair(METRIC_KEY_CD)
    ct_pair = _pair(METRIC_KEY_CT)
    return int(best_epoch), cd_pair, ct_pair


def _aggregate_seed_files_flat(
    label: str,
    files: List[Tuple[int, Path]],
    expected_seeds: Set[int],
    *,
    recorder: GapRecorder,
) -> Dict[str, List[float]]:
    use_files = _select_seed_files(label, files, expected_seeds, recorder=recorder)
    if not use_files:
        raise RuntimeError(label)
    best_epochs: List[int] = []
    test_aucs: List[float] = []
    hold_aucs: List[float] = []
    for _seed, path in use_files:
        ep, t, h = _read_flat_metrics(path)
        best_epochs.append(ep)
        test_aucs.append(t)
        hold_aucs.append(h)
    return {
        "best_epochs": best_epochs,
        "test_aucs": test_aucs,
        "hold_aucs": hold_aucs,
    }


def _aggregate_mt_files(
    label: str,
    files: List[Tuple[int, Path]],
    expected_seeds: Set[int],
    *,
    recorder: GapRecorder,
) -> TaskMetricsCols:
    use_files = _select_seed_files(label, files, expected_seeds, recorder=recorder)
    if not use_files:
        raise RuntimeError(label)
    cd_ep: List[int] = []
    ct_ep: List[int] = []
    cd_test: List[float] = []
    cd_hold: List[float] = []
    ct_test: List[float] = []
    ct_hold: List[float] = []
    for _seed, path in use_files:
        ep, (t_cd, h_cd), (t_ct, h_ct) = _read_multitask_metrics(path)
        cd_ep.append(ep)
        ct_ep.append(ep)
        cd_test.append(t_cd)
        cd_hold.append(h_cd)
        ct_test.append(t_ct)
        ct_hold.append(h_ct)
    # Same checkpoint picks one epoch column; duplication keeps median-of-epochs stable.
    return {
        "cd": {"best_epochs": cd_ep, "test_aucs": cd_test, "hold_aucs": cd_hold},
        "ct": {"best_epochs": ct_ep, "test_aucs": ct_test, "hold_aucs": ct_hold},
    }


def _load_task_cols_for_spec(
    hyenadna_dir: Path,
    spec: Dict[str, object],
    *,
    recorder: GapRecorder,
) -> Optional[TaskMetricsCols]:
    name = str(spec["name"])
    max_length = int(spec["max_length"])
    seeds = spec["expected_seeds"]
    mode = str(spec["task_mode"])

    if mode == "multitask_file":
        files_mt = _seed_files(hyenadna_dir, "mt", name, max_length)
        if not files_mt:
            return None
        return _aggregate_mt_files(f"mt_{name}", files_mt, set(seeds), recorder=recorder)  # type: ignore[arg-type]

    cols: TaskMetricsCols = {}

    need_cd = mode in ("cancer_diagnosis", "dual_file")
    need_ct = mode in ("cancer_type", "dual_file")

    if need_cd:
        files_cd = _seed_files(hyenadna_dir, "cd", name, max_length)
        if files_cd:
            cols["cd"] = _aggregate_seed_files_flat(f"cd_{name}", files_cd, set(seeds), recorder=recorder)  # type: ignore[arg-type]
    if need_ct:
        files_ct = _seed_files(hyenadna_dir, "ct", name, max_length)
        if files_ct:
            cols["ct"] = _aggregate_seed_files_flat(f"ct_{name}", files_ct, set(seeds), recorder=recorder)  # type: ignore[arg-type]

    if mode == "dual_file":
        if "cd" not in cols or "ct" not in cols:
            return None
        return cols
    if need_cd and "cd" not in cols:
        return None
    if need_ct and "ct" not in cols:
        return None
    return cols


def _empty_task_metrics() -> Dict[str, List[float]]:
    return {"best_epochs": [], "test_aucs": [], "hold_aucs": []}


def collect_table_rows(
    hyenadna_dir: Path,
    experiment_rows: List[Dict[str, object]],
    recorder: GapRecorder,
) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    missing: List[str] = []

    pending_cd: Optional[Dict[str, object]] = None
    pending_cd_metrics: Optional[Dict[str, List[float]]] = None

    def flush_cd() -> None:
        nonlocal pending_cd, pending_cd_metrics
        if pending_cd is None:
            return
        name = str(pending_cd["name"])
        tm_cd = pending_cd_metrics or _empty_task_metrics()
        tm_ct = _empty_task_metrics()
        rows_out.append(
            {
                "label_html": html.escape(name),
                "description_html_pair": (
                    pending_cd["description"],  # raw HTML fragment
                    None,
                ),
                "task_metrics": {
                    "cd": tm_cd,
                    "ct": tm_ct,
                },
            }
        )
        pending_cd = None
        pending_cd_metrics = None

    def flush_ct_alone(bundle: Dict[str, object], m_ct: Dict[str, List[float]]) -> None:
        name = str(bundle["name"])
        rows_out.append(
            {
                "label_html": html.escape(name),
                "description_html_pair": (bundle["description"], None),
                "task_metrics": {
                    "cd": _empty_task_metrics(),
                    "ct": m_ct,
                },
            }
        )

    i = 0
    while i < len(experiment_rows):
        spec = experiment_rows[i]
        mode = str(spec["task_mode"])
        label = html.escape(str(spec["name"]))

        if mode == "multitask_file":
            flush_cd()
            cols = _load_task_cols_for_spec(hyenadna_dir, cast(Dict[str, object], spec), recorder=recorder)
            if cols is None:
                missing.append(f"mt_{spec['name']} ({_len_token(int(spec['max_length']))})")
                i += 1
                continue
            rows_out.append(
                {
                    "label_html": label,
                    "description_html_pair": (spec["description"], None),
                    "task_metrics": cols,
                }
            )
            i += 1
            continue

        if mode == "dual_file":
            flush_cd()
            cols = _load_task_cols_for_spec(hyenadna_dir, cast(Dict[str, object], spec), recorder=recorder)
            if cols is None:
                mn = spec["name"]
                missing.append(
                    f"cd_and_ct_{mn} ({_len_token(int(spec['max_length']))})"
                )
            else:
                rows_out.append(
                    {
                        "label_html": label,
                        "description_html_pair": (spec["description"], None),
                        "task_metrics": cols,
                    }
                )
            i += 1
            continue

        if mode == "cancer_diagnosis":
            cols = _load_task_cols_for_spec(hyenadna_dir, cast(Dict[str, object], spec), recorder=recorder)
            tm_cd_data = cols.get("cd") if cols else None
            if tm_cd_data is None:
                flush_cd()
                missing.append(f"cd_{spec['name']} ({_len_token(int(spec['max_length']))})")
                i += 1
                continue
            merged_cd = tm_cd_data
            if pending_cd is not None:
                flush_cd()
            pending_cd = spec
            pending_cd_metrics = merged_cd
            i += 1
            continue

        if mode == "cancer_type":
            cols = _load_task_cols_for_spec(hyenadna_dir, cast(Dict[str, object], spec), recorder=recorder)
            tm_ct = cols.get("ct") if cols else None
            if tm_ct is None:
                flush_cd()
                missing.append(f"ct_{spec['name']} ({_len_token(int(spec['max_length']))})")
                i += 1
                continue

            if pending_cd is None:
                flush_ct_alone(spec, tm_ct)
            else:
                cd_bundle = pending_cd
                cd_desc_raw = cd_bundle["description"]
                name_cd = cd_bundle["name"]
                name_ct = spec["name"]
                paired_label = (
                    f"{html.escape(str(name_cd))} / {html.escape(str(name_ct))}"
                )
                tm_cd_eff = pending_cd_metrics or _empty_task_metrics()
                tm_ct_eff = tm_ct
                rows_out.append(
                    {
                        "paired_names_html": paired_label,
                        "description_html_pair": (cd_desc_raw, None),
                        "task_metrics": {"cd": tm_cd_eff, "ct": tm_ct_eff},
                    }
                )
                pending_cd = None
                pending_cd_metrics = None
            i += 1
            continue

        raise SystemExit(f"Internal error: unknown task_mode {mode!r}.")

    flush_cd()

    if missing:
        raise SystemExit(_format_problem_report(recorder, missing_ablations=missing))
    return rows_out


def _mean_std(values: List[float]) -> Tuple[float, Optional[float]]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def _fmt_mean_std(values: List[float], *, decimals: int) -> str:
    if not values:
        return "\u2014"  # em dash
    mean, std = _mean_std(values)
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


def _fmt_median_epoch(values: List[int]) -> str:
    if not values:
        return "\u2014"  # em dash
    med = statistics.median(values)
    if float(med).is_integer():
        return str(int(med))
    return f"{med:.1f}"


def format_table_html(
    rows: Sequence[Dict[str, object]],
    *,
    decimals: int,
) -> str:
    """Render HTML. Rows carry prose (and optional pairing) plus ``task_metrics``."""

    thead = (
        "<thead>\n"
        "<tr>\n"
        "<th rowspan='2'>Ablation</th>"
        "<th colspan='3'>Cancer diagnosis</th>"
        "<th colspan='3'>Cancer type</th>\n"
        "</tr>\n"
        "<tr>\n"
        "<th>Epoch</th>"
        "<th>Test</th>"
        "<th>Holdout</th>"
        "<th>Epoch</th>"
        "<th>Test</th>"
        "<th>Holdout</th>\n"
        "</tr>\n"
        "</thead>\n"
    )
    body_rows: List[str] = []
    for row in rows:
        task_metrics_obj = row["task_metrics"]
        assert isinstance(task_metrics_obj, dict)
        task_metrics: TaskMetricsCols = task_metrics_obj  # type: ignore[assignment]
        tm_cd_raw = dict(task_metrics.get("cd") or _empty_task_metrics())
        tm_ct_raw = dict(task_metrics.get("ct") or _empty_task_metrics())


        paired_names_html = row.get("paired_names_html")
        pair = row.get("description_html_pair")
        if isinstance(paired_names_html, str):
            first_cell = paired_names_html
        elif isinstance(pair, tuple) and pair[1] is not None:
            d0 = pair[0]
            d1 = pair[1]
            first_cell = f"{d0} / {d1}"
        elif isinstance(pair, tuple):
            first_cell = str(pair[0])
        else:
            first_cell = ""

        cd_med_ep = html.escape(_fmt_median_epoch(tm_cd_raw["best_epochs"]))
        cd_test = html.escape(_fmt_mean_std(tm_cd_raw["test_aucs"], decimals=decimals))
        cd_hold = html.escape(_fmt_mean_std(tm_cd_raw["hold_aucs"], decimals=decimals))
        ct_med_ep = html.escape(_fmt_median_epoch(tm_ct_raw["best_epochs"]))
        ct_test = html.escape(_fmt_mean_std(tm_ct_raw["test_aucs"], decimals=decimals))
        ct_hold = html.escape(_fmt_mean_std(tm_ct_raw["hold_aucs"], decimals=decimals))

        desc_cell = first_cell.strip() if first_cell else str(row.get("label_html", ""))
        body_rows.append(
            f"<tr>\n<td>{desc_cell}</td>"
            f"<td>{cd_med_ep}</td>"
            f"<td>{cd_test}</td>"
            f"<td>{cd_hold}</td>"
            f"<td>{ct_med_ep}</td>"
            f"<td>{ct_test}</td>"
            f"<td>{ct_hold}</td>\n</tr>"
        )
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hyenadna_dir = repo_root / "results" / "hyenadna"
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    experiment_rows = _experiment_rows(repo_root)
    recorder = GapRecorder()
    rows = collect_table_rows(hyenadna_dir, experiment_rows, recorder)
    seed_report = _format_problem_report(recorder, missing_ablations=[])
    if seed_report.strip():
        print(seed_report, file=sys.stderr, flush=True)
    text = format_table_html(rows, decimals=DECIMALS)
    out_path = repo_root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
