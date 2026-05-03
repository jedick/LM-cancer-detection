---
name: manuscript-tables
description: >-
  Refreshes Table 1, 2, or 3 in manuscript/report.md from classifier JSON.
  User picks the table with a digit after the skill (e.g. /manuscript-tables 1):
  1 = tetramer (table1_from_classifier.py), 2 = UC/CAP grid test AUC
  (table2_uc_cap_from_classifier.py), 3 = UC/CAP triple test+holdout
  (table3_uc_cap_from_classifier.py). Replace HTML only between the matching
  <!-- classifier-table-N --> markers.
---

# Manuscript tables

## How to invoke

The user selects **one** table with a digit **1**, **2**, or **3** after the skill (for example `/manuscript-tables 1`, `/manuscript-tables 2`, or “manuscript-tables **3**”). Follow **only** the matching section below for that run. Do not refresh other tables unless the user asks.

| Digit | Table | Command | HTML markers in `manuscript/report.md` |
| ---: | --- | --- | --- |
| **1** | Tetramer classifiers | `python helpers/table1_from_classifier.py` | `<!-- classifier-table-1 -->` … `<!-- /classifier-table-1 -->` |
| **2** | UC/CAP grid (test AUC only) | `python helpers/table2_uc_cap_from_classifier.py` | `<!-- classifier-table-2 -->` … `<!-- /classifier-table-2 -->` |
| **3** | UC/CAP selected triple (test + holdout) | `python helpers/table3_uc_cap_from_classifier.py` | `<!-- classifier-table-3 -->` … `<!-- /classifier-table-3 -->` |

If no digit is given, ask which table (1–3) to refresh, or whether to run all three in order (1 then 2 then 3).

All commands assume the **repository root** as the current working directory.

---

# Manuscript Table 1 (classifier AUC)

## Goal

Keep **Table 1** in `manuscript/report.md` in sync with **eight JSON files** under `results/tetramer/`:

- `cancer_diagnosis_{baseline,knn,svm,random_forest}.json`
- `cancer_type_{baseline,knn,svm,random_forest}.json`

(produced by `make fit_tetramer` / `scripts/fit_classifier.py --tetramer` via `experiments.yaml`).

## Steps

1. From the **repository root**, run:

   ```bash
   python helpers/table1_from_classifier.py
   ```

   Optional overrides:
   - `--tetramer-dir PATH` (default: `results/tetramer`)
   - `--decimals N` (default: 3)
   - `--markdown` — pipe table (two-line header) instead of default **HTML** nested header table

2. Open **`manuscript/report.md`**. Find the block between these markers:

   - `<!-- classifier-table-1 -->`
   - `<!-- /classifier-table-1 -->`

3. **Replace only the lines between those two markers** (not the markers themselves) with the script’s stdout. Default output is an HTML `<table>` (nested headers: task × test/holdout). Preserve one newline after the closing marker if the surrounding prose expects it.

4. Save the manuscript. Follow **`.cursor/rules/manuscript-prose.mdc`** for any prose edits nearby (for example update surrounding sentences if the table now includes holdout and random forest).

## Notes

- Rows: **Majority class** (baseline), **KNN**, **SVM**, **Random Forest**. Columns: per task, **Test** and **Holdout** ROC AUC.
- If the script exits with “Missing expected JSON”, run `make fit_tetramer` (or add the missing `{task}_{model}` experiment) until all eight files exist.

---

# Manuscript Table 2 (UC/CAP grid, test AUC only)

## Goal

Sync **Table 2** with `results/uc_cap/<feat>/cancer_{diagnosis,type}_{knn,svm,random_forest}.json` for every `run_uc_cap_pipeline` row (after defaults merge), same order as `FEAT=1..N`.

## Steps

1. From the repository root:

   ```bash
   python helpers/table2_uc_cap_from_classifier.py
   ```

   Optional: `--uc-cap-dir`, `--decimals`.

2. Replace lines between `<!-- classifier-table-2 -->` and `<!-- /classifier-table-2 -->` in `manuscript/report.md` with stdout.

---

# Manuscript Table 3 (selected UC/CAP triple, test + holdout)

## Goal

Sync **Table 3** with KNN, SVM, and random forest JSON for one feature triple. Defaults match the manuscript prose: *n*<sub>UC</sub> = 2000, *K* = 2000, *n*<sub>CAP</sub> = 10000 (resolved to `results/uc_cap/<feat>/` via `experiments.yaml` + `defaults.yaml`).

## Steps

1. From the repository root:

   ```bash
   python helpers/table3_uc_cap_from_classifier.py
   ```

   Optional: `--n-uc`, `--n-clusters`, `--n-cap`, `--uc-cap-dir`, `--decimals`.

2. Replace lines between `<!-- classifier-table-3 -->` and `<!-- /classifier-table-3 -->` in `manuscript/report.md` with stdout.

## Notes

- If resolution fails (“No run_uc_cap_pipeline row matches”), adjust `--n-*` or add the row to `experiments.yaml`.
- Missing JSON under the resolved feat dir: run `make fit_uc_cap` for the needed `(FEAT, EXPT)` cells.
