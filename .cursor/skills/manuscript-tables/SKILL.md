---
name: manuscript-tables
description: >-
  Refreshes Table 1, 3, or 4 in manuscript.md from classifier JSON.
  User picks the table with a digit after the skill (e.g. /manuscript-tables 1):
  1 = tetramer (table1_tetramer.py), 3 = UC/CAP selected triple
  (table3_uc_cap.py), 4 = HyenaDNA cache/model grid (table4_hyenadna.py).
  Replace HTML only between the matching
  <!-- classifier-table-N --> markers.
---

# Manuscript tables

## How to invoke

The user selects **one** table with a digit **1**, **3**, or **4** after the skill (for example `/manuscript-tables 1`, `/manuscript-tables 3`, or “manuscript-tables **4**”). Follow **only** the matching section below for that run. Do not refresh other tables unless the user asks.

| Digit | Table | Command | HTML markers in `manuscript.md` |
| ---: | --- | --- | --- |
| **1** | Tetramer classifiers | `python helpers/table1_tetramer.py` | `<!-- classifier-table-1 -->` … `<!-- /classifier-table-1 -->` |
| **3** | UC/CAP selected triple (test + holdout) | `python helpers/table3_uc_cap.py` | `<!-- classifier-table-3 -->` … `<!-- /classifier-table-3 -->` |
| **4** | HyenaDNA cache/model grid (test + holdout) | `python helpers/table4_hyenadna.py` | `<!-- classifier-table-4 -->` … `<!-- /classifier-table-4 -->` |

If no digit is given, ask which table (1, 3, or 4) to refresh, or whether to run all three in order (1 then 3 then 4).

All commands assume the **repository root** as the current working directory.

---

# Manuscript Table 1 (classifier AUC)

## Goal

Keep **Table 1** in `manuscript.md` in sync with **eight JSON files** under `results/tetramer/`:

- `cancer_diagnosis_{baseline,knn,svm,random_forest}.json`
- `cancer_type_{baseline,knn,svm,random_forest}.json`

(produced by `make fit_tetramer` / `scripts/fit_classifier.py --tetramer` via `experiments.yaml`).

## Steps

1. From the **repository root**, run:

   ```bash
   python helpers/table1_tetramer.py
   ```

   Optional overrides:
   - `--tetramer-dir PATH` (default: `results/tetramer`)
   - `--decimals N` (default: 3)
   - `--markdown` — pipe table (two-line header) instead of default **HTML** nested header table

2. Open **`manuscript.md`**. Find the block between these markers:

   - `<!-- classifier-table-1 -->`
   - `<!-- /classifier-table-1 -->`

3. **Replace only the lines between those two markers** (not the markers themselves) with the script’s stdout. Default output is an HTML `<table>` (nested headers: task × test/holdout). Preserve one newline after the closing marker if the surrounding prose expects it.

4. Save the manuscript. Follow **`.cursor/rules/manuscript-prose.mdc`** for any prose edits nearby (for example update surrounding sentences if the table now includes holdout and random forest).

## Notes

- Rows: **Majority class** (baseline), **KNN**, **SVM**, **Random Forest**. Columns: per task, **Test** and **Holdout** ROC AUC.
- If the script exits with “Missing expected JSON”, run `make fit_tetramer` (or add the missing `{task}_{model}` experiment) until all eight files exist.

---

# Manuscript Table 3 (selected UC/CAP triple, test + holdout)

## Goal

Sync **Table 3** with KNN, SVM, and random forest JSON for one feature triple. Defaults match the manuscript prose: *n*<sub>UC</sub> = 2000, *K* = 5000, *n*<sub>CAP</sub> = 10000 (resolved to `results/uc_cap/<feat>/` via `experiments.yaml` + `defaults.yaml`).

## Steps

1. From the repository root:

   ```bash
   python helpers/table3_uc_cap.py
   ```

   Optional: `--n-uc`, `--n-clusters`, `--n-cap`, `--uc-cap-dir`, `--decimals`.

2. Replace lines between `<!-- classifier-table-3 -->` and `<!-- /classifier-table-3 -->` in `manuscript.md` with stdout.

---

# Manuscript Table 4 (HyenaDNA cache/model grid, test + holdout)

## Goal

Sync **Table 4** with HyenaDNA JSON metrics under `results/hyenadna/*/*.json`, using these six cache/model max-length rows in order:

- 1k-1k
- 2k-1k
- 2k-2k
- 4k-1k
- 4k-2k
- 4k-4k

Columns are `max_length` (cache, model), then cancer diagnosis/test+holdout and cancer type/test+holdout AUC.

## Steps

1. From the repository root:

   ```bash
   python helpers/table4_hyenadna.py
   ```

   Optional: `--hyenadna-dir`, `--decimals`.

2. Replace lines between `<!-- classifier-table-4 -->` and `<!-- /classifier-table-4 -->` in `manuscript.md` with stdout.

## Notes

- If resolution fails (“No run_uc_cap_pipeline row matches”), adjust `--n-*` or add the row to `experiments.yaml`.
- Missing JSON under the resolved feat dir: run `make fit_uc_cap` for the needed `(FEAT, EXPT)` cells.
