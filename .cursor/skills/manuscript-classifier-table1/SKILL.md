---
name: manuscript-classifier-table1
description: >-
  Regenerates Table 1 in manuscript/report.md from tetramer classifier JSON
  metrics under results/tetramer/. Runs helpers/table1_from_classifier.py and
  replaces the table between HTML comment markers. Use when tetramer JSON
  results change, when updating the Results table, or when the user asks to
  refresh Table 1.
---

# Manuscript Table 1 (classifier AUC)

## Goal

Keep **Table 1** in `manuscript/report.md` in sync with **six JSON files** under `results/tetramer/`:

- `cancer_diagnosis_{baseline,knn,random_forest}.json`
- `cancer_type_{baseline,knn,random_forest}.json`

(produced by `make fit_tetramer` / `scripts/fit_tetramer_classifier.py` via `experiments.yaml`).

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

- Rows: **Majority class** (baseline), **KNN**, **Random Forest**. Columns: per task, **Test** and **Holdout** ROC AUC.
- If the script exits with “Missing expected JSON”, run `make fit_tetramer` (or add the missing `{task}_{model}` experiment) until all six files exist.
