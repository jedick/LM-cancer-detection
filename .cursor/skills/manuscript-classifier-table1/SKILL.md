---
name: manuscript-classifier-table1
description: >-
  Regenerates Table 1 in report/manuscript.md from binary-task classifier logs:
  outputs/cancer_diagnosis_results.txt and outputs/cancer_type_results.txt.
  Runs report/table1_from_classifier.py and replaces the Markdown table between
  HTML comment markers. Use when either task output changes, when updating the
  Results table, or when the user asks to refresh Table 1.
---

# Manuscript Table 1 (classifier AUC)

## Goal

Keep **Table 1** in `report/manuscript.md` in sync with:

- **`outputs/cancer_diagnosis_results.txt`** (stdout from `scripts/fit_tetranucleotide_classifier.py --task cancer_diagnosis --baselines`)
- **`outputs/cancer_type_results.txt`** (stdout from `scripts/fit_tetranucleotide_classifier.py --task cancer_type --baselines`)

## Steps

1. From the **repository root**, run:

   ```bash
   python report/table1_from_classifier.py
   ```

   Optional overrides:
   - `--diagnosis-input PATH`
   - `--type-input PATH`
   - `--decimals N` (default 3)

2. Open **`report/manuscript.md`**. Find the block between these markers:

   - `<!-- classifier-table-1 -->`
   - `<!-- /classifier-table-1 -->`

3. **Replace only the lines between those two markers** (not the markers themselves) with the script’s stdout. Preserve one newline after the closing marker if the surrounding prose expects it.

4. Save the manuscript. Follow **`.cursor/rules/manuscript-prose.mdc`** for any prose edits nearby.

## Notes

- The script includes **Majority class** and **KNN** only.
- Table columns are binary-task AUCs: **Cancer diagnosis AUC** and **Cancer type AUC**.
- If parsing fails, re-run the classifier and ensure each task log contains lines like:
  - `  KNN: ROC AUC = ...`
  - `  Majority class: ROC AUC = ...`
