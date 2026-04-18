---
name: manuscript-classifier-table1
description: >-
  Regenerates Table 1 in report/manuscript.md from outputs/classifier_results.txt.
  Runs report/table1_from_classifier.py and replaces the Markdown table between HTML
  comment markers. Use when classifier_results.txt changes, when updating the Results
  table, or when the user asks to refresh Table 1 or the KNN classification table.
---

# Manuscript Table 1 (classifier AUC)

## Goal

Keep **Table 1** in `report/manuscript.md` in sync with **`outputs/classifier_results.txt`** (stdout from `scripts/fit_tetranucleotide_classifier.py`, typically run with **`--baselines`**).

## Steps

1. From the **repository root**, run:

   ```bash
   python report/table1_from_classifier.py
   ```

   Optional: `--input PATH` if the log lives elsewhere; `--decimals N` (default 3).

2. Open **`report/manuscript.md`**. Find the block between these markers:

   - `<!-- classifier-table-1 -->`
   - `<!-- /classifier-table-1 -->`

3. **Replace only the lines between those two markers** (not the markers themselves) with the script’s stdout. Preserve one newline after the closing marker if the surrounding prose expects it.

4. Save the manuscript. Follow **`.cursor/rules/manuscript-prose.mdc`** for any prose edits nearby.

## Notes

- The script includes **Majority class** and **KNN** only (macro- and micro-averaged one-vs-rest ROC AUC).
- If parsing fails, re-run the classifier and ensure the log still contains lines starting with `  KNN:` and `  Majority class:` under the test-set sections.
