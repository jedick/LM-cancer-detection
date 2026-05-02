# Cancer detection from gut microbiomes using a DNA language model

**Goal:** Implement a cancer prediction and classification pipeline using 16S rRNA gene sequences from fecal samples.

**Tasks:** Cancer diagnosis (cancer vs healthy) and cancer type (breast vs colorectal).

## Study design

**Stage 1** uses 16S rRNA sequences from eight studies (four breast cancer, four colorectal cancer). All eight studies contribute to model fitting and testing via in-study train/test splits. In-study test sets will yield optimistic AUC estimates, as the model has seen sequences from the same study during training.

**Stage 2** addresses this by validating on four holdout studies (two breast, two colorectal) not seen during training. AUC values are expected to drop substantially for all models. *The central research question is whether the language model generalizes better across studies than classical approaches.*

## Methods

**Feature engineering**

- *Run-averaged tetramer profiles:* 4-mer counts are summed across all sequences in a run and converted to percentages. This collapses within-run sequence heterogeneity into a single 256-dimensional vector per run.
- *Sequence-level cluster abundance profiles (UC/CAP):* Sequences are individually characterized by their 4-mer composition, clustered unsupervised, and each run is then represented by the distribution of its sequences across clusters. This preserves within-run compositional structure that flat aggregation discards.

**Classical ML models:** KNN, random forest, logistic regression, SVM

## Sample data, labeling, and exclusion

The `data/` directory contains directories for different cancer types (breast and colorectal).
Within each directory are CSV files named for the source study (last name initials of first three authors and year).
`Run` and `BioSample` are the SRA Run and BioSample accession numbers.
Other named columns are specific to the studies.
For some datasets, sample descriptions are mapped to a `cohort` column that is restricted to `cancer` or `control`.

*For model training, sample labels are normalized from the original source labels.*
The `sample_label` column in each CSV file is restricted to these values: `healthy`, `breast_cancer`, `colorectal_cancer`, or `benign`.
The `benign` category contains adenomas and benign colon polyps and breast ductal carcinoma in-situ (DCIS).
`breast_cancer` includes invasive tumors and `colorectal_cancer` includes carcinoma.
`healthy` includes the control samples in each study - these may be variously described as healthy or normal controls.

The `sample_used` column is a Boolean that indicates whether samples are used in the analysis.
Samples labeled as `benign` and non-fecal samples in some studies are excluded from the analysis.

## Data analysis pipeline

Install script dependencies from the repository root: `pip install -r requirements.txt`.

This project is organized as a Makefile-driven analysis pipeline. Paths and default
parameters are stored in `defaults.yaml`, and scripts load those defaults
directly. `Makefile` provides convenience targets for the common pipeline steps.
The `datasets.csv` file identifies development and holdout studies in the
`partition` column. Train/val/test splits are taken only from development studies,
and metrics are calculated separately for test (development) and holdout studies.

Start with `make download_data` to download the 16S rRNA gene sequence data from SRA and save the gz files under `fasta/`.
See the table for an overview of all the steps and read below for details.

| Step | Script | Target |
|-----|--------|--------|
| 1. Download | download_sra_data.py | `make download_data` |
| 2. Tetramer counts | calculate_tetramer_frequencies.py | `make tetramer_frequencies` |
| 3. Tetramer classifier | fit_tetramer_classifier.py | `make fit_tetramer` |
| 4. Sequence cache | build_uc_cap_sequence_cache.py | `make sequence_cache` |
| 5. UC/CAP pipeline | run_uc_cap_pipeline.py | `make run_uc_cap` |
| 6. UC/CAP classifier | fit_uc_cap_classifier.py | `make fit_uc_cap` |

For non-default settings, use Makefile variables (`EXPT=...` for tetramer classifier runs, `FEAT=...` for UC/CAP pipeline feature sets) or run scripts directly with their supported options.

If you want to see why Make would rebuild a target (including recursive prerequisite chains), use `make explain-<target>`.
For example, run `make explain-run_uc_cap`; replace the part after `explain-` with any Make target name.

<details>
<summary>Inputs/Outputs by step</summary>

1. Download. Inputs: `data/**/*.csv`. Outputs: `fasta/<study>/<Run>.fasta.gz`.
2. Tetramer counts. Inputs: `fasta/<study>/<Run>.fasta.gz`. Outputs: `outputs/tetramer_frequencies.csv`, `outputs/<cancer>/<study>/<Run>.csv.xz`.
3. Tetramer classifier. Inputs: `outputs/tetramer_frequencies.csv`. Outputs: default run (`make fit_tetramer`) writes `results/scratch/fit_tetramer_classifier_*.json`; experiment runs (`make fit_tetramer EXPT=N`) write `results/tetramer/{name}.json`.
4. Sequence cache. Inputs: `outputs/<cancer>/<study>/<Run>.csv.xz`. Outputs: `outputs/uc_cap/sequence_counts_first_{n_max_per_run}_all_runs.parquet`.
5. UC/CAP pipeline. Inputs: `defaults.yaml`, `experiments.yaml` (for `FEAT>=1` or `FEAT=0`), `outputs/uc_cap/sequence_counts_first_{n_max_per_run}_all_runs.parquet`. Outputs: `outputs/uc_cap/uc{n}_k{k}/cap{n}.csv` (baseline from `defaults.yaml`; further feature-set rows from `experiments.yaml`).
6. UC/CAP classifier. Inputs: `outputs/uc_cap/sequence_counts_first_{n_max_per_run}_all_runs.parquet`, `outputs/tetramer_frequencies.csv`. Outputs: `outputs/uc_cap/uc{n}_k{k}/cap{n}.csv`, `results/scratch/fit_uc_cap_classifier_*.json`.

</details>

## Details

### Tetramer frequencies

`calculate_tetramer_frequencies.py` builds tetramer (4-mer) profiles from the downloaded FASTA files.
The script only processes rows with `sample_used=TRUE` in the CSV files under `data/`.
It writes two outputs with 256 feature columns in the same lexicographic ACGT 4-mer order:

1. `outputs/tetramer_frequencies.csv` is used by the run-level tetramer classifer.
It has one row per run with labels plus 256 columns of **percentage** 4-mer frequencies (counts summed over every sequence in that run's FASTA, then scaled to 100%).
2. The per-run files `outputs/<cancer_type>/<study_name>/<Run>.csv.xz` are used by the UC/CAP classifier.
They hold 256 columns of **raw integer** 4-mer counts with no header row, one row per FASTA sequence in encounter order.

### Run-level tetramer classifier

`make fit_tetramer` fits run-level classifiers on `outputs/tetramer_frequencies.csv`, with CLR, scaling, and PCA.
Default task/model and other settings are resolved from `defaults.yaml` with results written to `results/scratch/`.
Add `EXPT=N` to use experiment name and overrides from `experiments.yaml` and write results to `results/tetramer/{name}.json`.
Hyperparameters are chosen on validation, then ROC AUC is reported for test and holdout.
Supported models are `knn`, `random_forest`, `logistic_regression`, and `baseline` (majority class).

Use `make fit_tetramer EXPT=1` to run a single experiment and `make fit_tetramer EXPT=0` to run all experiments.
The `EXPT=0` Make targets support incremental rebuilds and `-j` for parallel execution (for example, `make -j4 fit_tetramer EXPT=0`).

### Unsupervised Clustering with Cluster Abundance Profiling

This stage takes the sequence-level tetramer count files described above.

- `make sequence_cache` builds a cached sequence-level tetramer table for UC/CAP exploration.
  It keeps the first 10000 rows (configured in `defaults.yaml`) from each run file and saves the result in a Parquet table under `outputs/uc_cap`.
- `make run_uc_cap` generates feature sets for downstream classification.
  It performs unsupervised clustering on a subset of cached rows (`n_uc`) and then assigns a larger subset of sequences (`n_cap`) to the learned cluster centroids, producing run-level cluster abundance profiles.
  Its main output is `outputs/uc_cap/uc{n_uc}_k{n_clusters}/cap{n_cap}.csv`, where each row is a sequencing run and the `cluster_*` columns are normalized per-run abundances.
  Parameters come from `defaults.yaml` with optional merging of a row defining a CAP feature set from `run_uc_cap_pipeline` in `experiments.yaml`.
  Use `make run_uc_cap FEAT=1` to generate a single feature set and `make run_uc_cap FEAT=0` to build all pipeline feature sets incrementally.
- `make fit_uc_cap` uses the features generated by the pipeline and the shared run splits to fit classification models.
