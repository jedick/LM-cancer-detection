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

This project is organized as a Makefile-driven analysis pipeline. Paths and default
parameters are stored in `defaults.yaml`, and scripts load those defaults
directly. `Makefile` provides convenience targets for the common pipeline steps, listed below.

Start with `make download_data` to download the 16S rRNA gene sequence data from SRA and save the gz files under `fasta/`.
See the list for a quick overview of the steps and read below for details.

1. Installation: `pip install -r requirements.txt` installs dependencies including the local `hyenadna` package in editable mode.
2. Download data: `make download_data` downloads 16S sequences from SRA (about 9 GB on disk).
3. Tetramer counts: `make tetramer_frequencies` counts tetramers in sequences and gets run-level means (several hours / about 5 GB on disk).
4. Tetramer classifier: `make -j4 fit_tetramer EXPT=0` generates results files in `results/tetramer` (about 6 min).
5. Sequence cache: `make sequence_cache` generates a Parquet file with tetramer counts for the first 10000 sequences in each run.
6. UC/CAP pipeline: `make run_uc_cap FEAT=0` generates cluster abundance profiles in `outputs/uc_cap` (about 40 min / 100GB RAM).
7. UC/CAP classifier: `make -j4 fit_uc_cap FEAT=0 EXPT=0` generates results files in `results/uc_cap` (about 2 hr).
8. HyenaDNA run tensors: `make run_tensors` builds `outputs/run_tensors/*.pt` from FASTA files (about 15 min).
9. HyenaDNA classifier: `make train_hyenadna EXPT=0` generate HyenaDNA experiment results in `results/hyenadna` (about 16 hr).
10. Use `helpers/table*.py` and `helpers/figure*py` scripts to make manuscript tables and figures from the results files.

Notes:

- `EXPT=0` means to run all experiments for a target listed in `experiments.yaml`.
  Can add `-j` for parallel execution of non-GPU tasks.
- `FEAT=0` is used to build all feature sets for the UC/CAP pipeline.
  Use e.g. `FEAT=1` or `EXPT=1` for a single feature set or experiment.
- Steps 5 and 6 (sequence cache and UC/CAP pipeline) are the most memory-hungry steps.
- Use the Cursor skill `/manuscript-tables 1` (or 3, 4) to execute a table helper and insert the HTML table in `manuscript.md`.

If you want to see why Make would rebuild a target (including recursive prerequisite chains), use `make explain-<target>`.
For example, run `make explain-run_uc_cap FEAT=0`; replace the part after `explain-` with any Make target name.

## Details

### Datasets

The `datasets.csv` file identifies development and holdout studies in the
`partition` column. Train/val/test splits are taken only from development studies,
and metrics are calculated separately for test (development) and holdout studies.
Split proportions are configured in `defaults.yaml (currently 0.70/0.15/0.15 for train/val/test).

### Download data

`make download_data` reads study metadata files in `data/**/*.csv` and outputs FASTA files to `fasta/<study>/<Run>.fasta.gz`.

### Tetramer frequencies

`make tetramer_frequencies` builds tetramer (4-mer) profiles from the downloaded FASTA files.
The script only processes rows with `sample_used=TRUE` in the CSV files under `data/`.
It writes two outputs with 256 feature columns in the same lexicographic ACGT 4-mer order:

1. `outputs/tetramer_frequencies.csv` is used by the run-level tetramer classifer.
It has one row per run with `Run` plus 256 columns of **percentage** 4-mer frequencies (counts summed over every sequence in that run's FASTA, then scaled to 100%).
2. The per-run files `outputs/<cancer_type>/<study_name>/<Run>.csv.xz` are used by the UC/CAP classifier.
They hold 256 columns of **raw integer** 4-mer counts with no header row, one row per FASTA sequence in encounter order.

### Run-level tetramer classifier

`make fit_tetramer` fits models on `outputs/tetramer_frequencies.csv`, with CLR, scaling, and PCA.
Default task/model and other settings are resolved from `fit_classifier` in `defaults.yaml` with results written to `results/scratch/`.
Add `EXPT=N` to get experiment name and model configuration from `experiments.yaml` and write results to `results/tetramer/{name}.json`.
Hyperparameters are chosen on validation, then ROC AUC is reported for test and holdout.
Supported models are `baseline`, `knn`, `random_forest`, `logistic_regression`, and `svm`.

Output files:
- Default run (`make fit_tetramer`) writes `results/scratch/tetramer_*.json`
- Experiment runs (`make fit_tetramer EXPT=N`) write under `results/tetramer/{name}.json` using name from `experiments.yaml`

### Unsupervised Clustering with Cluster Abundance Profiling

This stage takes the per-run tetramer count files and generates feature sets (cluster abundance profiles) used for classification.

- `make sequence_cache` builds a cached sequence-level tetramer table for UC/CAP exploration.
  It reads per-run tetramer counts from `outputs/<cancer>/<study>/<Run>.csv.xz`, kaeeps the first 10000 rows (configured in `defaults.yaml`) from each run,
  and saves the result in a Parquet table at `outputs/uc_cap/sequence_counts_first_10000_all_runs.parquet`.
- `make run_uc_cap` without arguments builds a CSV at `outputs/uc_cap/uc{n}_k{k}/cap{n}.csv` with defaults from `defaults.yaml`.
  With `FEAT=N` / `FEAT=0`, it drives the same pipeline to generate one or all feature sets using parameters from `run_uc_cap_pipeline` in `experiments.yaml` .
  Each CSV row is a sequencing run; `cluster_*` columns are normalized per-run abundances.
    - Use `make run_uc_cap FEAT=0` to generate all UC/CAP feature sets.
    - Use `make fit_uc_cap FEAT=0 EXPT=0` to run all experiments for all UC/CAP feature sets.
    - `make fit_uc_cap FEAT=0` or `EXPT=0` alone is rejected; a specific FEAT or EXPT is required when sweeping over the other.
- `make fit_uc_cap` fits the UC/CAP features using the same models and hyperparameters described above for the tetramer classifier.
  With `FEAT=M EXPT=N` it writes results to `results/uc_cap/{M}/{name}.json` using the N'th named `fit_classifier` experiment in `experiments.yaml`
    - Outputs: default `make fit_uc_cap` uses scratch JSON; Make-driven `FEAT`/`EXPT` sweeps write under `results/uc_cap/<FEAT>/`.

### HyenaDNA

HyenaDNA code is located under `hyenadna/`.
Install it as a local package from the repository root (this is included in `requirements.txt`):
`pip install -e .`

Source files:
- `standalone_hyenadna.py` was downloaded from [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)
- `huggingface_wrapper.py` and `inference_example.py` were extracted from the [HyenaDNA Colab Notebook](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL)
- Local modifications are summarized in the comments in each file.
- To run the inference example: `cd hyenadna; python -c 'import inference_example as ex; ex.inference_single()'`

Project execution:
- Build cached tensors: `make run_tensors` builds a single run-level tensor cache from `defaults.yaml` `run_tensors`.
- Train/evaluate HyenaDNA: `make train_hyenadna` uses defaults from `defaults.yaml`.
- Experiments: `make train_hyenadna EXPT=0` runs every `train_hyenadna` experiment in `experiments.yaml`.

Inputs/outputs:
- Run tensors:
  - Inputs: FASTA files and `run_tensors` configuration from `defaults.yaml`.
  - Outputs: one run tensor file per run at `outputs/run_tensors/<Run>.pt`.
- HyenaDNA classifier:
  - Inputs: cached run tensors, run labels/splits from `scripts/shared_utilities.py`, and pretrained weights under `checkpoints/<model>/` (or download on demand).
  - Outputs:
      - Default `make train_hyenadna` writes `results/scratch/train_hyenadna_<task>_<timestamp>.json`.
      - Experiment runs (`make train_hyenadna EXPT=N`) write under `results/hyenadna/{name}.json`.
