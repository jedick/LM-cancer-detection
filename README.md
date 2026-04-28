# Cancer detection from gut microbiomes using a DNA language model

**Goal:** Implement a cancer prediction and classification pipeline using 16S rRNA gene sequences from fecal samples.

**Tasks:** Cancer diagnosis (cancer vs healthy) and cancer type (breast vs colorectal).

## Study design

**Stage 1** uses 16S rRNA sequences from eight studies (four breast cancer, four colorectal cancer). All eight studies contribute to model fitting and testing via in-study train/test splits. In-study test sets will yield optimistic AUC estimates, as the model has seen sequences from the same study during training.

**Stage 2** addresses this by validating on four held-out studies (two breast, two colorectal) not seen during training. AUC values are expected to drop substantially for all models. *The central research question is whether the language model generalizes better across studies than classical approaches.*

## Methods

**Feature engineering**

- *Run-averaged tetranucleotide profiles:* 4-mer counts are summed across all sequences in a run and converted to percentages. This collapses within-run sequence heterogeneity into a single 256-dimensional vector per run.
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

The `sample_used` column is a Boolean that indicates whether samples are used for training.
Samples labeled as `benign` and non-fecal samples in some studies are excluded from training.

## Data analysis pipeline

Install script dependencies from the repository root: `pip install -r requirements.txt`.

This project is organized as a Makefile-driven analysis pipeline. Paths and default
parameters are stored in `configs/pipeline.yaml`, and `Makefile` defines targets
that run the pipeline scripts with those configured settings.

Start with `make download_data` to download the 16S rRNA gene sequence data from SRA and save the gz files under `fasta/`.
See the table for an overview of all the steps and read below for details.

| Step | Script | Target |
|-----|--------|--------|
| 1. Download | download_sra_data.py | `make download_data` |
| 2. Tetramer counts | calculate_tetranucleotide_frequencies.py | `make tetramer_frequencies` |
| 3. Sequence cache | build_uc_cap_sequence_cache.py | `make sequence_cache` |
| 4. Tetramer classifier | fit_tetranucleotide_classifier.py | `make fit_knn TASK=cancer_diagnosis` |
| 5. UC/CAP pipeline + classifier | run_uc_cap_pipeline.py + fit_uc_cap_classifier.py | `make fit_uc_cap TASK=cancer_diagnosis MODEL=random_forest N_UC=1000 N_CLUSTERS=2000 N_CAP=5000` |
| 6. UC/CAP grid | grid_uc_cap_pipeline.py | `make grid_uc_cap` |

<details>
<summary>Inputs/Outputs by step</summary>

1. Download. Inputs: `data/**/*.csv`. Outputs: `fasta/<study>/<Run>.fasta.gz`.
2. Tetramer counts. Inputs: `fasta/<study>/<Run>.fasta.gz`. Outputs: `outputs/tetranucleotide_frequencies.csv`, `outputs/<cancer>/<study>/<Run>.csv.xz`.
3. Sequence cache. Inputs: `outputs/<cancer>/<study>/<Run>.csv.xz`. Outputs: `outputs/uc_cap/sequence_counts_first_10000_all_runs.parquet`.
4. Tetramer classifier. Inputs: `outputs/tetranucleotide_frequencies.csv`. Outputs: `results/scratch/fit_tetranucleotide_classifier_*.json`.
5. UC/CAP pipeline + classifier. Inputs: `outputs/uc_cap/sequence_counts_first_10000_all_runs.parquet`, `outputs/tetranucleotide_frequencies.csv`. Outputs: `outputs/uc_cap/uc{n}_k{k}/cap{n}.csv`, `results/scratch/fit_uc_cap_classifier_*.json`.
6. UC/CAP grid. Inputs: `configs/pipeline.yaml`, `outputs/uc_cap/sequence_counts_first_10000_all_runs.parquet`. Outputs: `outputs/uc_cap/uc{n}_k{k}/cap{n}.csv` across the YAML grid.

</details>

## Details

### Tetranucleotide frequencies

`calculate_tetranucleotide_frequencies.py` builds tetranucleotide (4-mer) profiles from the downloaded FASTA files.
The script only processes rows with `sample_used=TRUE` in the CSV files under `data/`.
It writes two outputs from the same pass: `outputs/tetranucleotide_frequencies.csv` has one row per run with labels plus 256 columns of **percentage** 4-mer frequencies (counts summed over every sequence in that run's FASTA, then scaled to 100%).
The per-run files `outputs/<cancer_type>/<study_name>/<Run>.csv.xz` hold **raw integer** 4-mer counts per sequence for Unsupervised Clustering with Cluster Abundance Profiling (see below); each file has one row per FASTA sequence in encounter order, 256 columns with no header row, in the same lexicographic ACGT 4-mer order as the 256 feature columns in `outputs/tetranucleotide_frequencies.csv`.

### Run-level tetranucleotide classifier

`fit_tetranucleotide_classifier.py` fits a KNN classifier on `outputs/tetranucleotide_frequencies.csv`, with optional CLR, scaling, PCA, and a stratified train/validation/test split; hyperparameters are chosen on the validation set.
The script supports two binary tasks via `--task`: cancer diagnosis (cancer vs healthy) and cancer type (breast vs colorectal).
We ran the script with the `--baselines` argument and `--task=cancer_diagnosis` or `--task=cancer_type` to generate the `*_results.txt` files in `results/`.

### Unsupervised Clustering with Cluster Abundance Profiling

This stage takes the sequence-level tetranucleotide count files described above.

- `build_uc_cap_sequence_cache.py` builds a cached sequence-level tetranucleotide table for UC/CAP exploration.
  Running it with default arguments keeps only the first 5000 rows from each run file and saves the result in a Parquet table under `outputs/uc_cap`.
- `run_uc_cap_pipeline.py` performs unsupervised clustering on a subset of cached rows (`n_uc`) and then assigns a larger subset of sequences (`n_cap`) to the learned cluster centroids, producing run-level cluster abundance profiles.
  Its main output is `outputs/uc_cap/cap_features_*.csv`, where each row is a sequencing run and the `cluster_*` columns are normalized per-run abundances used as features for downstream classification.
- `fit_uc_cap_classifier.py` uses the features generated by the pipeline to fit classification models.
- `grid_uc_cap_pipeline.py` runs the pipeline for different combinations of parameters from `configs/pipeline.yaml`. Then `grid_uc_cap_classifier.py` uses the features generated by the pipeline to run the classifier with different models for the two tasks.
