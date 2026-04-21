*Cancer detection from gut microbiomes using a DNA language model*

Goal: to implement a cancer prediction and classification pipeline using 16S rRNA gene sequences from fecal samples.

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

## Downloading data

Install script dependencies from the repository root: `pip install -r requirements.txt`.

Use `scripts/download_sra_data.py` to download the 16S rRNA gene sequence data from SRA.
The gzipped sequence files are stored under `fasta/` in a separate directory for each study.

## Tetranucleotide frequencies

Use `scripts/calculate_tetranucleotide_frequencies.py` to build tetranucleotide (4-mer) profiles from the downloaded FASTA files.
The script only processes rows with `sample_used=TRUE` in the CSV files under `data/`.
It writes two outputs from the same pass: `outputs/tetranucleotide_frequencies.csv` has one row per run with labels plus 256 columns of **percentage** 4-mer frequencies (counts summed over every sequence in that run's FASTA, then scaled to 100%).
The per-run files `outputs/<cancer_type>/<study_name>/<Run>.csv` hold **raw integer** 4-mer counts per sequence for Unsupervised Clustering with Cluster Abundance Profiling (see below); each file has one row per FASTA sequence in encounter order, 256 columns with no header row, in the same lexicographic ACGT 4-mer order as the 256 feature columns in `outputs/tetranucleotide_frequencies.csv`.

## Run-level tetranucleotide classifier

Use `scripts/fit_tetranucleotide_classifier.py` to fit a KNN classifier on `outputs/tetranucleotide_frequencies.csv`, with optional CLR, scaling, PCA, and a stratified train/validation/test split; hyperparameters are chosen on the validation set.
The script supports two binary tasks via `--task`: cancer diagnosis (cancer vs healthy) and cancer type (breast vs colorectal).
We ran the script with the `--baselines` argument and `--task=cancer_diagnosis` or `--task=cancer_type` to generate the `*_results.txt` files in `outputs/`.

## Unsupervised Clustering with Cluster Abundance Profiling

This stage takes the sequence-level tetranucleotide count files described above.
Further documentation of the clustering and abundance profiling workflow will be added as that part of the pipeline is implemented.
