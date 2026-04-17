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

Use `scripts/calculate_tetranucleotide_frequencies.py` to build a CSV of tetranucleotide (4-mer) frequency profiles for model training from the downloaded FASTA files.
The script only process rows with `sample_used=TRUE` in the CSV files under `data/` and writes one table (`outputs/tetranucleotide_frequencies.csv`) with labels plus 256 feature columns (percentages of 4-mer counts across all sequences in each FASTA file).
