*Cancer detection from gut microbiomes using a DNA language model*

## Sample metadata, labeling, and exclusion

The `metadata/` directory contains CSV files assembled using publicly available BioSample information.
Each file has a `study_name` column, representing the first three authors and year of publication.
The `cancer_type` column is one of `breast` or `colorectal`.

`sample_id` and `BioSample` are the SRA Run and BioSample accession numbers.
Other named columns are specific to the studies and are used to assign the sample labels.
For some datasets, BioSample descriptions are mapped to a new `cohort` column that is restricted to `cancer` or `control`.

*For the purposes of model training, sample labels are normalized from the original source labels.*
The `sample_label` column in each CSV file is restricted to these values: `healthy`, `breast_cancer`, `colorectal_cancer`, or `benign`.

The `benign` category contains both adenomas (benign colon tumors) and breast ductal carcinoma in-situ (DCIS).
`breast_cancer` includes invasive tumors and `colorectal_cancer` includes carcinoma.
`healthy` includes the control samples in each study - these may be variously described as healthy, controls, or non-cancerous.

*Samples labeled as `benign` are not used for model training.*
Additionally, non-fecal samples provided in some studies are excluded from training.
The `sample_used` column is a Boolean that indicates whether samples are used for training.

## Downloading data

Use `scripts/download_sra_data.py` to download the sequence data from SRA.

