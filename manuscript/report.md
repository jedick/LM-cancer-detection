# Cancer detection with gut microbiomes using a DNA language model

## Methods

### Data sources

Each sample corresponds to a sequencing run with multiple 16S rRNA gene sequences. We collected sequencing runs from different studies (four for breast cancer and four for colorectal cancer). Studies were only included if cancer/healthy labels were available. We stored the SRA Run accessions (starting with SRR, ERR, or DRR) and study metadata in the repository and downloaded each run’s read archive from NCBI.

### Preprocessing

We normalized sample labels from the sources using a restricted set of labels: healthy, breast cancer, colorectal cancer, and benign.
The benign category contains adenomas and benign colon polyps and breast ductal carcinoma in-situ (DCIS).
Samples labeled breast cancer includes invasive tumors and those labeled colorectal cancer includes carcinoma.
All benign samples and non-fecal samples in some studies were kept in our data files for auditability but were excluded from training.

We split the data into stratified training, validation, and test sets (70:10:20). We held the validation set fixed (no cross-validation) to use the same splits for training both the LM (expensive compute - only one val set feasible) and this classical model.
The same global split was used for downstream tasks, including cancer diagnosis prediction (using all samples) and cancer type prediction (using only cancer-positive samples in each fold).

### Classification using run-level tetranucleotide frequencies

We calculated tetramer frequencies for each run by counting 4-mers within each sequence, summing counts over all sequences in the run, then converting to percentages. 

We trained a K-nearest neighbors classifier on the 256 frequency features. We applied a centered log-ratio transform (CLR), standardized the CLR coordinates, then applied PCA. The number of components in PCA starts with the maximum rank count followed by halves (128, 64, and so on capped by the training sample size) until the cumulative explained variance on the training fold falls below 0.9. We tuned number of PCA components, number of KNN nneighbors, and weights by grid search on the validation set only, then fit the chosen pipeline on the training split.

### Classification using cluster abundance profiles

Run-level tetranucleotide features summarize each sample with a single average profile and therefore do not directly capture how different sequence types are distributed within a run.
To preserve this within-run compositional structure, we used unsupervised clustering followed by cluster abundance profiles (UC/CAP), a reference-free and alignment-free approach.

Because the sequence-level table is large, we first fit the unsupervised clustering model on a bounded subset of sequences from each run (`n_uc`).
For each sequence, we computed a 256-dimensional tetranucleotide composition vector, then applied principal component analysis (PCA) and retained components that explained 95% of cumulative variance.
We then fit k-means in this reduced space to obtain K centroids that define a sequence codebook.

To construct run-level features, we assigned a larger subset of sequences per run (`n_cap`) to the learned centroids in the same transformed feature space.
We counted cluster memberships within each run and normalized by the number of assigned sequences, yielding a K-dimensional cluster abundance profile for each run.
These CAP vectors were then used as the feature matrix for supervised classification for both binary tasks (cancer versus healthy and breast versus colorectal), with the downstream classifier selected separately.

## Results

We defined two binary classification tasks: cancer vs healthy (diagnosis) and breast vs colorectal cancer (cancer type).
Performance metrics (AUC - area under the receiver operating characteristic curve) are reported below for the test set for each task and model.

### KNN classification

Table 1 reports test set ROC AUC for each binary task for the majority class baseline and the KNN classifier.

<!-- classifier-table-1 -->
| Model | Cancer diagnosis AUC | Cancer type AUC |
| :--- | ---: | ---: |
| Majority class | 0.500 | 0.500 |
| KNN | 0.645 | 0.956 |
<!-- /classifier-table-1 -->

The KNN model exceeds the baseline on both binary tasks.
