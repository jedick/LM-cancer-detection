# Cancer detection with gut microbiomes using a DNA language model

## Methods

### Data sources

Each sample corresponds to a sequencing run with multiple 16S rRNA gene sequences. We collected sequencing runs from different studies (four for breast cancer and four for colorectal cancer). Studies were only included if cancer/healthy labels were available. We stored the SRA Run accessions (starting with SRR, ERR, or DRR) and study metadata in the repository and downloaded each run’s read archive from NCBI.

### Preprocessing

We normalized sample labels from the sources using a restricted set of labels: healthy, breast cancer, colorectal cancer, and benign.
The benign category contains adenomas and benign colon polyps and breast ductal carcinoma in-situ (DCIS).
Samples labeled breast cancer includes invasive tumors and those labeled colorectal cancer includes carcinoma.
All benign samples and non-fecal samples in some studies were kept in our data files for auditability but were excluded from training.

We calculated tetramer frequencies for each run by counting 4-mers within each sequence, summing counts over all sequences in the run, then converting to percentages. 

### Classification

We split the data into stratified training, validation, and test sets (70:10:20). We held the validation set fixed (no cross-validation) to use the same splits for training both the LM (expensive compute - only one val set feasible) and this classical model.

We trained a K-nearest neighbors classifier on the 256 frequency features. We applied a centered log-ratio transform (CLR), standardized the CLR coordinates, then applied PCA. The number of components in PCA starts with the maximum rank count followed by halves (128, 64, and so on capped by the training sample size) until the cumulative explained variance on the training fold falls below 0.9. We tuned number of PCA components, number of KNN nneighbors, and weights by grid search on the validation set only, then fit the chosen pipeline on the training split. Performance metrics on the test set were calculated with scikit-learn’s classification_report.

## Results

### KNN classification


