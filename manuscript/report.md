# Cancer detection with gut microbiomes using a DNA language model

## Methods

### Data sources

Each sample corresponds to a sequencing run with multiple 16S rRNA gene sequences.
We collected sequencing runs from different studies (four for breast cancer and four for colorectal cancer).
Studies were only included if cancer/healthy labels were available.
We stored the SRA Run accessions (starting with SRR, ERR, or DRR) and study metadata in the repository and downloaded each run’s read archive from NCBI.

### Preprocessing

We normalized sample labels from the sources using a restricted set of labels: healthy, breast cancer, colorectal cancer, and benign.
The benign category contains adenomas and benign colon polyps and breast ductal carcinoma in-situ (DCIS).
Samples labeled breast cancer includes invasive tumors and those labeled colorectal cancer includes carcinoma.
All benign samples and non-fecal samples in some studies were kept in our data files for auditability but were excluded from training.

We divided studies into two groups: studies reserved for external evaluation (holdout) and studies used for model development.
Runs in holdout studies were excluded from the stratified development assignment.
Among development studies only, we assigned each sequencing run to stratified training, validation, or test sets in a 70:10:20 ratio.
We defined those assignments in advance from version-controlled study lists and per-study sample tables,
so they did not depend on the order in which derived feature files were later produced.

We held the validation set fixed (no cross-validation) so that we could use the same development splits when training the language model
(expensive compute limited us to a single validation set) and this classical baseline.
The same run-level split was used across downstream tasks, including cancer versus healthy prediction on all samples and
breast versus colorectal prediction restricted to cancer-positive samples.

### Classification using run-level tetramer frequencies

We calculated tetramer frequencies for each run by counting 4-mers within each sequence,
summing counts over all sequences in the run, then converting those counts to percentages so each run is described by 256 frequency features.

For the baseline, we used majority-class prediction.

For KNN, we applied a centered log-ratio transform (CLR), standardized the CLR coordinates, then applied PCA.
PCA candidate sizes stepped down from the largest feasible rank (up to 256) by successive halves,
keeping only sizes for which the leading components explained at least 90% of the variance on the training fold.
We tuned the PCA size, the number of neighbors, and the distance weights by grid search on the validation split only.

For random forest, we used the same CLR and standardization but did not use PCA. We tuned the number of trees (200 or 500),
the maximum depth of trees (unlimited depth or a cap of 10), and the minimum number of training samples required to form a leaf (1 or 2),
again using only the validation split.

After choosing hyperparameters on the validation split, we fit each final pipeline on the training split.

### Classification using cluster abundance profiles

Run-level tetramer features summarize each sample with a single average profile and therefore do not directly capture
how different sequence types are distributed within a run.
To preserve this within-run compositional structure, we used unsupervised clustering followed by cluster abundance profiles (UC/CAP),
a reference-free and alignment-free approach.

Because the sequence-level table is large, we first fit the unsupervised clustering model using only sequences from runs in the training split,
with at most a fixed number of sequences per training run.
For each selected sequence, we computed a 256-dimensional tetramer composition vector,
then applied principal component analysis (PCA) and retained components that explained 95% of cumulative variance.
We then fit k-means in this reduced space to obtain K centroids that define a sequence codebook.

To construct run-level features, we applied the same PCA transformation and centroid assignments without refitting the unsupervised model,
using up to a larger per-run sequence budget for every run that entered the sequence-level table, including validation, test, and holdout runs.
We counted cluster memberships within each run and normalized by the number of assigned sequences, yielding a K-dimensional cluster abundance profile for each run.
These CAP vectors were then used as the feature matrix for supervised classification for both binary tasks
(cancer versus healthy and breast versus colorectal), with the downstream classifier selected separately.

## Results

We defined two binary classification tasks: cancer vs healthy (diagnosis) and breast vs colorectal cancer (cancer type).
Performance metrics (AUC - area under the receiver operating characteristic curve) are reported below for each task and model on the test and holdout splits.

### Tetramer-based classifiers

Table 1 summarizes ROC AUC on the test and holdout splits for each task for three models: a majority-class baseline, K nearest neighbors (KNN), and random forest.

<!-- classifier-table-1 -->
<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Cancer diagnosis AUC</th>
<th colspan="2">Cancer type AUC</th>
</tr>
<tr>
<th>Test</th><th>Holdout</th><th>Test</th><th>Holdout</th>
</tr>
</thead>
<tbody>
<tr>
<td>Majority class</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td>
</tr>
<tr>
<td>KNN</td><td>0.645</td><td>0.505</td><td>0.956</td><td>0.310</td>
</tr>
<tr>
<td>Random Forest</td><td>0.705</td><td>0.558</td><td>0.983</td><td>0.457</td>
</tr>
</tbody>
</table>

<!-- /classifier-table-1 -->

On the test split, KNN and RF both outperform the majority-class baseline for each task.
On the holdout split, both models have modest gains over the baseline for cancer diagnosis and score below the baseline for cancer type prediction.
RF achieves higher AUC than KNN in every test and holdout split.
