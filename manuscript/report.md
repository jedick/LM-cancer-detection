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

For SVM, we used the same centered log-ratio transform, standardization, and PCA construction as for KNN (including the same PCA candidate sizes and the rule that each candidate retained at least 90% of the variance on the training fold).
We fit a support vector machine with an RBF kernel and jointly tuned the PCA component count, the penalty parameter C, and the kernel width parameter gamma (scikit-learn’s *scale* versus *auto* settings), using only the validation split.

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
The sections are for different features used for classification: tetramer frequencies based on simple run-level aggregation and cluster abundance profiles derived from sequence-level unsupervised clustering.

### Tetramer-based classifiers

Table 1 summarizes ROC AUC on the test and holdout splits for each task for four models: a majority-class baseline, K nearest neighbors (KNN), a support vector machine (SVM), and random forest.

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
<td>KNN</td><td>0.661</td><td>0.521</td><td>0.948</td><td>0.327</td>
</tr>
<tr>
<td>SVM</td><td>0.718</td><td>0.575</td><td>0.999</td><td>0.272</td>
</tr>
<tr>
<td>Random Forest</td><td>0.706</td><td>0.590</td><td>0.982</td><td>0.535</td>
</tr>
</tbody>
</table>

<!-- /classifier-table-1 -->

For both tasks, all models outperform the majority-class baseline on the test split, especially for the cancer type prediction task.
The picture looks different on the holdout split.
Here, models show modest gains over the baseline for cancer diagnosis, while all models except for RF score below the baseline for cancer type prediction.
While SVM is the model with the best performance across tasks in the test split, only RF scores above the baseline in both holdout splits.

### UC/CAP-based classifiers

We explored different settings for *n*<sub>UC</sub> (number of sequences from each run used to build unsupervised clusters), *K* (number of clusters retained), and *n*<sub>CAP</sub> (number of sequences from each run assigned to cluster centroids).
Table 2 summarizes results of three classifiers on the *test data only* (in-study splits).

<!-- classifier-table-2 -->
<table>
<thead>
<tr>
<th rowspan="2"><i>n</i><sub>UC</sub></th>
<th rowspan="2"><i>K</i></th>
<th rowspan="2"><i>n</i><sub>CAP</sub></th>
<th colspan="3">Cancer diagnosis AUC (test)</th>
<th colspan="3">Cancer type AUC (test)</th>
</tr>
<tr>
<th>KNN</th>
<th>SVM</th>
<th>RF</th>
<th>KNN</th>
<th>SVM</th>
<th>RF</th>
</tr>
</thead>
<tbody>
<tr>
<td>1000</td><td>2000</td><td>5000</td><td>0.686</td><td>0.756</td><td>0.743</td><td>0.991</td><td>1.000</td><td>1.000</td>
</tr>
<tr>
<td>2000</td><td>2000</td><td>5000</td><td>0.698</td><td>0.789</td><td>0.744</td><td>0.996</td><td>1.000</td><td>0.999</td>
</tr>
<tr>
<td>1000</td><td>5000</td><td>5000</td><td>0.704</td><td>0.776</td><td>0.735</td><td>0.996</td><td>1.000</td><td>1.000</td>
</tr>
<tr>
<td>2000</td><td>5000</td><td>5000</td><td>0.705</td><td>0.784</td><td>0.737</td><td>0.978</td><td>1.000</td><td>1.000</td>
</tr>
<tr>
<td>1000</td><td>2000</td><td>10000</td><td>0.728</td><td>0.762</td><td>0.733</td><td>0.994</td><td>1.000</td><td>0.998</td>
</tr>
<tr>
<td>2000</td><td>2000</td><td>10000</td><td>0.736</td><td>0.790</td><td>0.741</td><td>0.997</td><td>1.000</td><td>1.000</td>
</tr>
<tr>
<td>1000</td><td>5000</td><td>10000</td><td>0.694</td><td>0.789</td><td>0.738</td><td>0.999</td><td>1.000</td><td>1.000</td>
</tr>
<tr>
<td>2000</td><td>5000</td><td>10000</td><td>0.676</td><td>0.770</td><td>0.741</td><td>0.993</td><td>1.000</td><td>1.000</td>
</tr>
</tbody>
</table>

<!-- /classifier-table-2 -->

The UC/CAP feature set generated with *n*<sub>UC</sub> = 2000, *K* = 2000, and *n*<sub>CAP</sub> = 10000 showed good performance across tasks and models, so we chose it for the remaining analysis.

Table 3 lists the AUC values for each model across tasks and splits (test or holdout).

<!-- classifier-table-3 -->
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
<td>KNN</td><td>0.736</td><td>0.466</td><td>0.997</td><td>0.500</td>
</tr>
<tr>
<td>SVM</td><td>0.790</td><td>0.664</td><td>1.000</td><td>0.488</td>
</tr>
<tr>
<td>Random Forest</td><td>0.741</td><td>0.573</td><td>1.000</td><td>0.380</td>
</tr>
</tbody>
</table>

<!-- /classifier-table-3 -->

For cancer diagnosis, SVM is the model with the best performance on holdout data.
For cancer type prediction, KNN is the only model that is on par with the baseline (majority-class) prediction.
Both SVM and Random Forest show sub-baseline performance on cancer type holdout data even though they achieved perfect classification on the in-study test splits.

## Discussion

Scores are consistently lower on the holdout splits, emphasizing over-optimistic metrics computed from in-study test splits.

The high test AUC scores >0.9 for cancer type prediction makes sense because the datasets for breast and colorectal cancer come from different studies.
The sample collection and sequencing is different from study to study, so the models have an easy time fitting to these differences.
In contrast, the lower test AUC scores near 0.7 for cancer diagnosis is expected for models that are challenged with distinguishing between cancer and healthy samples from the same studies.

Because the models learned differences between studies rather than biologically meaningful differences between cancer types, cancer type prediction has a huge performance drop for holdout data from studies the models haven't been trained on.
On the other hand, the performance is lower in the holdout split but often remains above baseline for cancer diagnosis, suggesting that the models learned biological differences between cancer and healthy samples that transfer to new studies.

Comparing the holdout splits in Tables 1 and 3, there is no clear net advantage to sequence clustering over run-level tetramer frequencies.
For cancer diagnosis, using UC/CAP instead of tetramer features leads to higher holdout AUC only for the SVM.
For cancer type, KNN and SVM achieve higher holdout AUC with UC/CAP (but still not above baseline),
whereas random forest shows a large holdout decline relative to the tetramer-based model despite near-perfect in-study test scores.
