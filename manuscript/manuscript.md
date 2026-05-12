# Cancer detection with gut microbiomes using a DNA language model

## Introduction

For `cancer_diagnosis` each dataset has two labels (cancer or healthy).
The controlled experimental conditions within a given study allow the model to learn patterns in sequences that distinguish between cancer and healthy samples.
Although these distinguishing patterns are small (e.g. changes in species abundance) they represent biological signal that may transfer to new datasets.

For `cancer_type` each dataset has a single label (all breast or all colorectal),
so the model inevitably learns study-level confounders rather than pure biological signal.
These study-level differences impart large signals in the sequences (e.g. different species and regions of the 16S gene)
that likely overwhelm the biological differences between cancer types.
This is why we expect `cancer_type` to be the *easier* task for in-study test data but the *harder* task for holdout data from unseen studies.

The first reads in a FASTA are more likely to carry consistent study-specific artifacts
(adapter remnants, library prep signatures, quality patterns from the sequencer).
A model trained on these reads could achieve artificially high holdout AUC if holdout studies share similar sequencing protocols with development studies.

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
Among development studies only, we assigned each sequencing run to stratified training, validation, or test sets in a 70:15:15 ratio.
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

### Classification using HyenaDNA sequence modeling

We trained HyenaDNA directly on run-level sequence data to test an end-to-end sequence model.
For each run, we read the FASTA file and split its sequences into a fixed number of non-overlapping sets.
Each set was packed to the model length limit and tokenized at the DNA character level.
Datasets were saved to disk so training runs could reuse cached tensors instead of rebuilding the dataset each time.

We initialized HyenaDNA from pretrained weights, used its classification head, and selected model size (for example 1k or 32k context),
pooling mode, learning rate, batch size, number of epochs, and whether to freeze the backbone through YAML configuration.
Because each run can produce multiple sequence sets, training loss was computed across all valid sets from each run.
For evaluation, we converted set-level outputs to one run-level prediction by aggregating logits across sets (mean or max).
We then computed ROC AUC on the same test and holdout splits as used in the tetramer and UC/CAP analyses.

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
<td>KNN</td><td>0.652</td><td>0.563</td><td>0.959</td><td>0.407</td>
</tr>
<tr>
<td>SVM</td><td>0.725</td><td>0.596</td><td>1.000</td><td>0.484</td>
</tr>
<tr>
<td>Random Forest</td><td>0.701</td><td>0.541</td><td>0.986</td><td>0.532</td>
</tr>
</tbody>
</table>

<!-- /classifier-table-1 -->

For both tasks, all models outperform the majority-class baseline on the test split, especially for the cancer type prediction task.
The picture looks different on the holdout split.
Here, models show modest gains over the baseline for cancer diagnosis, while all models except for RF score below the baseline for cancer type prediction.
While SVM is the model with the best performance across tasks in the test split, only RF scores above the baseline in both holdout splits.

### UC/CAP-based classifiers

We explored different settings for *n*<sub>UC</sub> (number of sequences from each run used to build unsupervised clusters), *K* (number of clusters retained), and *n*<sub>CAP</sub> (number of sequences from each run assigned to cluster centroids), summarized in Table 2.

| Feature set | *n*UC | *K* | *n*CAP |
|-|-|-|-|
| 1 | 1000 | 2000 | 5000 |
| 2 | 2000 | 2000 | 5000 |
| 3 | 1000 | 5000 | 5000 |
| 4 | 2000 | 5000 | 5000 |
| 5 | 1000 | 2000 | 10000 |
| 6 | 2000 | 2000 | 10000 |
| 7 | 1000 | 5000 | 10000 |
| 8 | 2000 | 5000 | 10000 |

For cancer diagnosis, SVM shows higher test and holdout AUC than random forest across all eight UC/CAP feature sets (Figure 1).
For cancer type prediction, both models show near-perfect in-study test performance, but holdout values drop sharply, especially for random forest.
Even with these drops, holdout AUC remains generally stable across feature sets, with SVM showing greater stability than random forest.

![Figure 1. UC/CAP feature-set stability for SVM and random forest across tasks.](figures/figure1_uc_cap.svg)

Table 3 lists AUC values for each model across tasks and splits (test and holdout) for the UC/CAP feature set with *n*<sub>UC</sub> = 2000, *K* = 5000, and *n*<sub>CAP</sub> = 10000.

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
<td>KNN</td><td>0.719</td><td>0.571</td><td>0.998</td><td>0.500</td>
</tr>
<tr>
<td>SVM</td><td>0.769</td><td>0.664</td><td>1.000</td><td>0.783</td>
</tr>
<tr>
<td>Random Forest</td><td>0.710</td><td>0.527</td><td>1.000</td><td>0.634</td>
</tr>
</tbody>
</table>

<!-- /classifier-table-3 -->

For cancer diagnosis, SVM has the best holdout performance.
For cancer type prediction, SVM has the best holdout performance, and random forest also scores above baseline.
KNN remains on par with the baseline (majority-class) prediction, while SVM and random forest stay far below their near-perfect in-study test performance.

### Classification with HyenaDNA: Stability problems

We report a fine-tuning grid for the pretrained 32k HyenaDNA model.
Even though HyenaDNA offers larger max-length models, with available hardware we can only use part of the sequence data for each run.
For each task (cancer diagnosis and cancer type) we trained separate heads on the same test and holdout splits used for the tetramer and UC/CAP classifiers, and we summarize ROC AUC.

To study how much sequence context per run matters, we used sets of sequences from each run FASTA file.
We varied the length per set (up to 1k, 2k, 4k, 8k, and 16k positions) and the number of non-overlapping sets drawn per run (5 versus 10).
A single large cache (10 sets each with 16k length) was built from consecutive sequences without shuffling within a FASTA file.
Shorter training configurations were obtained from that cached pack by truncating to each target length.

Figure 2 shows AUC on the test and holdout splits as a function of length per set, stratified by task (rows) and by the number of sets per run (columns).
Holdout performance is generally weaker than test performance, and the curves are not monotone in context length.
Increasing the number of bases modeled per set does not reliably improve generalization in these runs.
Training trajectories and validation metrics can also vary substantially between configurations, so we treat this section as a first end-to-end baseline that we expect to revise with additional data cleaning and model training choices.

![Figure 2. HyenaDNA set-length stability across tasks and number of sets.](figures/figure2_hyenadna.svg)

### Improving stability with domain adversarial training

Fine-tuned HyenaDNA for cancer type often achieves strong validation and in-study test ROC AUC while holdout performance remains sensitive to optimization trajectory, random seed, and training epoch.
Prior attempts to stabilize behavior are summarized in the [appendix](appendix.md): for example,
randomized sequence sampling when building run tensors (to avoid head-of-file artifacts),
shorter learning-rate schedules and warm-up variants, discriminative rates and gradient clipping,
short-cycle SWA, staged backbone freezing, and frozen pretrained embeddings with classical heads.
Those changes sometimes improved development metrics but did not yield a consistent, seed-robust holdout gain.
This suggest that study-level shortcuts are a structural confound that are not solvable using a single missing hyperparameter.

Domain-adversarial training targets exactly that failure mode: it encourages representations that support the clinical label
while being less predictive of which development study produced a sample.
We adopt the gradient-reversal formulation of Ganin *et al.* (2016), building on domain-adaptation perspectives
in which train and target domains differ in marginal feature distributions (Ben-David *et al.*, 2010).
Concretely, we add a study classifier on top of the pooled sequence representation,
connected through a gradient-reversal layer, so the shared trunk is trained to fool the study head while the cancer-type head is trained as before.

**Implementation (high level).**
The existing cancer-type head remains; a small MLP predicts study identity from the same pooled features.
An initial training phase updates only the cancer head, then a warm-up phase trains the study head without reversal (so gradients agree with study prediction).
Finally, the reversal layer is enabled so adversarial weighting penalizes study-predictive directions in the trunk.
Losses combine cross-entropy for cancer type (with class balancing as elsewhere) and weighted cross-entropy for study prediction.
Optimization uses the same development splits and validation-driven checkpoint selection as other HyenaDNA runs.

**Empirical takeaways (cancer type, current grid).**
For initial DANN model tuning, we compared validation-selected best epochs on the same random seed.
At learning rate \(10^{-4}\), adversarial training tended to hurt or erase holdout AUC relative to matched controls,
whereas at \(10^{-5}\) we could obtain better holdout ROC AUC, suggesting strong interaction with optimization strength.
Other knobs we explored were less rewarding: lengthening the study-head warm-up before enabling reversal,
adjusting the adversarial loss weight, and switching the cancer head to last-token pooling instead of the default pooled representation
either reduced holdout performance or failed to show a clear benefit in the experiments we examined.

Holdout trajectories remain noisy from epoch to epoch under both DANN and no-DANN training on a single seed.
Multi-seed comparisons at \(10^{-5}\) still show holdout variation, so we report DANN as a helpful inductive bias in this setting, not as a guarantee of stable curves.

We consolidate these HyenaDNA cancer-type ablations (best recipe, higher learning rate, DANN off, class-weighting and study-balance variants,
baseline architecture choices, and related controls) in **Table (TBD: HyenaDNA DANN / ablation grid)**.

**TODOs**

1. Populate the HyenaDNA DANN / ablation table from finalized JSON metrics (including at least two seeds where applicable).
2. After the ablation grid is set, re-run the sequence-set length axis (1k, 2k, 4k, 8k, 16k) under the chosen training recipe for a direct comparison to Figure 2.
3. Revisit `cancer_diagnosis` with the same adversarial idea if resources allow.
   Label structure differs from cancer type (within-study cancer vs. healthy), so transfer behavior may not mirror the cancer-type experiments.

## Discussion

Scores are consistently lower on the holdout splits, emphasizing over-optimistic metrics computed from in-study test splits.

The high test AUC scores >0.9 for cancer type prediction makes sense because the datasets for breast and colorectal cancer come from different studies.
The sample collection and sequencing is different from study to study, so the models have an easy time fitting to these differences.
In contrast, the lower test AUC scores near 0.7 for cancer diagnosis is expected for models that are challenged with distinguishing between cancer and healthy samples from the same studies.

Because the models learned differences between studies rather than biologically meaningful differences between cancer types,
cancer type prediction has a huge performance drop for holdout data from studies the models haven't been trained on.
On the other hand, the performance is lower in the holdout split but often remains above baseline for cancer diagnosis,
suggesting that the models learned biological differences between cancer and healthy samples that transfer to new studies.

Comparing the holdout splits in Tables 1 and 3, UC/CAP shows an overall advantage over run-level tetramer frequencies, most clearly for cancer type and for SVM.
Although SVM and random forest show near-perfect in-study test performance for cancer type prediction in both pipelines,
only UC/CAP yields substantial improvements above baseline on holdout studies.
This result illustrates the challenge of transferring learned patterns to new datasets and highlights SVM as the strongest model in our current set.

The training stability problem on HyenaDNA is structural: cancer type prediction is dominated by study-level confounders
(different sequencing protocols for breast vs. colorectal studies).
HyenaDNA is expressive enough to memorize those confounders, while SVM+UC/CAP is constrained enough that it can't
[nb. we need to explicitly check this - try smaller nUC in the range of HyenaDNA dataset sizes].
The instability isn't just optimization noise; it's the model finding different study-identity shortcuts on different seeds
and then collapsing on holdout studies with different protocols.
The [appendix](appendix.md) outlines the hypotheses and experiments we undertook to address this problem.

## References

1. Ganin, Y., Ustinova, H., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1–35. https://jmlr.org/papers/v17/15-239.html

2. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175. https://doi.org/10.1007/s10994-009-5152-4
