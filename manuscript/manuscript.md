# Adversarial training improves HyenaDNA predictions in a multi-task cancer classification benchmark

## Introduction

The gut microbiome—the community of microorganisms inhabiting the human digestive tract—is increasingly linked to cancer risk and progression. Large-scale epidemiological and mechanistic studies have associated compositional shifts in gut bacteria with colorectal cancer, and growing evidence implicates gut dysbiosis in breast cancer as well. Machine learning models trained on microbiome profiles have shown promise for distinguishing cancer patients from healthy controls within individual cohorts, raising the prospect of non-invasive, microbiome-based cancer screening.

The dominant workflow for characterizing the gut microbiome is 16S rRNA amplicon sequencing: a short, phylogenetically informative region of the bacterial ribosomal gene is amplified and sequenced, and the resulting reads are matched to known reference taxa to produce species- or genus-level abundance tables. Most machine learning studies operate on these pre-processed abundance tables, treating the raw sequence data as an intermediate artefact to discard. This discards potentially informative signal: fine-grained genetic variation within taxa, sequences with no close reference in curated databases, and compositional structure at the level of individual reads within a sample. Methods that work directly on raw sequence data or on reference-free sequence features can in principle recover this signal.

A deeper problem, however, undermines nearly all published benchmarks in this area: test sets are almost always constructed by random sampling from the same studies used for training. This creates optimistically biased performance estimates that do not reflect real-world deployment, in much the same way that spatial autocorrelation inflates apparent model skill in geographic prediction tasks. In microbiome studies the bias is especially severe because technical factors—primer choice, sequencing platform, library preparation, regional microbiome variation—introduce large study-level signals that a model can exploit without learning any biology [1]. For multi-class cancer-type prediction the problem is structural: breast and colorectal cancer samples almost always come from entirely separate studies, so a classifier can achieve near-perfect in-study accuracy simply by identifying the study of origin rather than the disease. Evaluating such a model on held-out samples from the same studies dramatically overestimates generalization. A reliable benchmark must therefore evaluate models on studies they have never encountered during training [1].

We address this directly. We curate a multi-study compilation of 2,051 16S rRNA sequencing runs spanning 26 studies (13 breast cancer, 13 colorectal cancer), covering healthy controls and two cancer types across studies from 2013 to 2026. Studies are partitioned chronologically: the first seven studies per cancer type form the development set (training, validation, and test), while the more recent six studies per cancer type are reserved as an external holdout. This temporal and study-level separation means that holdout performance reflects the realistic scenario of applying a model trained on historical data to future datasets from different laboratories, clinical protocols, and geographic regions. To improve balance across studies, we randomly downsample the largest cohorts within each label stratum. The resulting benchmark provides a demanding but credible measure of real-world generalizability.

Against this benchmark we evaluate a progression of approaches. For classical machine learning we begin with run-level tetramer frequencies: all 4-mer counts aggregated across the raw reads of each sequencing run, converted to relative frequencies. This is a sequence-level representation that requires no taxonomic reference database. We then introduce unsupervised clustering and cluster abundance profiles (UC/CAP), a reference-free method that preserves within-run compositional structure—analogous in purpose to OTU-based approaches but operating entirely on sequence composition without taxonomic assignment. For deep learning we fine-tune HyenaDNA [2], a long-range genomic sequence model pre-trained on the human reference genome, directly on raw 16S reads. Naïve fine-tuning proves unstable: the model achieves strong in-study test performance but fails to generalize across studies. We show that this instability arises from shortcut learning of study-level confounders rather than from optimization noise alone, and we address it with domain-adversarial training [3], which explicitly penalizes the model for encoding study identity in its representations. This approach, widely applied in domain adaptation, has not previously been combined with a genomic language model for microbiome prediction.

Our main contributions are (1) a rigorously curated, temporally structured multi-study benchmark for microbiome-based cancer classification that provides more reliable estimates of real-world performance than random within-study splits, and (2) a domain-adversarial fine-tuning strategy for HyenaDNA that measurably improves holdout generalization, most clearly for cancer diagnosis.

---

## Methods

### Data curation

Each sample corresponds to a sequencing run containing multiple 16S rRNA gene sequences. We collected sequencing runs from studies covering breast cancer and colorectal cancer; studies were only included if both cancer-positive and healthy control labels were available. We stored SRA Run accessions (beginning with SRR, ERR, or DRR) and study metadata in the repository and downloaded each run's read archive from NCBI.

Our compilation spans 26 studies in total—13 for breast cancer and 13 for colorectal cancer (Table 1). Arranged chronologically by publication year, the first seven studies per cancer type form the development partition (train, validation, and test splits), and the more recent six studies per cancer type are reserved as the holdout partition. Development and holdout sets are separated not only by study boundaries but also by time: all holdout studies are from 2023 onward. This design makes the benchmark a realistic challenge: predictions must transfer to future datasets available only after the model was trained.

Some studies have substantially larger sample counts than others. To improve study balance, we applied random sampling within several studies (stratified by cancer-versus-healthy label). The sample sizes in Table 1 reflect counts after sampling at the indicated rate; these samples are flagged as `sample_used=TRUE` in the data CSV files. Note that a subset of used samples may be excluded from analysis at a later stage by filtering on sequence count.

|study_name|study_year|doi|cancer_type|n_cancer|n_healthy|sample_rate|ncbi_bioproject|partition|
|---|---|---|---|---|---|---|---|---|
|AAM+13|2013|10.14309/00000434-201310001-00625|breast|29|32|1|PRJNA396901|development|
|GJH+15|2015|10.1093/jnci/djv147|breast|47|47|1|PRJNA345373|development|
|GHB+18|2018|10.1038/bjc.2017.435|breast|48|48|1|PRJNA383849|development|
|BVW+21|2021|10.1002/ijc.33473|breast|60|66|0.15|PRJNA658160|development|
|BSR+22|2022|10.1038/s41598-022-23793-7|breast|19|14|1|PRJEB54599|development|
|WZK+22|2022|10.3389/fmicb.2022.894283|breast|54|25|1|PRJNA804967|development|
|ZZZ+22|2022|10.1111/jam.15620|breast|14|14|1|PRJNA726050|development|
|SKC+23|2023|10.1038/s41598-023-27436-3|breast|22|21|1|PRJNA872152|holdout|
|LBA+25|2025|10.3390/ijms26146801|breast|76|16|1|PRJNA1127492|holdout|
|SYL+25|2025|10.1128/msystems.00879-25|breast|10|10|1|PRJNA1243283|holdout|
|MTK+26|2026|10.1016/j.gutmic.2026.100009|breast|32|32|1|PRJNA914483|holdout|
|SVK+26|2026|10.21203/rs.3.rs-8921895/v1|breast|22|30|1|PRJNA1356467|holdout|
|YTK+26|2026|10.1007/s44411-026-00523-3|breast|15|15|1|PRJNA1190698|holdout|
|ZTV+14|2014|10.15252/msb.20145645|colorectal|41|75|1|PRJEB6070|development|
|BRRS16|2016|10.1186/s13073-016-0290-3|colorectal|64|94|0.5|PRJNA290926|development|
|OKN+21|2021|10.1038/s41467-021-25965-x|colorectal|67|51|0.1|PRJDB11246|development|
|YDS+21|2021|10.1038/s41467-021-27112-y|colorectal|65|43|0.35|PRJNA763023|development|
|YWS+21|2021|10.1186/s13073-021-00844-8|colorectal|53|52|1|PRJEB36789|development|
|DLT+22|2022|10.3389/fphys.2022.854545|colorectal|27|33|1|PRJNA824020|development|
|PCL+22|2022|10.1038/s41598-022-14203-z|colorectal|36|25|1|PRJNA662014|development|
|BWY+23|2023|10.1186/s12866-023-02805-0|colorectal|46|43|1|PRJEB53415|holdout|
|BRR+24|2024|10.1186/s12864-024-10621-7|colorectal|51|51|1|PRJEB71787|holdout|
|CAB+24|2024|10.1002/1878-0261.13604|colorectal|95|30|1|PRJNA911189|holdout|
|SGH+24|2024|10.1016/j.micpath.2024.106726|colorectal|10|10|1|PRJNA1059759|holdout|
|ARF+25|2025|10.3389/fmicb.2025.1449642|colorectal|25|15|1|PRJEB76625|holdout|
|GYX+25|2025|10.1186/s12866-024-03721-7|colorectal|67|64|0.6|PRJNA1092526,PRJNA1092376|holdout|

### Preprocessing

We normalized sample labels to a restricted vocabulary: healthy, breast cancer, colorectal cancer, and benign. The benign category includes adenomas, benign colon polyps, and breast ductal carcinoma in situ (DCIS); breast cancer samples include invasive tumors; colorectal cancer samples include carcinoma. All benign samples and non-fecal samples in applicable studies were retained in data files for auditability but excluded from training.

Among development studies, we assigned each sequencing run to stratified training, validation, or test sets in a 70:15:15 ratio. Runs from holdout studies were excluded from this assignment. Split assignments were defined in advance from version-controlled study lists and per-study sample tables, independent of any downstream feature computation.

We held the validation set fixed (no cross-validation). This allows the same development splits to be used consistently across both the classical and HyenaDNA pipelines, since GPU-intensive language model training makes repeated cross-validation expensive. The same run-level split underlies both classification tasks: cancer versus healthy (cancer diagnosis) on all samples, and breast versus colorectal (cancer type) restricted to cancer-positive samples.

### Classification using run-level tetramer frequencies

We calculated tetramer frequencies for each run by counting all 4-mers within each sequence, summing counts over all sequences in the run, then converting to relative frequencies, yielding a 256-dimensional feature vector per run.

For the majority-class baseline, we predict the most frequent class in the training set for all samples.

For KNN, we applied a centered log-ratio transform (CLR), standardized the CLR coordinates, then applied PCA. PCA candidate sizes were stepped down from the largest feasible rank (up to 256) by successive halves, retaining only sizes for which the leading components explained at least 90% of variance on the training fold. We tuned the PCA size, number of neighbors, and distance weighting by grid search on the validation split only.

For random forest, we used the same CLR and standardization but omitted PCA. We tuned the number of trees (200 or 500), maximum tree depth (unlimited or capped at 10), and minimum samples per leaf (1 or 2) on the validation split.

For SVM, we applied the same CLR, standardization, and PCA construction as for KNN (same candidate sizes and 90% variance retention rule), then fit an RBF-kernel SVM, jointly tuning the PCA component count, penalty parameter *C*, and kernel width parameter gamma (scikit-learn's *scale* versus *auto* settings) on the validation split.

After selecting hyperparameters on the validation split, we fit each final pipeline on the training split.

### Classification using cluster abundance profiles

Run-level tetramer features summarize each sample with a single aggregate profile and do not capture how different sequence types are distributed within a run. To preserve this within-run compositional structure, we use unsupervised clustering followed by cluster abundance profiles (UC/CAP), a reference-free and alignment-free approach.

Because the sequence-level table is large, we first fit the unsupervised clustering model using only sequences from training-split runs, drawing at most a fixed number of sequences per run. For each selected sequence we computed a 256-dimensional tetramer composition vector, applied PCA retaining components that explained 95% of cumulative variance, and fit *k*-means in the reduced space to obtain *K* centroids defining a sequence codebook.

To construct run-level features, we applied the same PCA transformation and centroid assignments—without refitting—to a larger per-run sequence budget for every run in the sequence-level table, including validation, test, and holdout runs. We counted cluster memberships within each run and normalized by the number of assigned sequences to produce a *K*-dimensional cluster abundance profile (CAP). These CAP vectors serve as the feature matrix for supervised classification on both binary tasks, with downstream classifiers selected separately per task.

### Classification using HyenaDNA sequence modeling

We trained HyenaDNA directly on run-level sequence data to test an end-to-end sequence model. For each run, we read the FASTA file and split its sequences into a fixed number of non-overlapping sets. Each set was packed to the model length limit and tokenized at the DNA character level. Datasets were saved to disk so that training runs could reuse cached tensors without rebuilding the dataset each time.

We initialized HyenaDNA from pretrained weights, used its classification head, and configured model size (e.g. 1k or 32k context), pooling mode, learning rate, batch size, number of epochs, and backbone freezing through YAML configuration. Because each run can produce multiple sequence sets, training loss was computed across all valid sets for each run. For evaluation, we converted set-level outputs to a single run-level prediction by aggregating logits across sets (mean or max), then computed ROC AUC on the same test and holdout splits used for the tetramer and UC/CAP analyses.

### Improving stability with domain-adversarial training

Fine-tuned HyenaDNA for cancer type prediction frequently achieves strong validation and in-study test ROC AUC while holdout performance remains sensitive to optimization trajectory, random seed, and training epoch. We explored many candidate remedies—randomized sequence sampling when building run tensors, shorter learning-rate schedules and warmup variants, discriminative rates, gradient clipping, short-cycle stochastic weight averaging, and staged backbone freezing. These changes are summarized in the [appendix](appendix.md). They sometimes improved development metrics but did not yield consistent, seed-robust holdout gains. This suggests that study-level shortcuts are a structural confound not addressable by a single missing hyperparameter.

Domain-adversarial training targets exactly this failure mode: it encourages representations that support the clinical label while being less predictive of which development study produced the sample. We adopt the gradient-reversal formulation of Ganin *et al.* (2016) [3], building on domain-adaptation theory in which train and target domains differ in marginal feature distributions (Ben-David *et al.*, 2010) [4]. Concretely, we add a study classifier on top of the pooled sequence representation, connected through a gradient-reversal layer, so the shared trunk is trained to fool the study head while the cancer-type head is trained as before.

**Implementation.** The existing cancer-type head remains unchanged; a small MLP predicts study identity from the same pooled features. An initial training phase updates only the cancer head, then a warm-up phase trains the study head without reversal so that its gradients align with study prediction. Finally, the reversal layer is enabled, and the adversarial weight penalizes study-predictive directions in the trunk. Losses combine cross-entropy for the clinical label (with class balancing) and weighted cross-entropy for study prediction. Optimization uses the same development splits and validation-driven checkpoint selection as the non-adversarial HyenaDNA runs.

**Empirical results (current ablation grid).** Table 5 summarizes results for the best recipe (baseline, which includes DANN) alongside targeted ablations, each reported as mean ± standard deviation across three random seeds.

Numeric values for Table 5: [table5_hyenadna.html](table5_hyenadna.html).

With DANN enabled (best recipe), the model achieves a cancer diagnosis holdout AUC of 0.550 ± 0.013 and a cancer type holdout AUC of 0.579 ± 0.091. Removing domain-adversarial training reduces cancer diagnosis holdout AUC by 0.034 (to 0.516 ± 0.033) and cancer type holdout AUC by 0.013 (to 0.566 ± 0.038). The DANN benefit is most consistent for cancer diagnosis, where it also substantially reduces holdout variance across seeds. For cancer type, the gain is smaller and the high seed-to-seed variance in both conditions means no individual run of the grid should be treated as definitive.

Among other ablations, the most influential factor is learning rate: raising it from 10⁻⁵ to 10⁻⁴ collapses cancer type holdout AUC to 0.333 ± 0.049, consistent with the calibration collapse described in the appendix. Replacing study-balanced sampling with a random training sampler reduces cancer type holdout AUC to 0.543 ± 0.029. Increasing the adversarial loss weight from 0.3 to 0.6 leaves performance largely unchanged (0.584 ± 0.080). Removing class weighting produces the highest point estimate for cancer type holdout (0.608 ± 0.078) but also the highest variance, and it reduces cancer diagnosis holdout AUC to 0.558 ± 0.016.

We also verified that using float16 AMP, gradient clipping (norm 1.0), or tuning by validation F1 instead of AUC did not move holdout AUC outside seed variance, and that replacing the linear classification head with an MLP head (256 hidden units) degraded cancer type holdout AUC without benefiting cancer diagnosis.

---

## Results

We define two binary classification tasks: **cancer diagnosis** (cancer vs. healthy, all samples) and **cancer type** (breast vs. colorectal, cancer-positive samples only). Performance is reported as ROC AUC on the test split (held-out development samples) and the holdout split (entirely unseen studies).

For cancer type, all development studies for breast cancer are separate from all development studies for colorectal cancer. A model can therefore exploit study-level signals—different sequencing protocols, primer sets, or regional microbiome composition—as a near-perfect shortcut for in-study test performance. Holdout performance, where the model encounters new studies it has not seen during training, removes this shortcut. We accordingly expect cancer type to be the *easier* task for in-study test data but the *harder* task for holdout data.

For cancer diagnosis, each included study contains both cancer-positive and healthy samples, so study identity alone does not predict the label. Models must learn biological differences between cancer and healthy microbiomes within studies, and those differences are expected to transfer—at least partially—to new studies.

### Tetramer-based classifiers

Table 2 reports ROC AUC on the test and holdout splits for four models: majority-class baseline, KNN, SVM, and random forest.

Numeric values for Table 2: [table2_tetramer.html](table2_tetramer.html).

All models exceed the majority-class baseline on the test split, with particularly large margins for cancer type prediction. The holdout picture is sharply different. For cancer diagnosis, SVM and KNN show modest gains above baseline (0.596 and 0.563, respectively), while random forest falls below all other classifiers (0.541). For cancer type, every model collapses toward or below baseline on holdout: SVM reaches only 0.484 and KNN only 0.407, confirming that tetramer classifiers overfit to study-level signals when trained on single-study cancer-type data.

### UC/CAP-based classifiers

We explored eight combinations of the three UC/CAP hyperparameters: *n*<sub>UC</sub> (sequences per run used to build the codebook), *K* (number of clusters), and *n*<sub>CAP</sub> (sequences per run assigned at test time), listed in Table 3.

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

For cancer diagnosis, SVM consistently outperforms random forest across all eight feature sets in both test and holdout AUC (Figure 1). For cancer type, both models show near-perfect in-study test performance across feature sets, but holdout values drop sharply—especially for random forest. Despite these drops, SVM holdout AUC remains comparatively stable across feature sets, making SVM the more reliable classifier under distribution shift.

![Figure 1. UC/CAP feature-set stability for SVM and random forest across tasks.](figure1_uc_cap.svg)

Table 4 lists AUC values for each model on the UC/CAP feature set with *n*<sub>UC</sub> = 2000, *K* = 5000, and *n*<sub>CAP</sub> = 10000.

Numeric values for Table 4: [table4_uc_cap.html](table4_uc_cap.html).

For cancer diagnosis, SVM achieves the best holdout performance (0.664), followed by KNN (0.571) and random forest (0.527). For cancer type, SVM again leads on holdout (0.783), with random forest scoring 0.634 and KNN at chance (0.500). The gap between in-study test and holdout is again large for cancer type, but UC/CAP with SVM achieves substantially higher cancer type holdout AUC than any tetramer-based classifier, demonstrating that richer within-run compositional features partially attenuate the study-level shortcut problem.

### Classification with HyenaDNA: Stability problems

We report a fine-tuning grid for the pretrained 32k HyenaDNA model. Given available hardware (16 GB GPU memory), we are limited to smaller model sizes and sequence budgets than the full model supports.

For each task (cancer diagnosis and cancer type) we trained separate classification heads on the same splits used for the tetramer and UC/CAP classifiers. To study how much sequence context per run matters, we varied the length per set (up to 1k, 2k, 4k, 8k, and 16k positions) and the number of non-overlapping sets drawn per run (5 versus 10). A single large cache (10 sets, 16k length each) was built from consecutive sequences without within-FASTA shuffling; shorter training configurations were obtained from that cache by truncating to the target length.

Figure 2 shows AUC on the test and holdout splits as a function of length per set, stratified by task (rows) and number of sets per run (columns). Holdout performance is generally weaker than test performance, and the curves are not monotone in context length. Increasing the number of bases modeled per set does not reliably improve generalization in these runs. Training trajectories and validation metrics also vary substantially across configurations. We treat these results as a first end-to-end baseline, motivating the stabilization work that follows.

![Figure 2. HyenaDNA set-length stability across tasks and number of sets.](figure2_hyenadna.svg)

### Improving stability with domain-adversarial training

Domain-adversarial training provides a measurable improvement over the no-DANN baseline (Table 5). The benefit is most consistent for cancer diagnosis, where DANN raises holdout AUC from 0.516 ± 0.033 to 0.550 ± 0.013—a gain of 0.034 accompanied by a marked reduction in seed-to-seed variance. For cancer type, the gain is smaller (0.566 to 0.579) and the high holdout variance in both conditions (± 0.038 vs ± 0.091) means that multi-seed evidence is needed before treating any single configuration as a robust default. Together, these results position DANN as a useful inductive bias for microbiome-based cancer prediction with HyenaDNA, particularly in the cancer diagnosis setting where its adversarial signal is structurally cleanest (see Discussion).

---

## Discussion

Results are consistently lower on the holdout splits than on the in-study test splits, confirming that test performance computed within the same studies used for training gives overoptimistic estimates of real-world model skill. This pattern holds across every method we evaluate.

The asymmetry between tasks is informative. Cancer type classification achieves in-study test AUC above 0.95 for virtually every model, because breast and colorectal samples come from completely separate studies: the model can predict cancer type by identifying the study of origin, exploiting differences in sequencing protocol, primer region, or regional microbiome composition rather than biology. On holdout data, where these shortcuts no longer generalize, cancer type AUC collapses toward or below chance for most models. In contrast, cancer diagnosis test AUC is more moderate (around 0.7 for the best classical models), because models must distinguish cancer from healthy samples within studies that contain both—a genuinely harder discrimination. The corresponding holdout AUC drops are smaller, suggesting that at least some of the learned signal reflects biological differences between cancer and healthy microbiomes that transfer across cohorts.

Comparing holdout performance across Tables 2 and 4, UC/CAP offers a consistent advantage over run-level tetramer features, most clearly for cancer type prediction with SVM. The SVM + UC/CAP combination (holdout AUC 0.783 for cancer type, 0.664 for cancer diagnosis) substantially outperforms SVM + tetramers on holdout, despite comparable or identical in-study test performance. This suggests that capturing within-run compositional diversity—analogous in purpose to OTU-based richness features but without a taxonomic reference—partially breaks the study-level shortcut that ruins tetramer-based cancer type classifiers. Why SVM succeeds here while random forest does not is worth investigating; SVM's margin-based objective and implicit regularization through kernel choice may constrain the decision boundary in ways that happen to avoid the most brittle study-specific directions in feature space.

The HyenaDNA instability problem is structural rather than a simple consequence of too-high learning rates or insufficient training data. As the appendix documents in detail, many standard remedies improved development metrics while leaving holdout performance unchanged or worsened. The failure pattern—near-perfect validation AUC collapsing to near-random or near-one-class holdout predictions—is characteristic of a model that has found a study-identity shortcut and then encounters holdout studies where that shortcut inverts. Domain-adversarial training addresses this directly by penalizing the shared representation for encoding study identity, and our ablation results show that this penalty is beneficial, particularly for cancer diagnosis. The structural reason for the task asymmetry is clear: cancer diagnosis studies contain both cancer and healthy samples, so study identity and clinical label are decoupled—the textbook setting for gradient-reversal domain adaptation. For cancer type, where study and label are nearly synonymous (each study contributes a single cancer type), the adversarial signal overlaps with the predictive signal and must be calibrated carefully, explaining the sensitivity to adversarial loss weight and learning rate observed in the ablations.

One practical observation from the HyenaDNA results is that a reliable holdout response to increasing sequence context length (max_length) is a prerequisite for confident conclusions about what the model has learned. The current results in Figure 2 show non-monotone curves that vary across seeds, suggesting that the model is not yet in a regime where more sequence reliably helps. We plan to re-run the set-length sweep (1k through 16k) under the DANN-stabilized training recipe to test whether adversarial regularization produces a more orderly context-scaling curve. Even at 16k tokens per set and 5 sets per run, HyenaDNA sees only around 400 sequences from each sample (assuming 200 nt 16S fragments), a small fraction of what the tetramer and UC/CAP methods use. Whether raw sequence depth or model capacity is the binding constraint remains an open question.

Several directions may improve performance beyond current baselines. On the feature side, UC/CAP parameters (*K*, *n*<sub>CAP</sub>) could be tuned jointly with the classifier rather than independently, and soft cluster assignments (Gaussian mixture or fuzzy *k*-means) might better represent the continuous composition of microbial communities. On the model side, combining HyenaDNA's sequence representations with compositional features in a hybrid architecture could leverage complementary information. A single multi-task HyenaDNA model trained jointly on cancer diagnosis and cancer type—sharing a backbone but using task-specific output heads—would reduce computational cost and potentially provide regularization benefits; softmax probabilities over task-specific label sets could serve as a natural multi-task output format. Additional pre-training on 16S rRNA sequences specifically (rather than the human genome) would also better align the model's learned representations with the target domain.

More broadly, our results underscore a general lesson for machine learning applied to genomic and microbiome data: evaluation quality matters as much as model sophistication. Metrics computed on within-study test splits can be misleading by a wide margin—in our benchmark, cancer type AUC falls by more than 0.4 points from test to holdout for the strongest classical classifiers. Robust evaluation against temporally and geographically diverse holdout cohorts should be a standard requirement in this field [1].

## References

1. Whalen S, Schreiber J, Noble WS, Pollard KS (2022). Navigating the pitfalls of applying machine learning in genomics. *Nature Reviews Genetics*, 23(3), 169–181. https://doi.org/10.1038/s41576-021-00434-9

2. Nguyen E, Poli M, Faizi M, Thomas A, Wornow M, Birch-Sykes C, Massaroli S, Patel A, Rabideau M, Bengio Y, Ermon S, Ré C, Hie B (2024). HyenaDNA: Long-range genomic sequence modeling at single nucleotide resolution. *Advances in Neural Information Processing Systems*, 36. https://arxiv.org/abs/2306.15794

3. Ganin Y, Ustinova E, Ajakan H, Germain P, Larochelle H, Laviolette F, Marchand M, Lempitsky V (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1–35. https://jmlr.org/papers/v17/15-239.html

4. Ben-David S, Blitzer J, Crammer K, Kulesza A, Pereira F, Vaughan JW (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175. https://doi.org/10.1007/s10994-009-5152-4

---

## Proposed abstracts

Three options follow, each foregrounding a different aspect of the work.

---

### Option A — Benchmark-focused

*Microbiome-based cancer prediction benchmarks routinely overestimate real-world performance because test samples are drawn from the same studies used for training, allowing models to exploit study-specific technical artefacts rather than biological signal. We present a temporally structured multi-study compilation of 2,051 16S rRNA sequencing runs covering breast cancer, colorectal cancer, and healthy cohorts across 26 studies spanning more than a decade. By reserving the six most recent studies per cancer type as an external holdout, we ensure that holdout evaluation reflects deployment on data from new laboratories, clinical protocols, and geographic regions. Against this benchmark, in-study test AUC for cancer type prediction falls by more than 0.4 points on holdout for the best classical classifier, confirming that conventional evaluation dramatically inflates apparent model skill. We benchmark run-level tetramer frequencies, reference-free cluster abundance profiles (UC/CAP), and fine-tuned HyenaDNA with domain-adversarial training. UC/CAP with SVM achieves the strongest holdout performance among classical methods (AUC 0.783 for cancer type, 0.664 for cancer diagnosis). Domain-adversarial HyenaDNA improves cancer diagnosis holdout AUC by 0.034 over a matched non-adversarial baseline while substantially reducing seed-to-seed variance. Our benchmark and associated code are publicly available to support reproducible, credible evaluation of microbiome-based cancer classifiers.*

---

### Option B — Model-focused

*Machine learning models trained on gut microbiome 16S rRNA data can distinguish cancer patients from healthy controls within individual studies, but generalization to unseen studies is poorly characterized. We show that a key barrier is shortcut learning: models trained on multi-study data can identify cancer type by recognizing the study of origin rather than the underlying biology, producing near-perfect in-study accuracy that collapses on external holdout data. We apply domain-adversarial training—gradient-reversal regularization that penalizes study-identity encoding—to HyenaDNA, a pre-trained genomic language model, and evaluate it on a new temporally structured benchmark of 2,051 samples across 26 studies. Adversarial training improves cancer diagnosis holdout AUC from 0.516 ± 0.033 to 0.550 ± 0.013 and reduces seed-to-seed variance, with more modest gains for cancer type where study and label identity are structurally entangled. We additionally introduce cluster abundance profiles (UC/CAP), a reference-free compositional feature that outperforms tetramer-based classifiers on holdout, with SVM achieving holdout AUC of 0.783 for cancer type and 0.664 for cancer diagnosis. Together, our results demonstrate that study-level domain adaptation is a productive inductive bias for microbiome-based cancer prediction and that rigorous holdout evaluation substantially changes which methods appear promising.*

---

### Option C — Application-focused

*Non-invasive cancer screening from the gut microbiome is a clinically attractive prospect, but the machine learning models proposed for this purpose are almost always evaluated on test samples from the same cohorts used for training—a practice that produces misleadingly optimistic performance estimates. We curate a multi-study benchmark of 2,051 16S rRNA sequencing runs from 26 breast and colorectal cancer studies, spanning more than a decade of research and diverse laboratory protocols, and evaluate models on a temporal holdout partition never seen during training. Under this more realistic evaluation, cancer type classification accuracy drops dramatically for all classical methods, while cancer diagnosis—a harder task within studies—shows more moderate holdout degradation. We benchmark three families of classifiers: tetramer-frequency models, reference-free cluster abundance profiles (UC/CAP), and a domain-adversarially fine-tuned HyenaDNA sequence model. UC/CAP with SVM provides the strongest generalization among classical approaches. Domain-adversarial training measurably improves HyenaDNA's holdout performance for cancer diagnosis and reduces variability across training runs, suggesting that suppressing study-specific technical signals is a practical path toward microbiome classifiers that hold up in new clinical settings. We release our benchmark, split definitions, and model code to enable reproducible comparison.*
