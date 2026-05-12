# Appendix

## HyenaDNA Stability Configuration and Tuning Log

### Problem Statement and Benchmark Context
We observed repeated instability in HyenaDNA for cancer type prediction. Validation and test AUC were often high in development studies, but holdout AUC varied widely across seeds and epochs. This pattern contrasts with UC/CAP plus SVM, which reached a stronger and more stable holdout AUC for cancer type prediction. That benchmark indicates transferable signal exists in this task, so the HyenaDNA issue is not simply an impossible prediction problem.

### Symptoms That Motivated the Stability Work
The strongest failure pattern was calibration collapse on holdout studies. In some runs, the model predicted almost every holdout run as colorectal cancer; in others, it predicted almost every holdout run as breast cancer. These flips occurred even when validation and test AUC stayed high. In the same runs, we also saw large gradient norm spikes and abrupt changes in class-score moments, which suggests unstable optimization dynamics rather than only noisy data sampling.

### Data Processing Hypothesis: Sequence Selection Bias in Cached Tensors
Early HyenaDNA results were generated from cached tensors that effectively followed a head-of-file sequence policy. We then introduced random sequence sampling during run-tensor cache construction and verified the cache with audit reports. The rebuilt cache used random sampling across all cached runs with a median selected-sequence count of 8000. This change removed one deterministic bias source, but by itself it did not eliminate holdout instability.

### What Helped Most So Far
Study-balanced training sampling and class-weighted losses were the two interventions with the clearest positive effect on cancer type holdout transfer. Some runs under these recipes reached substantially better holdout AUC than earlier controls, including longer runs. At the same time, these gains were not consistently reproducible across seeds, so they remain candidate levers rather than a locked default.

### What Did Not Help Reliably
Several common model-side changes did not produce consistent improvements in holdout stability: discriminative learning rates, SWA in short runs, lower learning rate without additional controls, and warmup-cosine schedules in the tested settings. These changes often increased validation and test AUC without a matching holdout gain. In multiple runs, they still converged to near-one-class predictions on holdout studies.

### Operational Findings and Debugging Notes
We identified and fixed an AMP plus class-weight interaction bug. With bfloat16 AMP and class-weighted loss, cross-entropy initially failed due to a dtype mismatch between logits and class weights. We patched the training script to cast class weights to logits dtype and device before loss computation.

We also tested study-level validation ROC as a tuning target. In the current split, each validation study for cancer type is single-class, so per-study ROC is undefined and study-macro validation ROC becomes NaN. We retained study-level metrics as diagnostics, but reverted active checkpoint tuning to validation ROC for immediate experiments.

### Current Interpretation and Next-Step Hypothesis
Current evidence supports an optimization-and-calibration instability hypothesis. The model can fit strong development discrimination but remains sensitive to trajectory and tends to overcommit to study-associated class structure. Across many interventions, conclusions can change substantially with random seed, so no single configuration can yet be treated as a robust default. The next experiments prioritize reducing optimization variance and calibration collapse while keeping data processing fixed and the experiment grid small. After stabilizing cancer type behavior, we will transfer the selected training recipe to cancer diagnosis.

### Classification Using HyenaDNA Fixed Embeddings
We ran an orthogonal check that bypasses HyenaDNA fine-tuning: extract run-level features from a pretrained backbone and fit the downstream classifiers on those frozen representations. The goal was to test whether pretrained sequence representations alone carry transferable signal.

Development behavior differed by task: cancer type showed stronger in-study discrimination than cancer diagnosis, while cancer diagnosis remained comparatively modest. In both tasks, however, development signal did not reliably carry to holdout, and SVM did not consistently outperform random forest. Fixed embeddings remain a useful baseline diagnostic, but not a robust replacement for the current stability and transfer path.

### Stability Hypothesis Tracker

| Hypothesis | Intervention | Observed effect | Decision |
| --- | --- | --- | --- |
| Sequence-order artifacts in cached tensors inflate unstable behavior | Randomized sequence sampling when building run tensors; cache audits | Removed deterministic head-selection behavior; instability still present | Keep random sampling as default |
| Study imbalance drives shortcut learning | Study-balanced run sampler | Frequently improved cancer type holdout transfer compared with random sampling | Keep for cancer type runs |
| Class imbalance contributes to calibration collapse | Class-weighted loss (balanced and tempered balanced_sqrt) | Improved holdout in some runs, but strong seed dependence remained | Keep as candidate lever, not a locked default |
| AMP is a safe drop-in for class-weighted runs | bfloat16 AMP at memory-safe batch size | Mixed transfer performance; required dtype bug fix for class-weight path | Secondary option, not default |
| Optimization variance drives instability | Lower learning rate, warmup-cosine schedule, and gradient clipping | Reduced instability in some trajectories but did not yield a consistent cross-seed winner | Keep as secondary stabilizers; not sufficient alone |
| Freeze and unfreeze backbone improves transfer stability | Freeze_backbone_epochs with reduced backbone learning-rate multiplier | Mixed holdout outcomes with large post-unfreeze gradient spikes in some runs | Not selected as a default path |
| Pretrained HyenaDNA representations plus classical classifiers can bypass fine-tuning instability | Extract frozen run-level embeddings from pretrained `hyenadna-small-32k-seqlen` (no fine-tuning) and fit SVM/random forest | Strong development signals did not translate into reliable holdout transfer | Keep as a baseline check; not selected as primary path |
| Seed sensitivity dominates model selection risk | Repeated seed sweeps under near-identical configurations | Large holdout spread persisted despite similar validation and test AUC | Require multi-seed evidence before declaring improvements |

