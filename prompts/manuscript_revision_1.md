# Revising machine learning genomics manuscript with domain-adversarial training

Date: 20260514

## JMD

Please help me revise and polish a manuscript for a machine learning genomics project. Use the manuscript.md as a baseline. Figures and table are also attached for your reference. I'd like to improve the introduction. Some ideas:

- Start with background: why cancer prediction is important
- 16S sequences are traditionally used for taxonomic features, but that's not the only way to approach the problem
- When making predictions from genomic data, test splits from the same studies give overoptimistic results (perhaps similar to spatial autocorrelation and maybe even considered a type of data leakage)
- Our study is therefore based on a new data compilation with holdout datasets, allowing to generate reliable indicators of real-world model performance. Notably, this compilation represents older and more recent studies and is representative of different laboratories, clinical protocols, and regions. After random sampling from studies with many samples (to help balance study representation) we obtain 2051 16S rRNA sequencing runs samples with three labels (breast cancer, colorectal cancer, healthy).

We benchmark classical and deep learning techniques and against this new compilation. For classical ML we start with run-level tetramer frequencies, then investigate cluster abundance profiles to capture compositional structure. [This may be compared to using OTUs for clustering, although we don't use taxonomy here]. Our deep learning model is HyenaDNA with local modifications for domain adversarial training. This DANN method is shown to be effective in improving model performance, especially for the cancer diagnosis task.

Other notes:

- I think our major contributions are the benchmark (data curation) and domain-adversarial training. Perhaps this would be a good title: "Adversarial training improves HyenaDNA predictions in a multi-task cancer classification benchmark".
- Feel free to bring new ideas to the Introduction and Discussion. 
- For other parts of the manuscript please don't add or change the scientific content. However, you may restructure the manuscript to improve the flow.
- Correct small errors as you find them. If there are major factual inconsistencies, let me know.
- I'd like to have a few references in the Introduction (and one or two in the Discussion if appropriate). A possible reference is Whalen S, Schreiber J, Noble WS, Pollard KS (2022) Navigating the pitfalls of applying machine learning in genomics. Nat Rev Genet 23(3):169–181.
- Prefer active voice and computer science framing (often present-tense "we run ablations", but can use past tense where appropriate e.g. review of past work).
- We plan to remove the experiments with frozen pretrained embeddings with classical heads (this was mentioned but no results are shown).
- The prose should be updated to reflect the current state of Table 5, which was recently remade

Note that the experiments and manuscript are not currently finished. We plan to run HyenaDNA with increasing max_length for sets. The max_length is currently limited to 2k to run the ablations in a reasonable time - we'll sweep this from 1k to 16k to explore the effect of using more of the available sequences from each run. Even the 16k max_length * 5 sets would only use 400 sequences from each Run (assuming 200 nt long 16S fragments). This is a small number compared to the numbers of sequences used for tetramer and UC/CAP features. Because of current GPU memory (16GB) we are limited to relatively small HyenaDNA models, but we hope to show that we have at least configured a model that shows increasing performance as training sequence length increases. This was the problem with the instability of "vanilla" HyenaDNA (there was no stable response to increasing max_length - Fig. 2) that we hope will be improved with DANN.

Feel free to propose other improvements to data representation or model architecture/hyperparameters. One thing I'd like to do is have a single multi-label HyenaDNA model that could be used to make predictions for both tasks (e.g. using only softmax probs for the task-specific labels). Most importantly, keep the manuscript prose tight, accessible to a broad audience, and use plain English for most technical concepts. Once you're finished, show me three options for an abstract with different "flavors" (e.g. more data-focused, model-focused, or application-focused, but you can choose other categories than these).

Attached: appendix.md  figure1_uc_cap.svg  figure2_hyenadna.svg  manuscript.md  table2_tetramer.html  table4_uc_cap.html  table5_hyenadna.html

## Claude Sonnet 4.6 Adaptive

Now I have a full picture of the manuscript and all supporting materials. Let me write the polished revision.Good — I have a complete picture of the manuscript, all tables, and supporting materials. Now I'll write the polished revision.Here's a summary of what changed and why, organized by section.

**Title.** Updated to your suggested title, which emphasizes the two main contributions clearly.

**Introduction — fully rewritten.** The new structure follows the arc you described:
- Opens with the gut microbiome–cancer link as clinical motivation
- Explains why 16S data is typically processed into taxonomy tables and what raw-sequence approaches can add
- Introduces the data-leakage / study-autocorrelation problem in plain language, with a citation to Whalen et al. 2022 [1] as requested
- Describes the benchmark (2,051 runs, 26 studies, temporal split) concisely
- Summarizes the progression: tetramers → UC/CAP → HyenaDNA + DANN
- Closes with a crisp two-sentence statement of main contributions

**Methods.** Preserved all scientific content. Fixed: "compilaion" → "compilation"; "This suggest" → "This suggests"; tightened sentence structure throughout. Removed the mention of frozen-embedding experiments from the DANN section, as requested.

**Results.** Added a short framing paragraph before the subsections (defining the two tasks and explaining the expected test/holdout asymmetry — this was buried in the old Introduction). The DANN results subsection is fully rewritten to match the current Table 5 numbers: the old prose cited values (0.658, 0.625, 0.641) that don't appear in the updated table. The new prose accurately reports the ablation grid, including the DANN benefit for cancer diagnosis (+0.034, with variance reduction) and the more modest cancer type gain.

**Discussion.** Restructured for flow: test/holdout gap → task asymmetry → UC/CAP vs. tetramers → HyenaDNA instability → DANN interpretation → future work (context-length sweep, multi-task model, 16S-specific pre-training). Added the planned max_length sweep as a concrete next step, and the multi-task model idea (single backbone, task-specific softmax heads) as a suggestion. One discussion-appropriate Whalen et al. 2022 call is included.

**References.** Added HyenaDNA [2] and Whalen et al. [1]. Corrected the author list for Ganin et al. (original had "Ustinova, H." — should be "E.").

**Three abstracts** are at the end of the file, each foregrounding a different angle: benchmark curation (A), model/methods (B), and clinical application (C).
