# Run from the repository root. Paths/defaults come from defaults.yaml.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
CONFIG := $(ROOT)/defaults.yaml

# Read a top-level YAML section key from defaults.yaml via awk.
define yaml_section_value
$(strip $(shell awk -F': *' 'BEGIN{inside=0} $$1=="$(1)"{inside=1; next} inside && $$0 !~ /^  /{inside=0} inside && $$1=="  $(2)"{print $$2; exit}' "$(CONFIG)"))
endef

DATA_DIR := $(call yaml_section_value,paths,data_dir)
DATASETS_CSV := $(ROOT)/$(call yaml_section_value,paths,datasets_csv)
TETRAMER_FREQUENCIES_CSV := $(call yaml_section_value,paths,tetramer_frequencies_csv)
UC_CAP_ROOT := $(call yaml_section_value,paths,uc_cap_root)
EMBEDDINGS_DIR := $(call yaml_section_value,paths,embeddings_dir)
RUN_TENSORS_DIR := $(ROOT)/$(call yaml_section_value,paths,run_tensors_dir)
SEQUENCE_CACHE_N_MAX := $(call yaml_section_value,sequence_cache,n_max_per_run)
EXPT ?=
EXPT_ARG := $(if $(strip $(EXPT)),--expt $(EXPT),)
FEAT ?=
FEAT_ARG := $(if $(strip $(FEAT)),--feat $(FEAT),)

# Expand fit_tetramer experiment output paths from experiments.yaml template+names.
TETRAMER_EXPERIMENT_OUTPUTS := $(shell $(PYTHON) -c "import yaml,pathlib; r=pathlib.Path('$(ROOT)'); cfg=yaml.safe_load((r/'experiments.yaml').read_text(encoding='utf-8')); sec=cfg.get('fit_classifier',{}); tpl=sec.get('results_json_template','results/{features}/{name}.json'); exps=sec.get('experiments',[]); print(' '.join(str((r/tpl.format(name=e['name'], features='tetramer')).resolve()) for e in exps))")
# Experiment name stems (same order as --expt 1..N); used for results/uc_cap/<feat>/<name>.json.
EXPERIMENT_NAMES := $(shell $(PYTHON) -c "import yaml,pathlib; r=pathlib.Path('$(ROOT)'); cfg=yaml.safe_load((r/'experiments.yaml').read_text(encoding='utf-8')); exps=cfg.get('fit_classifier',{}).get('experiments',[]); print(' '.join(str(e['name']) for e in exps))")
# Enumerate fit_tetramer experiments so rules can run by --expt index.
TETRAMER_EXPERIMENT_INDICES := $(shell seq 1 $(words $(TETRAMER_EXPERIMENT_OUTPUTS)))
# Select a single fit_tetramer output path by 1-based EXPT index.
TETRAMER_EXPERIMENT_OUTPUT := $(if $(filter-out 0,$(strip $(EXPT))),$(word $(EXPT),$(TETRAMER_EXPERIMENT_OUTPUTS)),)

# CAP CSV outputs for each run_uc_cap_pipeline row in experiments.yaml (merged with defaults).
UC_CAP_FEATURE_OUTPUTS := $(shell $(PYTHON) $(ROOT)/helpers/list_uc_cap_feature_outputs.py "$(ROOT)")
UC_CAP_FEATURE_INDICES := $(shell seq 1 $(words $(UC_CAP_FEATURE_OUTPUTS)))
UC_CAP_FEATURE_OUTPUT := $(if $(filter-out 0,$(strip $(FEAT))),$(word $(FEAT),$(UC_CAP_FEATURE_OUTPUTS)),)
# Deferred: FEAT/EXPT come from the command line when these are expanded as fit_uc_cap prerequisites.
UC_CAP_SINGLE_EXPT_ALL_FEAT_OUTPUTS = $(foreach f,$(UC_CAP_FEATURE_INDICES),$(ROOT)/results/uc_cap/$(f)/$(word $(EXPT),$(EXPERIMENT_NAMES)).json)
UC_CAP_SINGLE_FEAT_ALL_EXPT_OUTPUTS = $(foreach e,$(TETRAMER_EXPERIMENT_INDICES),$(ROOT)/results/uc_cap/$(FEAT)/$(word $(e),$(EXPERIMENT_NAMES)).json)
# Deferred: all (feature set x experiment) JSON paths under results/uc_cap/<feat>/ (same rules as partial sweeps).
UC_CAP_FULL_GRID_OUTPUTS = $(foreach f,$(UC_CAP_FEATURE_INDICES),$(foreach e,$(TETRAMER_EXPERIMENT_INDICES),$(ROOT)/results/uc_cap/$(f)/$(word $(e),$(EXPERIMENT_NAMES)).json))
# Baseline CAP CSV (defaults.yaml merge only); real file target for default ``make run_uc_cap``.
UC_CAP_BASELINE_CAP_CSV := $(shell $(PYTHON) $(ROOT)/helpers/list_uc_cap_feature_outputs.py "$(ROOT)" --baseline)
EMBEDDING_FEATURE_OUTPUTS := $(shell $(PYTHON) $(ROOT)/helpers/list_embedding_feature_outputs.py "$(ROOT)")
EMBEDDING_FEATURE_INDICES := $(shell seq 1 $(words $(EMBEDDING_FEATURE_OUTPUTS)))
EMBEDDING_FEATURE_OUTPUT := $(if $(filter-out 0,$(strip $(FEAT))),$(word $(FEAT),$(EMBEDDING_FEATURE_OUTPUTS)),)
# Deferred: FEAT=0 EXPT=<n> fan-out output paths under results/embeddings/<feat>/.
EMBEDDINGS_SINGLE_EXPT_ALL_FEAT_OUTPUTS = $(foreach f,$(EMBEDDING_FEATURE_INDICES),$(ROOT)/results/embeddings/$(f)/$(word $(EXPT),$(EXPERIMENT_NAMES)).json)

TETRA_CSV := $(ROOT)/$(TETRAMER_FREQUENCIES_CSV)
SEQ_CACHE := $(ROOT)/$(UC_CAP_ROOT)/sequence_counts_first_$(SEQUENCE_CACHE_N_MAX)_all_runs.parquet

# Study metadata CSVs (typically small); used to rebuild tetramer frequencies when data change.
DATA_CSVS := $(shell find $(ROOT)/$(DATA_DIR) -type f -name '*.csv' 2>/dev/null)

.DEFAULT_GOAL := help

.PHONY: help download_data tetramer_counts tetramer_frequencies sequence_cache fit_tetramer fit_uc_cap run_uc_cap \
	train_hyenadna audit_run_tensors extract_embeddings extract_embeddings_baseline \
	fit_embeddings explain explain-%

help:
	@echo "LM-cancer-detection Makefile (script defaults from defaults.yaml)"
	@echo ""
	@echo "  make download_data"
	@echo "      Run scripts/download_sra_data.py."
	@echo ""
	@echo "  make tetramer_counts"
	@echo "      Count 4-mers per FASTA sequence; write missing"
	@echo "      outputs/tetramer_counts/<cancer>/<study>/<Run>.csv.xz (compute-heavy)."
	@echo "      FASTA inputs are not prerequisites (avoids huge dep lists)."
	@echo ""
	@echo "  make tetramer_frequencies"
	@echo "      Append missing Run rows to tetramer frequencies CSV from count files"
	@echo "      (depends on data CSVs). Run tetramer_counts first so count files exist."
	@echo "      Delete the frequencies CSV to force a full rebuild of that artifact."
	@echo ""
	@echo "  make sequence_cache"
	@echo "      Build $(SEQ_CACHE) via build_uc_cap_sequence_cache.py."
	@echo ""
	@echo "  make fit_tetramer"
	@echo "      Run scripts/fit_classifier.py --tetramer with defaults.yaml; default run passes"
	@echo "      --results-json (metrics under results/scratch/). Optional: EXPT=<n> for experiments."
	@echo "      Optional: EXPT=0 builds all configured experiments incrementally."
	@echo ""
	@echo "  make fit_uc_cap"
	@echo "      fit_classifier.py --uc_cap; default uses baseline CAP and --results-json (scratch)."
	@echo "      FEAT=<n> / EXPT=<n> mirror experiments.yaml (same indices as run_uc_cap / fit_tetramer)."
	@echo "      FEAT=0 EXPT=<n>: one experiment, all feature sets → results/uc_cap/1..N/<name>.json."
	@echo "      FEAT=<n> EXPT=0: one feature set, all experiments → results/uc_cap/<n>/<name>.json"
	@echo "      (same filenames as make fit_tetramer EXPT=0). FEAT=0 EXPT=0: full grid (every feat dir"
	@echo "      holds every experiment JSON). Disallowed: FEAT=0 or EXPT=0 alone without the other axis."
	@echo ""
	@echo "  make run_uc_cap"
	@echo "      Build the baseline CAP CSV (defaults.yaml) if needed; incremental vs inputs."
	@echo "      Optional: FEAT=<n> builds that feature-set CAP (1-based experiments.yaml index)."
	@echo "      Optional: FEAT=0 builds all configured UC/CAP pipeline feature sets incrementally."
	@echo ""
	@echo "  make run_tensors"
	@echo "      Run scripts/build_run_tensors.py once from defaults.yaml run_tensors settings."
	@echo ""
	@echo "  make extract_embeddings"
	@echo "      Build consolidated frozen HyenaDNA embedding CSVs under $(ROOT)/$(EMBEDDINGS_DIR)."
	@echo "      Optional: FEAT=<n> uses experiments.yaml extract_embeddings row n."
	@echo "      Optional: FEAT=0 builds all configured embedding feature sets incrementally."
	@echo ""
	@echo "  make fit_embeddings"
	@echo "      Run scripts/fit_classifier.py --embeddings."
	@echo "      Omit FEAT for baseline defaults.yaml extract_embeddings; FEAT=<n> uses experiments row."
	@echo "      Optional: EXPT=<n> picks fit_classifier experiment row."
	@echo "      FEAT=0 EXPT=<n> runs one experiment across all embedding feature sets"
	@echo "      and writes results/embeddings/<k>/<name>.json."
	@echo ""
	@echo "  make audit_run_tensors"
	@echo "      Summarize outputs/run_tensors/*.pt coverage/utilization under cache_audit/run_tensors/."
	@echo ""
	@echo "  make train_hyenadna"
	@echo "      Run scripts/train_hyenadna.py against outputs/run_tensors/*.pt."
	@echo "      Optional: EXPT=<n> runs one train_hyenadna experiment from experiments.yaml."
	@echo ""
	@echo "  make explain TARGET=<make_target>"
	@echo "      Compact dependency/mtime explanation using make --trace."
	@echo "      Examples: make explain TARGET=sequence_cache"
	@echo "                make explain-sequence_cache"
	@echo ""

tetramer_counts: $(DATA_CSVS) $(ROOT)/scripts/count_tetramers.py
	cd "$(ROOT)" && $(PYTHON) scripts/count_tetramers.py

$(TETRA_CSV): $(DATA_CSVS) $(ROOT)/scripts/calculate_tetramer_frequencies.py $(ROOT)/defaults.yaml
	@mkdir -p "$(dir $(TETRA_CSV))"
	cd "$(ROOT)" && $(PYTHON) scripts/calculate_tetramer_frequencies.py

tetramer_frequencies: $(TETRA_CSV)
	@echo "Up to date: $(TETRA_CSV)"

$(SEQ_CACHE): $(TETRA_CSV) $(ROOT)/scripts/build_uc_cap_sequence_cache.py $(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/build_uc_cap_sequence_cache.py

sequence_cache: $(SEQ_CACHE)
	@echo "Up to date: $(SEQ_CACHE)"

# Real file target: default ``run_uc_cap`` depends on this path (no duplicate recipe for FEAT>=1 rows).
$(UC_CAP_BASELINE_CAP_CSV): $(SEQ_CACHE) $(TETRA_CSV) \
		$(ROOT)/scripts/run_uc_cap_pipeline.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/helpers/list_uc_cap_feature_outputs.py
	cd "$(ROOT)" && $(PYTHON) scripts/run_uc_cap_pipeline.py

download_data:
	cd "$(ROOT)" && $(PYTHON) scripts/download_sra_data.py

ifeq ($(strip $(EXPT)),0)
fit_tetramer: $(TETRAMER_EXPERIMENT_OUTPUTS)
	@echo "Up to date: all fit_tetramer experiments"
else ifneq ($(strip $(EXPT)),)
fit_tetramer: $(TETRAMER_EXPERIMENT_OUTPUT)
	@if test -z "$(TETRAMER_EXPERIMENT_OUTPUT)"; then \
		echo "Invalid EXPT=$(EXPT). Use EXPT=0 for all, or EXPT=1..N from experiments.yaml."; \
		exit 2; \
	fi
else
fit_tetramer: $(TETRA_CSV) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --tetramer --results-json $(EXPT_ARG)
endif

define tetramer_experiment_rule
$(word $(1),$(TETRAMER_EXPERIMENT_OUTPUTS)): $(TETRA_CSV) \
		$(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --tetramer --expt $(1)
endef

$(foreach i,$(TETRAMER_EXPERIMENT_INDICES),$(eval $(call tetramer_experiment_rule,$(i))))

$(RUN_TENSORS_DIR): $(DATA_CSVS) $(DATASETS_CSV) $(ROOT)/scripts/build_run_tensors.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/build_run_tensors.py

run_tensors: $(RUN_TENSORS_DIR)
	@echo "Up to date: $(RUN_TENSORS_DIR)"

audit_run_tensors: $(ROOT)/scripts/audit_run_tensors.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/audit_run_tensors.py

train_hyenadna: $(DATA_CSVS) $(DATASETS_CSV) $(ROOT)/scripts/train_hyenadna.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/train_hyenadna.py --results-json $(EXPT_ARG)

extract_embeddings_baseline: $(RUN_TENSORS_DIR) \
		$(ROOT)/scripts/extract_embeddings.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/helpers/list_embedding_feature_outputs.py
	cd "$(ROOT)" && $(PYTHON) scripts/extract_embeddings.py

ifeq ($(strip $(FEAT)),0)
extract_embeddings: $(EMBEDDING_FEATURE_OUTPUTS)
	@echo "Up to date: all extract_embeddings feature sets"
else ifneq ($(strip $(FEAT)),)
extract_embeddings: $(EMBEDDING_FEATURE_OUTPUT)
	@if test -z "$(EMBEDDING_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=0 for all, or FEAT=1..N from experiments.yaml extract_embeddings."; \
		exit 2; \
	fi
	@:
else
extract_embeddings: extract_embeddings_baseline
	@:
endif

define embedding_feature_rule
$(word $(1),$(EMBEDDING_FEATURE_OUTPUTS)): $(RUN_TENSORS_DIR) \
		$(ROOT)/scripts/extract_embeddings.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml \
		$(ROOT)/helpers/list_embedding_feature_outputs.py
	cd "$(ROOT)" && $(PYTHON) scripts/extract_embeddings.py --feat $(1)
endef
$(foreach i,$(EMBEDDING_FEATURE_INDICES),$(eval $(call embedding_feature_rule,$(i))))

ifneq ($(filter fit_embeddings,$(MAKECMDGOALS)),)
ifeq ($(strip $(FEAT)),0)
ifeq ($(strip $(EXPT)),)
$(error fit_embeddings: FEAT=0 requires EXPT=1..N (one experiment, all embedding feature sets).)
endif
ifeq ($(strip $(EXPT)),0)
$(error fit_embeddings: FEAT=0 EXPT=0 is unsupported; use EXPT=1..N.)
endif
endif
endif

# results/embeddings/<feat_index>/<experiment_name>.json for FEAT=0 EXPT=<n> sweeps.
define embeddings_subdir_json_rule
$(ROOT)/results/embeddings/$(1)/$(word $(2),$(EXPERIMENT_NAMES)).json: $(word $(1),$(EMBEDDING_FEATURE_OUTPUTS)) \
		$(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --embeddings --feat $(1) --expt $(2) --results-json $(ROOT)/results/embeddings/$(1)/$(word $(2),$(EXPERIMENT_NAMES)).json
endef
$(foreach f,$(EMBEDDING_FEATURE_INDICES),$(foreach e,$(TETRAMER_EXPERIMENT_INDICES),$(eval $(call embeddings_subdir_json_rule,$(f),$(e)))))

ifeq ($(strip $(FEAT)),0)
ifneq ($(strip $(EXPT)),)
fit_embeddings: $(EMBEDDINGS_SINGLE_EXPT_ALL_FEAT_OUTPUTS)
	@echo "Up to date: fit_embeddings EXPT=$(EXPT) (all feature sets under results/embeddings/<k>/)"
endif
else
ifneq ($(strip $(FEAT)),)
fit_embeddings: $(EMBEDDING_FEATURE_OUTPUT) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@if test -z "$(EMBEDDING_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=1..N from experiments.yaml extract_embeddings, or omit FEAT for baseline."; \
		exit 2; \
	fi
	@if test -n "$(strip $(EXPT))"; then \
		cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --embeddings --feat $(FEAT) $(EXPT_ARG); \
	else \
		cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --embeddings --feat $(FEAT) --results-json; \
	fi
else
fit_embeddings: extract_embeddings_baseline $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --embeddings $(EXPT_ARG)
endif
endif

ifeq ($(strip $(FEAT)),0)
run_uc_cap: $(UC_CAP_FEATURE_OUTPUTS)
	@echo "Up to date: all run_uc_cap pipeline feature sets"
else ifneq ($(strip $(FEAT)),)
run_uc_cap: $(UC_CAP_FEATURE_OUTPUT)
	@if test -z "$(UC_CAP_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=0 for all, or FEAT=1..N from experiments.yaml."; \
		exit 2; \
	fi
	@:
else
run_uc_cap: $(UC_CAP_BASELINE_CAP_CSV)
	@:
endif

define uc_cap_feature_rule
$(word $(1),$(UC_CAP_FEATURE_OUTPUTS)): $(SEQ_CACHE) $(TETRA_CSV) \
		$(ROOT)/scripts/run_uc_cap_pipeline.py \
		$(ROOT)/helpers/list_uc_cap_feature_outputs.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/run_uc_cap_pipeline.py --feat $(1)
endef

$(foreach i,$(UC_CAP_FEATURE_INDICES),$(eval $(call uc_cap_feature_rule,$(i))))

ifneq ($(filter fit_uc_cap,$(MAKECMDGOALS)),)
ifeq ($(strip $(FEAT)),0)
ifeq ($(strip $(EXPT)),)
$(error fit_uc_cap: FEAT=0 requires EXPT=0 (full grid) or EXPT=1..N (one experiment, all feature sets).)
endif
endif
ifeq ($(strip $(EXPT)),0)
ifeq ($(strip $(FEAT)),)
$(error fit_uc_cap: EXPT=0 requires FEAT=1..N, or FEAT=0 EXPT=0 for the full grid.)
endif
endif
endif

# results/uc_cap/<feat_index>/<experiment_name>.json (all sweep modes including FEAT=0 EXPT=0).
define uc_cap_subdir_json_rule
$(ROOT)/results/uc_cap/$(1)/$(word $(2),$(EXPERIMENT_NAMES)).json: $(word $(1),$(UC_CAP_FEATURE_OUTPUTS)) \
		$(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --feat $(1) --expt $(2) --results-json $(ROOT)/results/uc_cap/$(1)/$(word $(2),$(EXPERIMENT_NAMES)).json
endef
$(foreach f,$(UC_CAP_FEATURE_INDICES),$(foreach e,$(TETRAMER_EXPERIMENT_INDICES),$(eval $(call uc_cap_subdir_json_rule,$(f),$(e)))))

ifeq ($(strip $(FEAT)),0)
ifeq ($(strip $(EXPT)),0)
fit_uc_cap: $(UC_CAP_FULL_GRID_OUTPUTS) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@echo "Up to date: fit_uc_cap full grid (results/uc_cap/<k>/ for each experiment JSON)."
else ifneq ($(strip $(EXPT)),)
fit_uc_cap: $(UC_CAP_SINGLE_EXPT_ALL_FEAT_OUTPUTS) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@echo "Up to date: fit_uc_cap EXPT=$(EXPT) (all feature sets: results/uc_cap/<k>/$(word $(EXPT),$(EXPERIMENT_NAMES)).json)"
else
# FEAT=0 with EXPT unset: invalid for fit_uc_cap (see parse-time check when fit_uc_cap is a goal).
# Dummy rule avoids expanding $(word $(EXPT),...) during parsing for other goals (e.g. run_uc_cap FEAT=0).
fit_uc_cap: $(ROOT)/scripts/fit_classifier.py
	@:
endif
else ifeq ($(strip $(EXPT)),0)
ifneq ($(strip $(FEAT)),)
fit_uc_cap: $(UC_CAP_SINGLE_FEAT_ALL_EXPT_OUTPUTS)
	@echo "Up to date: fit_uc_cap FEAT=$(FEAT) (all experiments under results/uc_cap/$(FEAT)/)"
endif
else ifneq ($(strip $(FEAT)),)
fit_uc_cap: $(UC_CAP_FEATURE_OUTPUT) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@if test -z "$(UC_CAP_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=1..N from experiments.yaml, FEAT=0 EXPT=0 (full grid), or FEAT=0 EXPT=1..N."; \
		exit 2; \
	fi
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --feat $(FEAT) $(EXPT_ARG)
else
fit_uc_cap: $(UC_CAP_BASELINE_CAP_CSV) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap $(EXPT_ARG) --results-json
endif

TARGET ?=

explain:
	@if test -z "$(TARGET)"; then \
		echo "Usage: make explain TARGET=<make_target>"; \
		echo "   or: make explain-<make_target>"; \
		exit 2; \
	fi
	cd "$(ROOT)" && $(PYTHON) scripts/explain_make_trace.py "$(TARGET)"

explain-%:
	cd "$(ROOT)" && $(PYTHON) scripts/explain_make_trace.py "$*"
