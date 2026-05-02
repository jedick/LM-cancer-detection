# Run from the repository root. Paths/defaults come from defaults.yaml.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
CONFIG := $(ROOT)/defaults.yaml

# Read a top-level YAML section key from defaults.yaml via awk.
define yaml_section_value
$(strip $(shell awk -F': *' 'BEGIN{inside=0} $$1=="$(1)"{inside=1; next} inside && $$0 !~ /^  /{inside=0} inside && $$1=="  $(2)"{print $$2; exit}' "$(CONFIG)"))
endef

DATA_DIR := $(call yaml_section_value,paths,data_dir)
TETRAMER_FREQUENCIES_CSV := $(call yaml_section_value,paths,tetramer_frequencies_csv)
UC_CAP_ROOT := $(call yaml_section_value,paths,uc_cap_root)
SEQUENCE_CACHE_N_MAX := $(call yaml_section_value,sequence_cache,n_max_per_run)
EXPT ?=
EXPT_ARG := $(if $(strip $(EXPT)),--expt $(EXPT),)
FEAT ?=
FEAT_ARG := $(if $(strip $(FEAT)),--feat $(FEAT),)

# Expand fit_tetramer experiment output paths from experiments.yaml template+names.
TETRAMER_EXPERIMENT_OUTPUTS := $(shell $(PYTHON) -c "import yaml,pathlib; r=pathlib.Path('$(ROOT)'); cfg=yaml.safe_load((r/'experiments.yaml').read_text(encoding='utf-8')); sec=cfg.get('fit_classifier',{}); tpl=sec.get('results_json_template','results/{features}/{name}.json'); exps=sec.get('experiments',[]); print(' '.join(str((r/tpl.format(name=e['name'], features='tetramer')).resolve()) for e in exps))")
# Enumerate fit_tetramer experiments so rules can run by --expt index.
TETRAMER_EXPERIMENT_INDICES := $(shell seq 1 $(words $(TETRAMER_EXPERIMENT_OUTPUTS)))
# Select a single fit_tetramer output path by 1-based EXPT index.
TETRAMER_EXPERIMENT_OUTPUT := $(if $(filter-out 0,$(strip $(EXPT))),$(word $(EXPT),$(TETRAMER_EXPERIMENT_OUTPUTS)),)

# CAP CSV outputs for each run_uc_cap_pipeline row in experiments.yaml (merged with defaults).
UC_CAP_FEATURE_OUTPUTS := $(shell $(PYTHON) $(ROOT)/helpers/list_uc_cap_feature_outputs.py "$(ROOT)")
UC_CAP_FEATURE_INDICES := $(shell seq 1 $(words $(UC_CAP_FEATURE_OUTPUTS)))
UC_CAP_FEATURE_OUTPUT := $(if $(filter-out 0,$(strip $(FEAT))),$(word $(FEAT),$(UC_CAP_FEATURE_OUTPUTS)),)
# Baseline CAP CSV (defaults.yaml merge only); real file target for default ``make run_uc_cap``.
UC_CAP_BASELINE_CAP_CSV := $(shell $(PYTHON) $(ROOT)/helpers/list_uc_cap_feature_outputs.py "$(ROOT)" --baseline)

TETRA_CSV := $(ROOT)/$(TETRAMER_FREQUENCIES_CSV)
SEQ_CACHE := $(ROOT)/$(UC_CAP_ROOT)/sequence_counts_first_$(SEQUENCE_CACHE_N_MAX)_all_runs.parquet

# Study metadata CSVs (typically small); used to rebuild tetramer frequencies when data change.
DATA_CSVS := $(shell find $(ROOT)/$(DATA_DIR) -type f -name '*.csv' 2>/dev/null)

.DEFAULT_GOAL := help

.PHONY: help download_data tetramer_frequencies sequence_cache fit_tetramer fit_uc_cap run_uc_cap explain explain-%

help:
	@echo "LM-cancer-detection Makefile (script defaults from defaults.yaml)"
	@echo ""
	@echo "  make download_data"
	@echo "      Run scripts/download_sra_data.py."
	@echo ""
	@echo "  make tetramer_frequencies"
	@echo "      Append missing Run rows to tetramer frequencies CSV and"
	@echo "      create missing per-run sequence count files under outputs/."
	@echo "      (depends on data CSVs). Existing rows/files are left unchanged."
	@echo "      FASTA inputs are not listed as prerequisites"
	@echo "      (avoids huge dep lists); delete the CSV output to force a rebuild."
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
	@echo "      fit_classifier.py --uc_cap on the baseline CAP CSV; default passes --results-json"
	@echo "      (metrics under results/scratch/). Prereq: baseline CAP. Optional: FEAT=<n>."
	@echo "      Optional: FEAT=0 classifies every configured UC/CAP feature set (1..N)."
	@echo ""
	@echo "  make run_uc_cap"
	@echo "      Build the baseline CAP CSV (defaults.yaml) if needed; incremental vs inputs."
	@echo "      Optional: FEAT=<n> builds that feature-set CAP (1-based experiments.yaml index)."
	@echo "      Optional: FEAT=0 builds all configured UC/CAP pipeline feature sets incrementally."
	@echo ""
	@echo "  make explain TARGET=<make_target>"
	@echo "      Compact dependency/mtime explanation using make --trace."
	@echo "      Examples: make explain TARGET=sequence_cache"
	@echo "                make explain-sequence_cache"
	@echo ""

$(TETRA_CSV): $(DATA_CSVS) $(ROOT)/scripts/calculate_tetramer_frequencies.py \
		$(ROOT)/scripts/shared_splits.py
	@mkdir -p "$(dir $(TETRA_CSV))"
	cd "$(ROOT)" && $(PYTHON) scripts/calculate_tetramer_frequencies.py

tetramer_frequencies: $(TETRA_CSV)
	@echo "Up to date: $(TETRA_CSV)"

$(SEQ_CACHE): $(TETRA_CSV) $(ROOT)/scripts/build_uc_cap_sequence_cache.py
	cd "$(ROOT)" && $(PYTHON) scripts/build_uc_cap_sequence_cache.py --output "$(SEQ_CACHE)"

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

ifeq ($(strip $(FEAT)),0)
fit_uc_cap: $(UC_CAP_FEATURE_OUTPUTS) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@set -e; for i in $(UC_CAP_FEATURE_INDICES); do \
		cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --feat $$i --results-json; \
	done
else ifneq ($(strip $(FEAT)),)
fit_uc_cap: $(UC_CAP_FEATURE_OUTPUT) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@if test -z "$(UC_CAP_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=0 for all feature sets, or FEAT=1..N from experiments.yaml."; \
		exit 2; \
	fi
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --feat $(FEAT) --results-json
else
fit_uc_cap: $(UC_CAP_BASELINE_CAP_CSV) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --results-json
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
