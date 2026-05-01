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

# Expand fit_tetramer experiment output paths from experiments.yaml template+names.
TETRAMER_EXPERIMENT_OUTPUTS := $(shell $(PYTHON) -c "import yaml,pathlib; r=pathlib.Path('$(ROOT)'); cfg=yaml.safe_load((r/'experiments.yaml').read_text(encoding='utf-8')); sec=cfg.get('fit_tetramer_classifier',{}); tpl=sec.get('results_json_template','results/tetramer/{name}.json'); exps=sec.get('experiments',[]); print(' '.join(str((r/tpl.format(name=e['name'])).resolve()) for e in exps))")
# Enumerate fit_tetramer experiments so rules can run by --expt index.
TETRAMER_EXPERIMENT_INDICES := $(shell seq 1 $(words $(TETRAMER_EXPERIMENT_OUTPUTS)))
# Select a single fit_tetramer output path by 1-based EXPT index.
TETRAMER_EXPERIMENT_OUTPUT := $(if $(filter-out 0,$(strip $(EXPT))),$(word $(EXPT),$(TETRAMER_EXPERIMENT_OUTPUTS)),)

TETRA_CSV := $(ROOT)/$(TETRAMER_FREQUENCIES_CSV)
SEQ_CACHE := $(ROOT)/$(UC_CAP_ROOT)/sequence_counts_first_$(SEQUENCE_CACHE_N_MAX)_all_runs.parquet

# Study metadata CSVs (typically small); used to rebuild tetramer frequencies when data change.
DATA_CSVS := $(shell find $(ROOT)/$(DATA_DIR) -type f -name '*.csv' 2>/dev/null)

.DEFAULT_GOAL := help

.PHONY: help download_data tetramer_frequencies sequence_cache fit_tetramer fit_uc_cap grid_uc_cap explain explain-%

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
	@echo "      Run fit_tetramer_classifier.py with defaults.yaml."
	@echo "      Optional: EXPT=<n> passes --expt <n>."
	@echo "      Optional: EXPT=0 builds all configured experiments incrementally."
	@echo ""
	@echo "  make fit_uc_cap"
	@echo "      Run fit_uc_cap_classifier.py with script defaults;"
	@echo "      use script CLI directly for non-default settings."
	@echo ""
	@echo "  make grid_uc_cap"
	@echo "      Run scripts/grid_uc_cap_pipeline.py (grid from defaults.yaml)."
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
fit_tetramer: $(TETRA_CSV)
	cd "$(ROOT)" && $(PYTHON) scripts/fit_tetramer_classifier.py $(EXPT_ARG)
endif

define tetramer_experiment_rule
$(word $(1),$(TETRAMER_EXPERIMENT_OUTPUTS)): $(TETRA_CSV) \
		$(ROOT)/scripts/fit_tetramer_classifier.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_tetramer_classifier.py --expt $(1)
endef

$(foreach i,$(TETRAMER_EXPERIMENT_INDICES),$(eval $(call tetramer_experiment_rule,$(i))))

fit_uc_cap: $(SEQ_CACHE) $(TETRA_CSV)
	cd "$(ROOT)" && $(PYTHON) scripts/run_uc_cap_pipeline.py
	cd "$(ROOT)" && $(PYTHON) scripts/fit_uc_cap_classifier.py --results-json

grid_uc_cap: $(SEQ_CACHE) $(TETRA_CSV)
	cd "$(ROOT)" && $(PYTHON) scripts/grid_uc_cap_pipeline.py

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
