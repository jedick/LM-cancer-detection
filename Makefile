# Run from the repository root. Paths and defaults load from configs/pipeline.yaml
# via scripts/pipeline_emit_mk.py (requires PyYAML).

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
PIPELINE_CFG := $(ROOT)/configs/pipeline.yaml
PIPE_EMIT := $(ROOT)/scripts/pipeline_emit_mk.py

# $(shell ...) collapses newlines; load each path with --get (requires PyYAML).
ifeq ($(shell $(PYTHON) -c "import yaml" >/dev/null 2>&1 && echo ok),ok)
PIPE_DATA_DIR := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.data_dir)
PIPE_FASTA_DIR := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.fasta_dir)
PIPE_OUTPUTS_DIR := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.outputs_dir)
PIPE_RESULTS_DIR := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.results_dir)
PIPE_RESULTS_SCRATCH_DIR := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.results_scratch_dir)
PIPE_TETRANUCLEOTIDE_FREQUENCIES_CSV := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.tetranucleotide_frequencies_csv)
PIPE_UC_CAP_ROOT := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.uc_cap_root)
PIPE_SEQUENCE_CACHE_PARQUET := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get paths.sequence_cache_parquet)
PIPE_DEFAULT_TASK := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get tetranucleotide.default_task)
PIPE_DEFAULT_UC_CAP_CLASSIFIER := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get uc_cap_classifiers.0)
PIPE_SEQUENCE_CACHE_INPUT_SUFFIX := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get sequence_cache.input_suffix)
PIPE_SEQUENCE_CACHE_N_MAX := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get sequence_cache.n_max_per_run)
PIPE_SEQUENCE_CACHE_COMPRESSION := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get sequence_cache.parquet_compression)
PIPE_UCAP_RANDOM_STATE := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.random_state)
PIPE_UCAP_PCA_VARIANCE := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.pca_variance)
PIPE_UCAP_PCA_COMPONENTS := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.pca_components)
PIPE_UCAP_SEQ_NORMALIZE := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.seq_normalize)
PIPE_UCAP_SEQ_LOG1P := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.seq_log1p)
PIPE_UCAP_CAP_TRANSFORM := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.cap_transform)
PIPE_UCAP_CLR_PSEUDOCOUNT := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.clr_pseudocount)
PIPE_UCAP_BATCH_SIZE := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.batch_size)
PIPE_UCAP_MAX_ITER := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.max_iter)
PIPE_UCAP_CHUNK_SIZE := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --get run_uc_cap_defaults.chunk_size)
else
$(error PyYAML is required for the Makefile (pip install pyyaml))
endif

# Optional overrides for build_uc_cap_sequence_cache.py (defaults come from YAML).
SEQ_INPUT_SUFFIX ?= $(PIPE_SEQUENCE_CACHE_INPUT_SUFFIX)
SEQ_N_MAX ?= $(PIPE_SEQUENCE_CACHE_N_MAX)
SEQ_COMPRESSION ?= $(PIPE_SEQUENCE_CACHE_COMPRESSION)

# Task / model / UC-CAP triple (override on command line, e.g. make fit_knn TASK=cancer_type).
TASK ?= $(PIPE_DEFAULT_TASK)
MODEL ?= $(PIPE_DEFAULT_UC_CAP_CLASSIFIER)
N_UC ?= 1000
N_CLUSTERS ?= 2000
N_CAP ?= 5000

TETRA_CSV := $(ROOT)/$(PIPE_TETRANUCLEOTIDE_FREQUENCIES_CSV)
SEQ_CACHE := $(ROOT)/$(PIPE_SEQUENCE_CACHE_PARQUET)
CAP_CSV_REL := $(shell $(PYTHON) $(PIPE_EMIT) $(PIPELINE_CFG) --render-cap-csv --n-uc $(N_UC) --n-clusters $(N_CLUSTERS) --n-cap $(N_CAP))
CAP_CSV := $(ROOT)/$(CAP_CSV_REL)

UCAP_PCA_COMPONENTS_ARG := $(if $(strip $(PIPE_UCAP_PCA_COMPONENTS)),--pca-components $(PIPE_UCAP_PCA_COMPONENTS),)
UCAP_NO_SEQ_NORMALIZE_ARG := $(if $(filter false 0 no,$(PIPE_UCAP_SEQ_NORMALIZE)),--no-seq-normalize,)
UCAP_SEQ_LOG1P_ARG := $(if $(filter true 1 yes,$(PIPE_UCAP_SEQ_LOG1P)),--seq-log1p,)

# Study metadata CSVs (typically small); used to rebuild tetramer frequencies when data change.
DATA_CSVS := $(shell find $(ROOT)/$(PIPE_DATA_DIR) -type f -name '*.csv' 2>/dev/null)

.DEFAULT_GOAL := help

.PHONY: help download_data tetramer_frequencies sequence_cache fit_knn fit_uc_cap grid_uc_cap

help:
	@echo "LM-cancer-detection Makefile (paths from configs/pipeline.yaml)"
	@echo ""
	@echo "  make download_data"
	@echo "      Run scripts/download_sra_data.py (reads $(PIPE_DATA_DIR)/**/*.csv,"
	@echo "      writes $(PIPE_FASTA_DIR)/<study>/<Run>.fasta.gz)."
	@echo ""
	@echo "  make tetramer_frequencies"
	@echo "      Build $(PIPE_TETRANUCLEOTIDE_FREQUENCIES_CSV) and per-run sequence"
	@echo "      count files under $(PIPE_OUTPUTS_DIR)/ (depends on data CSVs)."
	@echo "      FASTA inputs under $(PIPE_FASTA_DIR)/ are not listed as prerequisites"
	@echo "      (avoids huge dep lists); delete the CSV output to force a rebuild."
	@echo ""
	@echo "  make sequence_cache  [SEQ_INPUT_SUFFIX=$(SEQ_INPUT_SUFFIX)]"
	@echo "      Build $(PIPE_SEQUENCE_CACHE_PARQUET) via build_uc_cap_sequence_cache.py."
	@echo "      Defaults from YAML: input_suffix=$(PIPE_SEQUENCE_CACHE_INPUT_SUFFIX)"
	@echo "      n_max=$(PIPE_SEQUENCE_CACHE_N_MAX) compression=$(PIPE_SEQUENCE_CACHE_COMPRESSION)"
	@echo ""
	@echo "  make fit_knn [TASK=$(PIPE_DEFAULT_TASK)]"
	@echo "      Run fit_tetranucleotide_classifier.py; JSON -> $(PIPE_RESULTS_SCRATCH_DIR)/"
	@echo "      (use --results-json with no path, see script help)."
	@echo ""
	@echo "  make fit_uc_cap [TASK=...] [MODEL=...] [N_UC=...] [N_CLUSTERS=...] [N_CAP=...]"
	@echo "      Build CAP CSV if missing, then fit_uc_cap_classifier.py; JSON -> scratch."
	@echo "      Defaults: TASK=$(PIPE_DEFAULT_TASK) MODEL=$(PIPE_DEFAULT_UC_CAP_CLASSIFIER)"
	@echo "      N_UC=1000 N_CLUSTERS=2000 N_CAP=5000"
	@echo ""
	@echo "  make grid_uc_cap"
	@echo "      Run scripts/grid_uc_cap_pipeline.py (grid from pipeline.yaml)."
	@echo ""

$(TETRA_CSV): $(DATA_CSVS) $(ROOT)/scripts/calculate_tetranucleotide_frequencies.py \
		$(ROOT)/scripts/shared_splits.py
	@mkdir -p "$(dir $(TETRA_CSV))"
	cd "$(ROOT)" && $(PYTHON) scripts/calculate_tetranucleotide_frequencies.py

tetramer_frequencies: $(TETRA_CSV)
	@echo "Up to date: $(TETRA_CSV)"

$(SEQ_CACHE): $(TETRA_CSV) $(ROOT)/scripts/build_uc_cap_sequence_cache.py
	cd "$(ROOT)" && $(PYTHON) scripts/build_uc_cap_sequence_cache.py \
		--input-suffix "$(SEQ_INPUT_SUFFIX)" \
		--n-max "$(SEQ_N_MAX)" \
		--compression "$(SEQ_COMPRESSION)"

sequence_cache: $(SEQ_CACHE)
	@echo "Up to date: $(SEQ_CACHE)"

download_data:
	cd "$(ROOT)" && $(PYTHON) scripts/download_sra_data.py

fit_knn: $(TETRA_CSV)
	cd "$(ROOT)" && $(PYTHON) scripts/fit_tetranucleotide_classifier.py \
		--task "$(TASK)" \
		--csv "$(PIPE_TETRANUCLEOTIDE_FREQUENCIES_CSV)" \
		--results-json

fit_uc_cap: $(SEQ_CACHE) $(TETRA_CSV)
	@if ! test -f "$(CAP_CSV)"; then \
		echo "Building missing $(CAP_CSV)"; \
		cd "$(ROOT)" && $(PYTHON) scripts/run_uc_cap_pipeline.py \
			--n-uc $(N_UC) --n-clusters $(N_CLUSTERS) --n-cap $(N_CAP) \
			--random-state "$(PIPE_UCAP_RANDOM_STATE)" \
			--pca-variance "$(PIPE_UCAP_PCA_VARIANCE)" \
			$(UCAP_PCA_COMPONENTS_ARG) \
			$(UCAP_NO_SEQ_NORMALIZE_ARG) \
			$(UCAP_SEQ_LOG1P_ARG) \
			--cap-transform "$(PIPE_UCAP_CAP_TRANSFORM)" \
			--clr-pseudocount "$(PIPE_UCAP_CLR_PSEUDOCOUNT)" \
			--batch-size "$(PIPE_UCAP_BATCH_SIZE)" \
			--max-iter "$(PIPE_UCAP_MAX_ITER)" \
			--chunk-size "$(PIPE_UCAP_CHUNK_SIZE)"; \
	fi
	cd "$(ROOT)" && $(PYTHON) scripts/fit_uc_cap_classifier.py \
		--csv "$(CAP_CSV_REL)" \
		--task "$(TASK)" \
		--classifier "$(MODEL)" \
		--run-metadata-csv "$(PIPE_TETRANUCLEOTIDE_FREQUENCIES_CSV)" \
		--results-json

grid_uc_cap: $(SEQ_CACHE) $(TETRA_CSV)
	cd "$(ROOT)" && $(PYTHON) scripts/grid_uc_cap_pipeline.py
