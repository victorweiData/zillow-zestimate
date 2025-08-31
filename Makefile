# =========================
# Project config / exports
# =========================
export PYTHONPATH := $(shell pwd)
export MLFLOW_TRACKING_URI := file:$(PWD)/mlruns
export MLFLOW_EXPERIMENT_NAME ?= Zillow Baselines

.DEFAULT_GOAL := help

# OS detection
OS := $(shell uname)

# Paths
DATA            := data/processed/train_features.parquet
PROCESSED_DIR   := data/processed
ARTIFACTS_DIR   := $(PROCESSED_DIR)/artifacts
REPORTS_DIR     := reports

# Optuna defaults (override on CLI: make tune-lgbm N_TRIALS=120 ...)
N_TRIALS        ?= 60
TIMEOUT         ?=
SAMPLE_ROWS     ?= 80000
PRUNER          ?= hyperband
PATIENCE        ?= 25
SEED            ?= 42

LGB_MAX_ROUNDS  ?= 3000
LGB_ESR         ?= 100
XGB_MAX_ROUNDS  ?= 3000
XGB_ESR         ?= 100
CAT_ITERS       ?= 8000
CAT_ESR         ?= 100

LGB_STORAGE     := sqlite:///$(ARTIFACTS_DIR)/optuna_lgbm.db
XGB_STORAGE     := sqlite:///$(ARTIFACTS_DIR)/optuna_xgb.db
CAT_STORAGE     := sqlite:///$(ARTIFACTS_DIR)/optuna_cat.db

# OS-specific env creation command
ifeq ($(OS),Darwin)
ENV_CREATE = conda env create -f env/environment.macos.yml || true
else
ENV_CREATE = mamba env create -f env/environment.yml || conda env create -f env/environment.yml
endif

# =========================
# Phony targets
# =========================
.PHONY: help env env-pip check data prep lgbm catboost xgb trees trees-fast \
        nn stack agg-fi mlflow nb \
        tune-lgbm tune-xgb tune-cat tune-all \
        smoke-lgbm smoke-xgb smoke-cat \
        clean clean-oof clean-reports

# =========================
# Help
# =========================
help:
	@echo "Targets:"
	@echo "  env            - create conda env (platform-aware)"
	@echo "  env-pip        - create venv + pip install"
	@echo "  data           - download & preprocess (raw -> parquet -> folds -> features)"
	@echo "  prep           - (re)build folds + features only"
	@echo "  trees          - train LightGBM / CatBoost / XGBoost baselines"
	@echo "  trees-fast     - ultra-fast smoke training for all trees"
	@echo "  stack          - ridge stack of OOFs"
	@echo "  agg-fi         - combine per-model feature importances"
	@echo "  tune-<model>   - Optuna tuning (lgbm/xgb/cat)"
	@echo "  tune-all       - run all three tuners"
	@echo "  mlflow         - launch MLflow UI"
	@echo "  nb             - launch JupyterLab"
	@echo "  clean-*        - remove generated artifacts (oof/reports)"

# =========================
# Environments
# =========================
env:
	$(ENV_CREATE)
	@echo "conda activate zillow"

env-pip:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
	pip install -r env/requirements.txt

check:
	@command -v python >/dev/null || (echo "Python not found"; exit 1)

# =========================
# Data pipeline
# =========================
data: check
	./scripts/get_data.sh
	python scripts/csv_to_parquet.py
	python scripts/make_folds.py
	python src/data/prepare_features.py

prep:
	python scripts/make_folds.py
	python src/data/prepare_features.py

# =========================
# Baseline trainers
# =========================
lgbm:
	python src/models/lgbm_baseline.py --data $(DATA)

catboost:
	python src/models/catboost_baseline.py --data $(DATA)

xgb:
	python src/models/xgb_baseline.py --data $(DATA)

trees: lgbm catboost xgb

# Ultra-fast smoke settings for all trees (for logging checks)
trees-fast: smoke-lgbm smoke-cat smoke-xgb

smoke-lgbm:
	python src/models/lgbm_baseline.py --data $(DATA) \
		--num-boost-round 50 --esr 10 --learning-rate 0.3 \
		--num-leaves 15 --max-depth 4 --min-data-in-leaf 2000 \
		--feature-fraction 0.4 --bagging-fraction 0.4 --bagging-freq 1

smoke-xgb:
	python src/models/xgb_baseline.py --data $(DATA) \
		--num-boost-round 100 --esr 20 --eta 0.2 --max-depth 4 \
		--min-child-weight 20 --subsample 0.6 --colsample-bytree 0.6

smoke-cat:
	python src/models/catboost_baseline.py --data $(DATA) \
		--iterations 500 --esr 50 --learning-rate 0.2 \
		--depth 6 --l2-leaf-reg 3.0 --subsample 0.7

nn:
	python src/models/ft_transformer.py

# =========================
# Stacking & analysis
# =========================
stack:
	python src/models/stack_ensemble.py

agg-fi:
	python scripts/aggregate_feature_importance.py --stem train_features

# =========================
# Tuning (Optuna)
# =========================
tune-lgbm:
	@mkdir -p $(ARTIFACTS_DIR) $(REPORTS_DIR) config/tuned
	python src/tune/lgbm_tune.py \
		--data $(DATA) \
		--n-trials $(N_TRIALS) $(if $(TIMEOUT),--timeout $(TIMEOUT)) \
		--study-name zillow_lgbm_mae --storage $(LGB_STORAGE) \
		--pruner $(PRUNER) --max-rounds $(LGB_MAX_ROUNDS) --esr $(LGB_ESR) \
		--sample-rows $(SAMPLE_ROWS) --seed $(SEED) --patience $(PATIENCE) \
		--save-best

tune-xgb:
	@mkdir -p $(ARTIFACTS_DIR) $(REPORTS_DIR) config/tuned
	python src/tune/xgb_tune.py \
		--data $(DATA) \
		--n-trials $(N_TRIALS) $(if $(TIMEOUT),--timeout $(TIMEOUT)) \
		--study-name zillow_xgb_mae --storage $(XGB_STORAGE) \
		--pruner $(PRUNER) --max-rounds $(XGB_MAX_ROUNDS) --esr $(XGB_ESR) \
		--sample-rows $(SAMPLE_ROWS) --seed $(SEED) --patience $(PATIENCE) \
		--save-best

tune-cat:
	@mkdir -p $(ARTIFACTS_DIR) $(REPORTS_DIR) config/tuned
	python src/tune/cat_tune.py \
		--data $(DATA) \
		--n-trials $(N_TRIALS) $(if $(TIMEOUT),--timeout $(TIMEOUT)) \
		--study-name zillow_cat_mae --storage $(CAT_STORAGE) \
		--pruner $(PRUNER) --iterations $(CAT_ITERS) --esr $(CAT_ESR) \
		--sample-rows $(SAMPLE_ROWS) --seed $(SEED) --patience $(PATIENCE) \
		--save-best

tune-all: tune-lgbm tune-xgb tune-cat

# =========================
# Dev utilities
# =========================
mlflow:
	mlflow ui --backend-store-uri "file:$(PWD)/mlruns" --host 0.0.0.0 --port 5000

nb:
	jupyter lab