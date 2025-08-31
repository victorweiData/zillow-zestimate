# Zillow Zestimate – End‑to‑End Tabular ML Pipeline

This repository contains a **reproducible workflow** for the Kaggle *Zillow Prize: Home Value Prediction* task:
- data acquisition & preprocessing
- rich, domain‑driven **feature engineering**
- **tree baselines** (LightGBM / XGBoost / CatBoost) with MLflow tracking
- **hyperparameter tuning** with Optuna (robust search spaces + pruners)
- **OOF stacking** (Ridge)
- **feature importance aggregation → selection → retrain**
- **notebook** to run the selection + retrain pipeline interactively
- quick **submission** generation (CatBoost baseline)

> Everything logs to **MLflow** (local file store) so you can compare runs, params, artifacts, and figures.

---

## 1) Environment

```bash
# conda (macOS auto-switch; linux uses mamba/conda)
make env
# or plain venv + pip
make env-pip
```

Environment variables used:
```bash
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
export MLFLOW_EXPERIMENT_NAME="Zillow Baselines"
```

---

## 2) Data Pipeline

Raw files expected under `data/raw/` (Kaggle Zillow):
- `properties_2016.csv`, `properties_2017.csv`
- `train_2016_v2.csv`, `train_2017.csv`
- `sample_submission.csv`

Run the full pipeline:
```bash
make data
# -> data/processed/zillow_with_folds.parquet
# -> data/processed/train_features.parquet
```

What happens:
1. `scripts/get_data.sh` *(optional stub)* downloads/copies raw.
2. `scripts/csv_to_parquet.py` converts large CSVs to parquet.
3. `scripts/make_folds.py` merges, optimizes dtypes, builds stratified **KFold** (`fold` column).
4. `src/data/prepare_features.py` creates domain features (see below) → `train_features.parquet`.

### Feature Engineering Highlights
- **Property characteristics**: sqft/room ratios, bath/bed, age, tax ratios, winsorized heavy tails.
- **Location intelligence**: lat/lon to xyz, distance to LA, geo KMeans clusters, neighborhood freq encodings.
- **Financial indicators**: tax per sqft, structure/land/tax ratios.
- **Temporal patterns**: year/month/quarter, sin/cos month; (optionally) time since sale.
- **Data quality**: row‑wise missing fraction; per‑field missing flags.
- **Types**: integers are small dtypes, floats standardized, objects → categories/ints; safe encoders for trees.

---

## 3) Baseline Training (Trees)

Train all three:
```bash
make trees
# or individually
make lgbm
make xgb
make catboost
```

Artifacts:
- OOF predictions: `models/oof_lgb_base.parquet`, `oof_xgb_base.parquet`, `oof_cat_base.parquet`
- Feature importances & plots under `reports/`
- MLflow run with params, metrics, figures

**CatBoost** supports `--use-cats` to pass raw high‑card categorical columns using `Pool(cat_features=...)`.

---

## 4) MLflow

Launch UI:
```bash
make mlflow
# open http://localhost:5000
```

Each run logs:
- parameters & CV stats
- OOF MAE + per‑fold MAE
- feature importance CSV/PNG
- scatter/residual plots
- saved models and (for LGBM/XGB) fold ensembles

---

## 5) Hyperparameter Tuning (Optuna)

Run all tuners (LGBM/XGB/Cat):
```bash
make tune-all
```
Each tuner:
- supports **Hyperband** or **Median** pruning
- time‑/row‑subsampling for fast iterations: `--sample-rows 80000`
- logs trials to MLflow and SQLite (`data/processed/artifacts/optuna_*.db`)
- writes best params to `config/tuned/*_best.json`

Resume or adjust trials:
```bash
make tune-lgbm
make tune-xgb
make tune-cat
```

Common tips handled in the scripts:
- LGBM: `num_leaves < 2^max_depth`, alias `min_gain_to_split`/`min_split_gain`
- XGB: extra knobs `colsample_bylevel`, `colsample_bynode` (optional), ensure `num_parallel_tree=1` for MAE
- Cat: conditional `subsample` when `bootstrap_type=Bernoulli`

---

## 6) Refit with Best Params

Use the helper to refit baselines from `config/tuned/*.json`:
```bash
python scripts/train_with_best.py --model all
# or: --model lgbm|xgb|cat
```
Re‑creates OOFs and logs full artifacts.

---

## 7) Stacking

Simple ridge stack on available OOFs:
```bash
python src/models/stack_ensemble.py
# -> models/oof_stack.parquet
# -> reports/stack_weights_auto.csv
```
Prints OOF MAE and learned weights.

> Optionally run month‑bias diagnostics:
> ```bash
> python scripts/check_month_bias.py
> # optional debias
> python src/models/debias_month_offsets.py
> python src/models/stack_ensemble.py
> ```

---

## 8) Feature Importance Aggregation → Selection → Retrain

Aggregate FI across models:
```bash
python scripts/aggregate_feature_importance.py --stem train_features --topk 60
# -> reports/fi_combined_consensus.csv
# -> reports/fi_combined_top.png
```

Select features (top‑K or cumulative coverage) and write allow‑list:
```bash
python scripts/select_features.py --method topk --k 120
# or
python scripts/select_features.py --method cumimp --threshold 0.95

# pointer file used by trainers:
# config/feature_lists/selected_features.txt
```

Retrain all models with the allow‑list:
```bash
python src/models/lgbm_baseline.py  --data data/processed/train_features.parquet --feature-list config/feature_lists/selected_features.txt
python src/models/xgb_baseline.py   --data data/processed/train_features.parquet --feature-list config/feature_lists/selected_features.txt
python src/models/catboost_baseline.py --data data/processed/train_features.parquet --feature-list config/feature_lists/selected_features.txt --use-cats

python src/models/stack_ensemble.py
```

---

## 9) Notebook – Interactive Selection & Retrain

A ready‑to‑run notebook is provided:
- **`zillow_feature_selection_and_retrain.ipynb`** (artifact in this README folder when downloaded)
- Aggregates FI → selects features → retrains all trees → stacks → optional month‑bias check
- Toggle **smoke run** for quick iterations

Open in JupyterLab and run top‑to‑bottom.

---

## 10) Submission

Quick CatBoost‑only submission (kernel‑style):
```bash
python scripts/make_submission_catboost.py
# -> submissions/only_catboost.csv
```
This script merges train with properties (2016/2017), creates consistent date features, handles NA, detects categorical columns, and trains a small CatBoost ensemble. It then writes the 6 required month columns for Kaggle.

> A stacked submission script can be added (predict test per model, apply ridge weights).

---

## 11) Makefile Cheat‑Sheet

```make
# env
make env            # conda/mamba
make env-pip        # venv + pip

# pipeline
make data           # raw → parquet → folds → features
make prep           # folds + features only

# baselines
make trees          # lgbm + catboost + xgb
make lgbm
make catboost
make xgb

# tuning
make tune-all
make tune-lgbm
make tune-xgb
make tune-cat

# stack & notebook
make stack
make nb             # JupyterLab
make mlflow         # MLflow UI
```

---

## 12) Troubleshooting

**CatBoost: `Invalid type for cat_feature ... = 1.0`**  
Cause: predicting on a numeric DataFrame while training used `Pool(cat_features=...)`.  
Fixes:
- Predict on the **same type** you trained eval with, e.g. `valid_pool`.
- Or cast categorical columns to `str` on both train/valid/test before `predict`.

**XGBoost: `Boosting random forest is not supported for current objective`**  
Cause: `num_parallel_tree > 1` with `reg:absoluteerror`.  
Fix: ensure `num_parallel_tree = 1` (or drop param).

**LightGBM: `train and valid dataset categorical_feature do not match`**  
Cause: mismatched categorical handling or dtypes across folds.  
Fix: don’t pass `categorical_feature` lists unless strictly necessary; ensure identical columns and dtypes across folds.

**Optuna sample size indexing error (`index out of bounds`)**  
Cause: allocating OOF by full dataset size but indexing sampled rows.  
Fix: allocate `oof` with `len(subset)` or assign via subset indices; scripts are patched accordingly.

---

## 13) Files & Folders (key ones)

```
config/
  config.yaml
  tuned/
    lgbm_best.json
    xgb_best.json
    cat_best.json
  feature_lists/
    selected_features.txt  # pointer (generated)
data/
  raw/                     # Kaggle CSVs
  processed/
    zillow_with_folds.parquet
    train_features.parquet
    artifacts/optuna_*.db
models/
  oof_*.parquet            # OOF predictions
  lgb_ensemble.joblib      # saved folds (LGBM)
reports/
  fi_*                     # feature importance CSV/PNGs
  training_summary_*.json
scripts/
  get_data.sh
  csv_to_parquet.py
  make_submission_catboost.py
  aggregate_feature_importance.py
  select_features.py
  train_with_best.py
src/
  data/prepare_features.py
  models/{lgbm_baseline.py,xgb_baseline.py,catboost_baseline.py,stack_ensemble.py}
  tune/{lgbm_tune.py,xgb_tune.py,cat_tune.py}
  utils/...
```

---

## 14) Repro Quickstart

```bash
# 1) Data + features
make data

# 2) Baselines + stack
make trees
python src/models/stack_ensemble.py

# 3) Tune
make tune-all

# 4) Refit with best params
python scripts/train_with_best.py --model all

# 5) Aggregate FI → select → retrain + stack
python scripts/aggregate_feature_importance.py --stem train_features --topk 60
python scripts/select_features.py --method topk --k 120
python src/models/lgbm_baseline.py  --data data/processed/train_features.parquet --feature-list config/feature_lists/selected_features.txt
python src/models/xgb_baseline.py   --data data/processed/train_features.parquet --feature-list config/feature_lists/selected_features.txt
python src/models/catboost_baseline.py --data data/processed/train_features.parquet --feature-list config/feature_lists/selected_features.txt --use-cats
python src/models/stack_ensemble.py

# 6) Submission (CatBoost baseline)
python scripts/make_submission_catboost.py
```

---

**Author**: Victor Wei
