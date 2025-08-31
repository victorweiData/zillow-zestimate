import os, json, argparse, math, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.integration import MLflowCallback

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/config.yaml")
ap.add_argument("--data", default=None)
ap.add_argument("--target", default="logerror")
ap.add_argument("--fold-col", default="fold")
ap.add_argument("--exclude", default="transactiondate,parcelid,year")
ap.add_argument("--n-trials", type=int, default=100)
ap.add_argument("--timeout", type=int, default=None)
ap.add_argument("--study-name", default="zillow_lgbm_mae")
ap.add_argument("--storage", default="sqlite:///data/processed/artifacts/optuna_lgbm.db")
ap.add_argument("--pruner", choices=["median","hyperband","none"], default="median")
ap.add_argument("--max-rounds", type=int, default=3000)
ap.add_argument("--esr", type=int, default=100)
ap.add_argument("--sample-rows", type=int, default=0, help="0=full; else random sample size")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--patience", type=int, default=30, help="early stop study if no improvement N trials; 0=off")
ap.add_argument("--save-best", action="store_true", help="write best params to config/tuned/lgbm_best.json")
args = ap.parse_args()

# ---------- Data ----------
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "Zillow Baselines")
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment(EXPERIMENT)

cfg = yaml.safe_load(open(args.config))
proc = Path(cfg["paths"]["processed_dir"])
data_path = Path(args.data) if args.data else proc/"train_features.parquet"
df = pd.read_parquet(data_path)

target = args.target
fold_col = args.fold_col
base_excl = {"logerror", fold_col, target, *args.exclude.split(",")}
features = [c for c in df.columns if c not in base_excl]
if args.sample_rows and args.sample_rows < len(df):
    df = df.sample(n=args.sample_rows, random_state=args.seed)
# Ensure contiguous row indices for safe OOF assignment
df = df.reset_index(drop=True)
# Use only the folds that actually exist in this (possibly sampled) frame
fold_values = sorted(pd.Series(df[fold_col]).dropna().unique())
n_folds = len(fold_values)

# ---------- Optuna setup ----------
sampler = TPESampler(seed=args.seed, multivariate=True)
if args.pruner == "median":
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)
elif args.pruner == "hyperband":
    pruner = HyperbandPruner()
else:
    pruner = optuna.pruners.NopPruner()

study = optuna.create_study(
    direction="minimize",
    study_name=args.study_name,
    storage=args.storage,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner,
)

# Convergence callback
def make_convergence_callback(patience: int):
    best = [math.inf]; noimp = [0]
    def _cb(study_, trial):
        if patience <= 0: 
            return
        v = study_.best_value
        if v + 1e-12 < best[0]:
            best[0] = v; noimp[0] = 0
        else:
            noimp[0] += 1
            if noimp[0] >= patience:
                print(f"[Optuna] Early stop study: no improvement for {patience} trials.")
                study_.stop()
    return _cb

mlf_cb = MLflowCallback(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    metric_name="oof_mae",
    create_experiment=True
)

def objective(trial: optuna.Trial) -> float:
    # --- Sample hyperparameters (proper distributions) ---
    max_depth = trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12])

    # num_leaves constrained by max_depth when depth > 0
    if max_depth is None or max_depth <= 0:
        leaves_upper = 1024
    else:
        leaves_upper = min(1024, int(2 ** max_depth))
    low_log2 = 4  # 16
    high_log2 = int(math.log2(leaves_upper))
    if high_log2 < low_log2:  # safety
        high_log2 = low_log2
    num_leaves = 2 ** trial.suggest_int("num_leaves_log2", low_log2, high_log2)

    params = dict(
        objective="mae",
        metric="mae",
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 5000, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-3, 1.0, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 0, 7),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        verbose=-1,
        n_jobs=-1,
        feature_pre_filter=False,
    )
    # Alias for version-compat (LightGBM accepts either; we set both)
    params["min_gain_to_split"] = params["min_split_gain"]

    # Optional extras
    if trial.suggest_categorical("use_feature_fraction_bynode", [False, True]):
        params["feature_fraction_bynode"] = trial.suggest_float("feature_fraction_bynode", 0.5, 1.0)

    # --- CV over your existing folds ---
    oof = np.zeros(len(df), dtype=np.float64)
    scores = []

    for f in fold_values:
        trn = df[df[fold_col] != f]
        val = df[df[fold_col] == f]
        X_tr = trn[features].apply(pd.to_numeric, errors="coerce")
        X_va = val[features].apply(pd.to_numeric, errors="coerce")

        dtrain = lgb.Dataset(X_tr, trn[target])
        dvalid = lgb.Dataset(X_va, val[target], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=args.max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(args.esr, verbose=False)]
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        fold_mae = np.mean(np.abs(pred - val[target].values))
        oof[val.index] = pred
        scores.append(fold_mae)

        # report to pruner
        trial.report(fold_mae, step=f)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))

# Optimize
callbacks = [mlf_cb, make_convergence_callback(args.patience)]
study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=callbacks)

# Save artifacts
best = study.best_trial
Path("config/tuned").mkdir(parents=True, exist_ok=True)
best_path = Path("config/tuned/lgbm_best.json")
if args.save_best:
    params = best.params.copy()
    # materialize derived params
    if "num_leaves_log2" in params:
        params["num_leaves"] = int(2 ** params.pop("num_leaves_log2"))
    json.dump(params, open(best_path, "w"), indent=2)
    print(f"Saved best params -> {best_path}")

# Trials CSV
df_trials = study.trials_dataframe()
df_trials.to_csv("reports/optuna_lgbm_trials.csv", index=False)

# Visualization (optional, requires plotly)
try:
    from optuna.visualization import plot_param_importances, plot_parallel_coordinate, plot_contour
    plot_param_importances(study).write_html("reports/optuna_lgbm_importance.html")
    plot_parallel_coordinate(study).write_html("reports/optuna_lgbm_parallel.html")
    plot_contour(study).write_html("reports/optuna_lgbm_contour.html")
except Exception as e:
    print("Visualization skipped:", e)

print("Best value:", study.best_value)
print("Best params:", best.params)
