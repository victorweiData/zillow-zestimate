import os, json, argparse, math, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from catboost import CatBoostRegressor, Pool
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
ap.add_argument("--study-name", default="zillow_cat_mae")
ap.add_argument("--storage", default="sqlite:///data/processed/artifacts/optuna_cat.db")
ap.add_argument("--pruner", choices=["median","hyperband","none"], default="median")
ap.add_argument("--iterations", type=int, default=8000)
ap.add_argument("--esr", type=int, default=100)   # od_wait
ap.add_argument("--sample-rows", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--patience", type=int, default=30)
ap.add_argument("--save-best", action="store_true")
args = ap.parse_args()

# ---------- Data & MLflow ----------
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

def make_convergence_callback(patience: int):
    best = [math.inf]; noimp = [0]
    def _cb(study_, trial):
        if patience <= 0: return
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
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli","MVS","Bayesian"])
    params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 1e2, log=True),
        random_seed=42,
        od_type="Iter",
        od_wait=args.esr,
        thread_count=-1,
        bootstrap_type=bootstrap_type,
        # common knobs
        random_strength=trial.suggest_float("random_strength", 0.0, 1.0),
        border_count=trial.suggest_int("border_count", 32, 255),
        leaf_estimation_iterations=trial.suggest_int("leaf_estimation_iterations", 1, 10),
    )

    # Conditional params
    if bootstrap_type == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    elif bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    # Column sampling (applies to any bootstrap)
    params["rsm"] = trial.suggest_float("rsm", 0.5, 1.0)

    oof = np.zeros(len(df), dtype=np.float64)
    scores = []
    for f in fold_values:
        trn = df[df[fold_col] != f]
        val = df[df[fold_col] == f]
        X_tr = trn[features].apply(pd.to_numeric, errors="coerce")
        X_va = val[features].apply(pd.to_numeric, errors="coerce")

        train_pool = Pool(X_tr, trn[target])
        valid_pool = Pool(X_va, val[target])

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)

        pred = model.predict(X_va)
        fold_mae = np.mean(np.abs(pred - val[target].values))
        oof[val.index] = pred
        scores.append(fold_mae)

        trial.report(fold_mae, step=f)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))

callbacks = [mlf_cb, make_convergence_callback(args.patience)]
study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=callbacks)

best = study.best_trial
Path("config/tuned").mkdir(parents=True, exist_ok=True)
if args.save_best:
    json.dump(best.params, open("config/tuned/cat_best.json", "w"), indent=2)
    print("Saved best params -> config/tuned/cat_best.json")

study.trials_dataframe().to_csv("reports/optuna_cat_trials.csv", index=False)

try:
    from optuna.visualization import plot_param_importances, plot_parallel_coordinate, plot_contour
    plot_param_importances(study).write_html("reports/optuna_cat_importance.html")
    plot_parallel_coordinate(study).write_html("reports/optuna_cat_parallel.html")
    plot_contour(study).write_html("reports/optuna_cat_contour.html")
except Exception as e:
    print("Visualization skipped:", e)

print("Best value:", study.best_value)
print("Best params:", best.params)
