import argparse, json, shlex, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["lgbm","xgb","cat","all"], default="all")
ap.add_argument("--data", default="data/processed/train_features.parquet")
args = ap.parse_args()

def run(cmd):
    print(">>", cmd)
    subprocess.run(shlex.split(cmd), check=True)

def train_lgbm():
    jp = Path("config/tuned/lgbm_best.json")
    params = json.load(open(jp)) if jp.exists() else {}
    # derive fields
    if "num_leaves_log2" in params and "num_leaves" not in params:
        params["num_leaves"] = 1 << int(params.pop("num_leaves_log2"))
    # map optuna -> CLI flags supported by lgbm_baseline.py
    m = {
        "learning_rate": "--learning-rate",
        "num_leaves": "--num-leaves",
        "feature_fraction": "--feature-fraction",
        "bagging_fraction": "--bagging-fraction",
        "bagging_freq": "--bagging-freq",
        "min_data_in_leaf": "--min-data-in-leaf",
        "max_depth": "--max-depth",
        "lambda_l1": "--reg-alpha",
        "lambda_l2": "--reg-lambda",
        # min_split_gain is not exposed in your baseline; ignored
    }
    flags = " ".join(f"{m[k]} {params[k]}" for k in m if k in params)
    cmd = f"python src/models/lgbm_baseline.py --data {args.data} {flags}"
    run(cmd)

def train_xgb():
    jp = Path("config/tuned/xgb_best.json")
    params = json.load(open(jp)) if jp.exists() else {}
    # force compatibility for our baseline
    params["num_parallel_tree"] = 1  # MAE objective requires this
    m = {
        "eta": "--eta",
        "max_depth": "--max-depth",
        "min_child_weight": "--min-child-weight",
        "subsample": "--subsample",
        "colsample_bytree": "--colsample-bytree",
        "reg_lambda": "--lambda_",
        # (baseline lacks flags for reg_alpha, gamma, colsample_bylevel/bynode/max_bin/grow_policy)
    }
    flags = " ".join(f"{m[k]} {params[k]}" for k in m if k in params)
    cmd = f"python src/models/xgb_baseline.py --data {args.data} {flags}"
    run(cmd)

def train_cat():
    jp = Path("config/tuned/cat_best.json")
    params = json.load(open(jp)) if jp.exists() else {}
    m = {
        "learning_rate": "--learning-rate",
        "depth": "--depth",
        "l2_leaf_reg": "--l2-leaf-reg",
        # optional extras (baseline may not expose): random_strength, border_count, leaf_estimation_iterations
    }
    flags = " ".join(f"{m[k]} {params[k]}" for k in m if k in params)
    # add a reasonable iterations/od if not provided on CLI; you can override
    cmd = f"python src/models/catboost_baseline.py --data {args.data} {flags} --iterations 6000 --esr 200 --use-cats"
    run(cmd)

if args.model in ("lgbm","all"): train_lgbm()
if args.model in ("xgb","all"):  train_xgb()
if args.model in ("cat","all"):  train_cat()
