import pandas as pd, yaml
from pathlib import Path
from src.utils.memory import report_memory, optimize_dtypes

cfg = yaml.safe_load(open("config/config.yaml"))
raw = Path(cfg["paths"]["raw_dir"])
ext = Path("data/external")
proc = Path(cfg["paths"]["processed_dir"])
proc.mkdir(parents=True, exist_ok=True)

use_train_cols = ["parcelid","logerror","transactiondate"]

def prefer_parquet(csv_name: str, parse_dates=None):
    pq = ext / (Path(csv_name).stem + ".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    # fallback to CSV (auto compression handled by csv_to_parquet.py normally)
    comp = "infer"
    with open(raw/csv_name, "rb") as f:
        if f.read(2) == b"\x1f\x8b": comp = "gzip"
    return pd.read_csv(raw/csv_name, compression=comp, parse_dates=parse_dates, low_memory=False)

def load_year(year: int) -> pd.DataFrame:
    if year == 2016:
        t, p = "train_2016_v2.csv", "properties_2016.csv"
    elif year == 2017:
        t, p = "train_2017.csv", "properties_2017.csv"
    else:
        raise ValueError("Year must be 2016 or 2017")
    train = prefer_parquet(t, parse_dates=["transactiondate"])[use_train_cols]
    props = prefer_parquet(p)
    df = train.merge(props, on="parcelid", how="left")
    df["year"] = year
    return df

print("Loading…")
df = pd.concat([load_year(2016), load_year(2017)], ignore_index=True)

report_memory(df, "merged_raw")
# Clip target tails
df = df.sort_values(cfg["cv"]["date_col"]).reset_index(drop=True)
q01, q99 = df["logerror"].quantile([0.01, 0.99])
df["logerror"] = df["logerror"].clip(q01, q99)

# Time-based folds
n_folds = cfg["cv"]["n_folds"]
dates = pd.to_datetime(df[cfg["cv"]["date_col"]])
df["fold"] = pd.qcut(dates.rank(method="first"), n_folds, labels=False)

# Optimize dtypes before saving
df = optimize_dtypes(df, convert_obj_to_cat=True, cat_threshold=0.2)
report_memory(df, "merged_opt")

out = proc/"zillow_with_folds.parquet"
df.to_parquet(out, index=False)
print("Wrote", out)
