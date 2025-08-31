# scripts/make_submission_catboost.py
import pandas as pd, numpy as np, gc
from pathlib import Path
from catboost import CatBoostRegressor

RAW = Path("data/raw")
OUT = Path("submissions"); OUT.mkdir(parents=True, exist_ok=True)

print("Loading raw CSVs…")
prop16 = pd.read_csv(RAW/"properties_2016.csv", low_memory=False)
prop17 = pd.read_csv(RAW/"properties_2017.csv", low_memory=False)
trn16  = pd.read_csv(RAW/"train_2016_v2.csv", parse_dates=["transactiondate"], low_memory=False)
trn17  = pd.read_csv(RAW/"train_2017.csv",    parse_dates=["transactiondate"], low_memory=False)
sample = pd.read_csv(RAW/"sample_submission.csv", low_memory=False)

def add_date_features(df):
    df["transaction_year"]        = df["transactiondate"].dt.year
    df["transaction_month_ix"]    = (df["transactiondate"].dt.year - 2016) * 12 + df["transactiondate"].dt.month
    df["transaction_quarter_ix"]  = (df["transactiondate"].dt.year - 2016) * 4  + df["transactiondate"].dt.quarter
    df.drop(columns=["transactiondate"], inplace=True)
    return df

print("Merging train with properties…")
trn16 = add_date_features(trn16).merge(prop16, how="left", on="parcelid")
trn17 = add_date_features(trn17).merge(prop17, how="left", on="parcelid")

# Optional: zero-out 2017 tax columns (avoid cross-year leakage)
print("Zeroing 2017 tax columns…")
tax_cols = [c for c in trn17.columns if c.startswith("tax")]
if tax_cols:
    trn17.loc[:, tax_cols] = np.nan

train = pd.concat([trn16, trn17], axis=0, ignore_index=True)

# Build test from 2016 properties and ADD the same date features
test = sample[["ParcelId"]].merge(prop16.rename(columns={"parcelid":"ParcelId"}), how="left", on="ParcelId")
test["transactiondate"] = pd.Timestamp("2016-12-01")  # constant pseudo date
test = add_date_features(test)

del prop16, prop17, trn16, trn17
gc.collect()

print("Selecting features (exclude high-missing, unique-only, and known non-features)…")
exclude_missing = []
num_rows = len(train)
for c in train.columns:
    if train[c].isna().any():
        if train[c].isna().sum() / float(num_rows) > 0.98:
            exclude_missing.append(c)

exclude_unique = []
for c in train.columns:
    u = train[c].nunique(dropna=True)
    if u <= 1:
        exclude_unique.append(c)

exclude_other = ["parcelid", "logerror", "propertyzoningdesc"]  # kernel-like exclusions
features = [c for c in train.columns
            if c not in exclude_missing
            and c not in exclude_unique
            and c not in exclude_other]

# Ensure test has all training feature columns (add missing as NaN)
missing_in_test = [c for c in features if c not in test.columns]
for c in missing_in_test:
    test[c] = np.nan

print(f"Using {len(features)} features")
print("Detecting categorical features…")
cat_inds = []
for i, c in enumerate(features):
    u = train[c].nunique(dropna=True)
    if (u < 1000) and not any(k in c for k in ("sqft","cnt","nbr","number")):
        cat_inds.append(i)

print(f"Categorical features: {len(cat_inds)}")

print("Fill NA with -999 (keeping it kernel-style)")
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

X = train[features].copy()
y = train["logerror"].astype(float)
X_test = test[features].copy()

# >>> IMPORTANT FIX: cast categorical columns to STRING for CatBoost <<<
if cat_inds:
    cat_cols = [features[i] for i in cat_inds]
    # CatBoost requires cat_features to be int or str; floats are invalid.
    # Convert to string (including the -999 sentinel, which becomes "-999").
    X.loc[:, cat_cols] = X[cat_cols].astype(str)
    X_test.loc[:, cat_cols] = X_test[cat_cols].astype(str)

print("Training CatBoost (fast, kernel-like settings)…")
num_ensembles = 5
pred_test = 0.0
for seed in range(num_ensembles):
    model = CatBoostRegressor(
        iterations=630,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=False
    )
    model.fit(X, y, cat_features=cat_inds if cat_inds else None)
    pred_test += model.predict(X_test)
pred_test /= num_ensembles

print("Writing submission…")
sub = pd.DataFrame({"ParcelId": test["ParcelId"]})
for label in ["201610","201611","201612","201710","201711","201712"]:
    sub[label] = pred_test

out_path = OUT/"only_catboost.csv"
sub.to_csv(out_path, index=False, float_format="%.6f")
print(f"Saved: {out_path.resolve()}")