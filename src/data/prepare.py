import pandas as pd, numpy as np, yaml
from pathlib import Path
from sklearn.cluster import KMeans
from src.utils.memory import report_memory, optimize_dtypes

# ----------------- load -----------------
cfg = yaml.safe_load(open("config/config.yaml"))
proc = Path(cfg["paths"]["processed_dir"])
df = pd.read_parquet(proc / "zillow_with_folds.parquet")
report_memory(df, "pre_feat")

# ----------------- helpers -----------------
truthy = {"y","yes","true","t","1","on"}
falsy  = {"n","no","false","f","0","off"}

def is_boolish_series(s: pd.Series) -> bool:
    u = pd.Series(s.dropna().astype(str).str.strip().str.lower().unique())
    if len(u) == 0: return True
    return set(u).issubset(truthy.union(falsy))

def safe_div(a, b):
    return a / (b.replace(0, np.nan) + 1e-3)

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def add_basic_features(df):
    d = df

    # lat/lon derived + distance to LA
    if {"latitude","longitude"}.issubset(d.columns):
        lat = d["latitude"] / 1e6
        lon = d["longitude"] / 1e6
        d["lat"] = lat.astype("float32"); d["lon"] = lon.astype("float32")
        d["x"] = (np.cos(np.radians(lat))*np.cos(np.radians(lon))).astype("float32")
        d["y"] = (np.cos(np.radians(lat))*np.sin(np.radians(lon))).astype("float32")
        d["z"] = (np.sin(np.radians(lat))).astype("float32")
        d["dist_la"] = haversine(lat, lon, 34.0522, -118.2437).astype("float32")

    # time features
    if "transactiondate" in d.columns:
        d["transactiondate"] = pd.to_datetime(d["transactiondate"])
        d["trans_year"]  = d["transactiondate"].dt.year.astype("int16")
        d["trans_month"] = d["transactiondate"].dt.month.astype("int8")
        d["trans_qtr"]   = d["transactiondate"].dt.quarter.astype("int8")
        m = d["trans_month"].astype("float32")
        d["sin_month"] = np.sin(2*np.pi*(m/12)).astype("float32")
        d["cos_month"] = np.cos(2*np.pi*(m/12)).astype("float32")

    # age
    if {"yearbuilt"}.issubset(d.columns) and "trans_year" in d.columns:
        age = (d["trans_year"] - d["yearbuilt"]).clip(lower=0, upper=150)
        d["home_age"] = age.fillna(age.median()).astype("float32")

    # ratios / densities
    if {"calculatedfinishedsquarefeet","bedroomcnt","bathroomcnt"}.issubset(d.columns):
        denom = d["bedroomcnt"].fillna(0) + d["bathroomcnt"].fillna(0)
        d["sqft_per_room"] = safe_div(d["calculatedfinishedsquarefeet"].fillna(0), denom).astype("float32")
        d["bath_per_bed"]  = safe_div(d["bathroomcnt"].fillna(0), d["bedroomcnt"].fillna(0)).astype("float32")

    if {"taxvaluedollarcnt","calculatedfinishedsquarefeet"}.issubset(d.columns):
        d["taxvalue_per_sqft"] = safe_div(
            d["taxvaluedollarcnt"].fillna(0),
            d["calculatedfinishedsquarefeet"].fillna(0)
        ).astype("float32")

    if {"structuretaxvaluedollarcnt","taxvaluedollarcnt"}.issubset(d.columns):
        d["structure_pct"] = safe_div(d["structuretaxvaluedollarcnt"].fillna(0),
                                      d["taxvaluedollarcnt"].fillna(0)).astype("float32")
    if {"landtaxvaluedollarcnt","taxvaluedollarcnt"}.issubset(d.columns):
        d["land_pct"] = safe_div(d["landtaxvaluedollarcnt"].fillna(0),
                                 d["taxvaluedollarcnt"].fillna(0)).astype("float32")
    if {"taxamount","taxvaluedollarcnt"}.issubset(d.columns):
        d["tax_rate"] = safe_div(d["taxamount"].fillna(0),
                                 d["taxvaluedollarcnt"].fillna(0)).astype("float32")

    # missingness
    base_cols = [c for c in d.columns if c not in ["logerror","transactiondate","fold"]]
    miss = d[base_cols].isna().sum(axis=1).astype("int16")
    d["missing_frac"] = (miss / max(1, len(base_cols))).astype("float32")

    # frequency encodings (safe for trees)
    def freq_enc(col):
        vc = d[col].astype("object").fillna("__nan__").value_counts(dropna=False)
        mp = vc / len(d)
        d[f"{col}__freq"] = d[col].astype("object").fillna("__nan__").map(mp).astype("float32")

    for col in ["propertycountylandusecode","regionidzip","regionidcity","regionidneighborhood"]:
        if col in d.columns:
            freq_enc(col)

    # geo clusters
    if {"lat","lon"}.issubset(d.columns):
        coords = np.column_stack([
            d["lat"].fillna(d["lat"].median()),
            d["lon"].fillna(d["lon"].median())
        ]).astype("float32")
        for k in (20, 50):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            idx = np.random.RandomState(42).choice(len(coords), size=min(20000, len(coords)), replace=False)
            km.fit(coords[idx])
            d[f"geo_cluster_{k}"] = km.predict(coords).astype("int32")

    # winsorize heavy tails
    for col in ["calculatedfinishedsquarefeet","taxamount","taxvaluedollarcnt"]:
        if col in d.columns:
            lo, hi = d[col].quantile([0.005, 0.995])
            d[f"{col}_wz"] = d[col].clip(lo, hi).astype("float32")

    return d

# ----------------- missing flags -----------------
for c in df.columns:
    if df[c].isna().any():
        df[f"{c}__isna"] = df[c].isna().astype("int8")

# ----------------- object -> bool/int/codes -----------------
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
for c in obj_cols:
    s = df[c]
    if is_boolish_series(s):
        s_norm = s.astype(str).str.strip().str.lower()
        df[c] = np.where(s.isna(), 0, s_norm.isin(truthy)).astype("int8")
    else:
        df[c] = s.astype("category").cat.codes.astype("int32")  # -1 for NaN stays

# ----------------- feature engineering (DO THIS BEFORE SCALING) -----------------
df = add_basic_features(df)

# ----------------- tidy types & scaling -----------------
if "fold" in df.columns and not pd.api.types.is_integer_dtype(df["fold"]):
    df["fold"] = df["fold"].astype("int8")

# standardize floats (exclude target)
float_cols = df.select_dtypes(include=["float32","float64"]).columns.tolist()
if "logerror" in float_cols:
    float_cols.remove("logerror")
df[float_cols] = df[float_cols].astype("float32")
df[float_cols] = (df[float_cols] - df[float_cols].mean()) / (df[float_cols].std() + 1e-9)

df = optimize_dtypes(df, convert_obj_to_cat=False)
report_memory(df, "post_feat")

# ----------------- save -----------------
out = proc / "train_proc.parquet"
df.to_parquet(out, index=False)
print("Wrote", out)