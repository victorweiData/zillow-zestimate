"""
Zillow Prize (Zestimate) — Feature Engineering Pipeline
-------------------------------------------------------
Inputs:
- data/processed/zillow_with_folds.parquet
  (Produced by scripts/make_folds.py; must contain columns:
   ['parcelid','logerror','transactiondate','fold', ... raw property columns])

Outputs:
- data/processed/train_features.parquet  (tree-friendly dtypes)
- data/processed/train_features.csv      (optional via --csv)
- data/processed/artifacts/*             (encoders, meta)

Design goals:
- Strong real-estate logic per feature group (property, location, financial, temporal).
- Fold-safe statistical features (target encoding, neighborhood stats).
- Toggleable groups to iterate quickly.
- Clean column names, appropriate dtypes for GBDTs.
"""

from __future__ import annotations
import argparse, json, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= Config & toggles =========
DEFAULT_TOGGLES = {
    "property_characteristics": True,
    "location_intelligence":    True,
    "financial_indicators":     True,
    "temporal_patterns":        True,
    "statistical_aggs":         True,   # includes OOF target encoding + neighborhood stats
    "interaction_effects":      True,
    "data_quality_metrics":     True,
    "scale_floats":             False,  # trees don't need scaling; turn on if feeding NNs later
}

# Categorical columns to consider for target encoding (if present)
TE_CATS = [
    "regionidzip", "regionidcity", "regionidneighborhood",
    "propertycountylandusecode", "propertylandusetypeid"
]

# Neighborhood/stat group keys (subset will be used if present)
NEIGH_KEYS = [
    "regionidzip", "regionidcity", "geo_cluster_50",
    "propertycountylandusecode"
]

ARTIF_DIR = Path("data/processed/artifacts")


# ========= Utilities =========
def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, alnum + _ only, collapse repeats."""
    def _clean(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^0-9a-z]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")
    df = df.copy()
    df.columns = [_clean(c) for c in df.columns]
    return df

def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a / (b.replace(0, np.nan) + eps)

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    aa = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(aa))

def optimize_tree_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast floats to float32, ints to smallest safe int; keep object->category."""
    out = df.copy()
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")
    for c in out.select_dtypes(include=["int64", "int32"]).columns:
        # keep room for codes like -1
        if out[c].min() >= -128 and out[c].max() <= 127:
            out[c] = out[c].astype("int16")
        elif out[c].min() >= -32768 and out[c].max() <= 32767:
            out[c] = out[c].astype("int16")
        else:
            out[c] = out[c].astype("int32")
    return out

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any() and c not in ("logerror",):
            df[f"{c}__isna"] = df[c].isna().astype("int8")
    return df

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df


# ========= Feature Groups =========
def feat_property_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business logic: interior space, function & livability.
    - size & density ratios
    - effective age of structure
    """
    d = df.copy()

    # Age of home at transaction
    if {"yearbuilt", "transactiondate"}.issubset(d.columns):
        trans_year = pd.to_datetime(d["transactiondate"]).dt.year
        d["home_age"] = (trans_year - d["yearbuilt"]).clip(lower=0, upper=150)
        d["home_age"] = d["home_age"].fillna(d["home_age"].median())

    # Room density / composition
    if {"calculatedfinishedsquarefeet","bedroomcnt","bathroomcnt"}.issubset(d.columns):
        denom = d["bedroomcnt"].fillna(0) + d["bathroomcnt"].fillna(0)
        d["sqft_per_room"] = safe_div(d["calculatedfinishedsquarefeet"].fillna(0), denom)
        d["bath_per_bed"]  = safe_div(d["bathroomcnt"].fillna(0), d["bedroomcnt"].fillna(0))

    # Lot vs improvements
    if {"lotsizesquarefeet","calculatedfinishedsquarefeet"}.issubset(d.columns):
        d["lot_to_building"] = safe_div(d["lotsizesquarefeet"].fillna(0),
                                        d["calculatedfinishedsquarefeet"].fillna(0))

    # Stories proxy: bathrooms + bedrooms per 1000 sqft
    if {"calculatedfinishedsquarefeet","bedroomcnt","bathroomcnt"}.issubset(d.columns):
        sqft = d["calculatedfinishedsquarefeet"].replace(0,np.nan)
        d["rooms_per_1k_sqft"] = (d["bedroomcnt"].fillna(0) + d["bathroomcnt"].fillna(0)) / (sqft/1000.0)

    return d




def feat_location_intelligence(df: pd.DataFrame, fit: bool, km_sample: int = 20000) -> pd.DataFrame:
    """
    Business logic: location drives value.
    - Spherical coords, distances, neighborhood proxies (clusters, grids)
    - Distance to local centroid and LA CBD (proxy amenity)
    Notes:
    - KMeans is fit only when `fit=True` (i.e., training folds).
    - Grid indices are NaN-safe: rows with missing lat/lon get sentinel -1.
    """
    d = df.copy()

    # lat/lon in degrees (Zillow stores as microdegrees)
    if {"latitude","longitude"}.issubset(d.columns):
        lat = d["latitude"] / 1e6
        lon = d["longitude"] / 1e6
        d["lat"] = lat
        d["lon"] = lon

        # Spherical embeddings
        d["x"] = np.cos(np.radians(lat))*np.cos(np.radians(lon))
        d["y"] = np.cos(np.radians(lat))*np.sin(np.radians(lon))
        d["z"] = np.sin(np.radians(lat))

        # Distance to LA City Hall (amenity/proxy CBD)
        d["dist_la_cbd_km"] = haversine_km(lat, lon, 34.0536909, -118.2427666)

        # Grid density: ~2km grid; NaN-safe with sentinel -1
        step = 0.02
        mask = lat.notna() & lon.notna()
        lat_g = pd.Series(np.int64(-1), index=d.index)
        lon_g = pd.Series(np.int64(-1), index=d.index)
        lat_g.loc[mask] = np.round((lat[mask] / step)).astype(np.int64)
        lon_g.loc[mask] = np.round((lon[mask] / step)).astype(np.int64)

        # Combine into a single key without bit-shifts (pandas doesn't support << on Series)
        d["geo_grid"] = (lat_g * (1 << 20)) + lon_g
        grid_counts = d["geo_grid"].value_counts()
        d["grid_density"] = d["geo_grid"].map(grid_counts).fillna(1).astype("int32")

        # Zip/city centroid distance (local amenity proxy)
        for key in ("regionidzip","regionidcity"):
            if key in d.columns:
                gp = d.groupby(key)[["lat","lon"]].median()
                d[f"dist_{key}_centroid_km"] = haversine_km(
                    lat, lon,
                    gp.reindex(d[key]).lat.values,
                    gp.reindex(d[key]).lon.values
                )

        # KMeans neighborhood clusters (fit-time only)
        if fit:
            coords = np.column_stack([
                lat.fillna(lat.median()),
                lon.fillna(lon.median())
            ]).astype("float32")
            n = len(coords)
            idx = np.random.RandomState(42).choice(n, size=min(km_sample, n), replace=False)
            for k in (20, 50):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(coords[idx])
                d[f"geo_cluster_{k}"] = km.predict(coords).astype("int32")
        else:
            # At transform time, expect cluster columns already present (no refit).
            pass

    return d
def feat_financial_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business logic: assessed values & taxes reflect value.
    - tax per sqft, structure/land proportions, tax rate
    """
    d = df.copy()
    if {"taxvaluedollarcnt","calculatedfinishedsquarefeet"}.issubset(d.columns):
        d["tax_per_sqft"] = safe_div(d["taxvaluedollarcnt"].fillna(0),
                                     d["calculatedfinishedsquarefeet"].fillna(0))
    if {"structuretaxvaluedollarcnt","taxvaluedollarcnt"}.issubset(d.columns):
        d["structure_ratio"] = safe_div(d["structuretaxvaluedollarcnt"].fillna(0),
                                        d["taxvaluedollarcnt"].fillna(0))
    if {"landtaxvaluedollarcnt","taxvaluedollarcnt"}.issubset(d.columns):
        d["land_ratio"] = safe_div(d["landtaxvaluedollarcnt"].fillna(0),
                                   d["taxvaluedollarcnt"].fillna(0))
    if {"taxamount","taxvaluedollarcnt"}.issubset(d.columns):
        d["effective_tax_rate"] = safe_div(d["taxamount"].fillna(0),
                                           d["taxvaluedollarcnt"].fillna(0))
    return d


def feat_temporal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business logic: seasonality & market cycles matter.
    - month/quarter seasonality; months since baseline
    - time since last sale (per parcel)
    """
    d = df.copy()
    if "transactiondate" in d.columns:
        d["transactiondate"] = pd.to_datetime(d["transactiondate"])
        d["trans_year"]  = d["transactiondate"].dt.year
        d["trans_month"] = d["transactiondate"].dt.month
        d["trans_qtr"]   = d["transactiondate"].dt.quarter
        m = d["trans_month"].astype("float32")
        d["sin_month"] = np.sin(2*np.pi*m/12.0)
        d["cos_month"] = np.cos(2*np.pi*m/12.0)
        # Months since a baseline (Jan 2016) — market phase proxy
        base = pd.Timestamp("2016-01-01")
        d["months_since_2016"] = ( (d["transactiondate"] - base).dt.days // 30 ).astype("int16")

    # Time since last sale per parcel (uses within-dataset history only)
    if {"parcelid","transactiondate"}.issubset(d.columns):
        d = d.sort_values(["parcelid","transactiondate"]).copy()
        prev_date = d.groupby("parcelid")["transactiondate"].shift(1)
        d["days_since_prev_sale"] = (d["transactiondate"] - prev_date).dt.days
        d["days_since_prev_sale"] = d["days_since_prev_sale"].fillna(-1).astype("int32")
        d = d.sort_index()

    return d


def _foldwise_map(train: pd.DataFrame, valid: pd.DataFrame, key: str, src_cols: list[str], agg: str):
    """
    Helper to compute fold-safe group stats on train and map to valid.
    """
    gp = train.groupby(key)[src_cols].agg(agg)
    gp.columns = [f"{key}__{c}__{agg}" for c in src_cols]
    mapped = gp.reindex(valid[key]).reset_index(drop=True)
    return mapped, gp


def feat_statistical_aggs(df: pd.DataFrame, n_folds: int, te_smoothing: float = 20.0) -> tuple[pd.DataFrame, dict]:
    """
    Fold-safe:
    - Target encoding (mean logerror) for selected categoricals with smoothing.
    - Neighborhood/property stats: group medians/means of strong physical/financial features.
    Returns df and artifacts dict (per-fold mappings + global priors).
    """
    d = df.copy()
    artifacts: dict = {"te_priors": {}, "neigh_stats": {}}

    # ===== Target Encoding (OOF) =====
    if "logerror" in d.columns:
        global_mean = d["logerror"].mean()
        artifacts["te_priors"]["global_mean"] = float(global_mean)

        for col in TE_CATS:
            if col not in d.columns: 
                continue
            oof = np.zeros(len(d), dtype="float32")
            artifacts["te_priors"][col] = {}

            for f in range(n_folds):
                trn = d[d.fold != f]
                val = d[d.fold == f]
                # per-category stats on train
                stats = trn.groupby(col)["logerror"].agg(["sum","count"])
                # smoothed mean
                te_map = (stats["sum"] + te_smoothing*global_mean) / (stats["count"] + te_smoothing)
                oof[val.index] = val[col].map(te_map).fillna(global_mean).astype("float32")
                # store mapping size (for debug/metadata)
                artifacts["te_priors"][col][f"n_keys_fold_{f}"] = int(stats.shape[0])

            d[f"te_{col}"] = oof.astype("float32")

        # Global backfill maps (optional, for future inference)
        global_maps = {}
        for col in TE_CATS:
            if col in d.columns:
                stats = d.groupby(col)["logerror"].agg(["sum","count"])
                te_map = (stats["sum"] + te_smoothing*global_mean) / (stats["count"] + te_smoothing)
                global_maps[col] = te_map.to_dict()
        artifacts["te_global"] = global_maps

    # ===== Neighborhood / Property Stats (OOF) =====
    base_src = [c for c in [
        "taxvaluedollarcnt","calculatedfinishedsquarefeet",
        "lotsizesquarefeet","bathroomcnt","bedroomcnt"
    ] if c in d.columns]

    for key in NEIGH_KEYS:
        if key not in d.columns or not base_src:
            continue

        for agg in ("median","mean"):
            col_names = [f"{key}__{c}__{agg}" for c in base_src]
            d[col_names] = np.nan  # allocate

            per_fold_store = {}
            for f in range(n_folds):
                trn = d[d.fold != f]
                val = d[d.fold == f]
                mapped, gp = _foldwise_map(trn, val, key, base_src, agg)
                d.loc[val.index, col_names] = mapped.values
                per_fold_store[f] = {"n_keys": int(gp.shape[0])}
            artifacts["neigh_stats"][f"{key}__{agg}"] = per_fold_store

    return d, artifacts


def feat_interaction_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business logic: interactions capture non-additive effects.
    Examples: age × size, tax rate × land/structure mix.
    """
    d = df.copy()
    if {"home_age","calculatedfinishedsquarefeet"}.issubset(d.columns):
        d["age_x_sqft"] = d["home_age"] * d["calculatedfinishedsquarefeet"].fillna(d["calculatedfinishedsquarefeet"].median())
    if {"effective_tax_rate","structure_ratio"}.issubset(d.columns):
        d["taxrate_x_structure"] = d["effective_tax_rate"] * d["structure_ratio"].fillna(0)
    if {"effective_tax_rate","land_ratio"}.issubset(d.columns):
        d["taxrate_x_land"] = d["effective_tax_rate"] * d["land_ratio"].fillna(0)
    if {"sqft_per_room","bath_per_bed"}.issubset(d.columns):
        d["space_efficiency"] = d["sqft_per_room"].fillna(0) * (1.0 + d["bath_per_bed"].fillna(0))
    return d


def feat_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business logic: completeness often correlates with reliability & value.
    - row-wise missing fraction, non-null count, simple quality score
    """
    d = df.copy()
    base_cols = [c for c in d.columns if c not in ["logerror","transactiondate","fold","parcelid"]]
    miss = d[base_cols].isna().sum(axis=1).astype("int16")
    d["missing_frac"] = (miss / max(1, len(base_cols))).astype("float32")
    d["non_null_count"] = (len(base_cols) - miss).astype("int16")
    d["data_quality_score"] = (1.0 - d["missing_frac"]).astype("float32")
    return d


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For trees: map object→category codes. Preserve -1 for NaN codes.
    """
    d = df.copy()
    obj_cols = d.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        d[c] = d[c].astype("category").cat.codes.astype("int32")
    return d


# ========= Main pipeline =========
def prepare_features(cfg_path: str, toggles: dict, write_csv: bool = False) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    proc = Path(cfg["paths"]["processed_dir"]); proc.mkdir(parents=True, exist_ok=True)
    ARTIF_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_parquet(proc/"zillow_with_folds.parquet")
    df = clean_colnames(df)
    df = ensure_datetime(df, "transactiondate")

    # Baseline missing flags first (used by multiple groups)
    if toggles.get("data_quality_metrics", True):
        df = add_missing_flags(df)

    n_folds = int(cfg.get("cv", {}).get("n_folds", 5))

    # Property
    if toggles.get("property_characteristics", True):
        df = feat_property_characteristics(df)

    # Location (fit-time clusters)
    if toggles.get("location_intelligence", True):
        df = feat_location_intelligence(df, fit=True)

    # Financials
    if toggles.get("financial_indicators", True):
        df = feat_financial_indicators(df)

    # Temporal
    if toggles.get("temporal_patterns", True):
        df = feat_temporal_patterns(df)

    # Statistical (OOF target enc + neighborhood stats)
    artifacts = {}
    if toggles.get("statistical_aggs", True):
        df, artifacts = feat_statistical_aggs(df, n_folds=n_folds, te_smoothing=20.0)

    # Interactions
    if toggles.get("interaction_effects", True):
        df = feat_interaction_effects(df)

    # Data quality summary (after adding features)
    if toggles.get("data_quality_metrics", True):
        df = feat_data_quality(df)

    # Encode categoricals for trees
    df = encode_categoricals(df)

    # Optional scaling for floats (off by default)
    if toggles.get("scale_floats", False):
        float_cols = [c for c in df.select_dtypes(include=["float32","float64"]).columns if c != "logerror"]
        df[float_cols] = (df[float_cols] - df[float_cols].mean()) / (df[float_cols].std() + 1e-9)

    # Final dtypes for tree models
    df = optimize_tree_dtypes(df)

    # Save
    out_parq = proc/"train_features.parquet"
    df.to_parquet(out_parq, index=False)
    if write_csv:
        df.to_csv(proc/"train_features.csv", index=False)

    # Save metadata/artifacts (for reproducibility)
    meta = {
        "toggles": toggles,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "artifacts_keys": list(artifacts.keys()),
    }
    (ARTIF_DIR/"meta.json").write_text(json.dumps(meta, indent=2))
    if "te_global" in artifacts:
        # store compact TE maps (can be large; keep as json of dict sizes)
        sizes = {k: len(v) for k,v in artifacts["te_global"].items()}
        (ARTIF_DIR/"te_global_sizes.json").write_text(json.dumps(sizes, indent=2))

    print(f"✅ Wrote {out_parq} with shape={df.shape}")
    if write_csv:
        print(f"✅ Wrote {proc/'train_features.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    ap.add_argument("--csv", action="store_true", help="Also write CSV output")
    # Toggle overrides from CLI (e.g., --no-te or --no-location)
    ap.add_argument("--no-property", action="store_true")
    ap.add_argument("--no-location", action="store_true")
    ap.add_argument("--no-financial", action="store_true")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--no-stat", action="store_true")
    ap.add_argument("--no-interact", action="store_true")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--scale-floats", action="store_true")
    args = ap.parse_args()

    toggles = DEFAULT_TOGGLES.copy()
    if args.no_property:  toggles["property_characteristics"] = False
    if args.no_location:  toggles["location_intelligence"]    = False
    if args.no_financial: toggles["financial_indicators"]     = False
    if args.no_temporal:  toggles["temporal_patterns"]        = False
    if args.no_stat:      toggles["statistical_aggs"]         = False
    if args.no_interact:  toggles["interaction_effects"]      = False
    if args.no_quality:   toggles["data_quality_metrics"]     = False
    if args.scale_floats: toggles["scale_floats"]             = True

    prepare_features(args.config, toggles, write_csv=args.csv)


if __name__ == "__main__":
    main()
