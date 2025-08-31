import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--stem", default="train_features", help="data_path.stem used in FI filenames")
ap.add_argument("--topk", type=int, default=60)
args = ap.parse_args()

rep = Path("reports")
paths = {
    "lgbm": rep / f"fi_lgbm_base_{args.stem}_detailed.csv",  # has mean_gain column
    "xgb":  rep / f"fi_xgb_base_{args.stem}.csv",            # has mean_gain column
    "cat":  rep / f"fi_cat_base_{args.stem}.csv",            # has mean_pvc column
}

dfs = {}
if paths["lgbm"].exists():
    l = pd.read_csv(paths["lgbm"], index_col=0)[["mean_gain"]].rename(columns={"mean_gain":"lgbm_gain"})
    dfs["lgbm"] = l
if paths["xgb"].exists():
    x = pd.read_csv(paths["xgb"], index_col=0)[["mean_gain"]].rename(columns={"mean_gain":"xgb_gain"})
    dfs["xgb"] = x
if paths["cat"].exists():
    c = pd.read_csv(paths["cat"], index_col=0)[["mean_pvc"]].rename(columns={"mean_pvc":"cat_pvc"})
    dfs["cat"] = c

if not dfs:
    raise SystemExit("No FI CSVs found in reports/. Run model scripts first.")

fi = pd.concat(dfs.values(), axis=1).fillna(0.0)

# Normalize each model’s scale, then compute consensus
for col in fi.columns:
    s = fi[col]
    denom = s.sum()
    fi[col + "_norm"] = (s / denom) if denom > 0 else s

fi["consensus"] = fi.filter(like="_norm").mean(axis=1)

fi_sorted = fi.sort_values("consensus", ascending=False)
out_csv = rep / "fi_combined_consensus.csv"
fi_sorted.to_csv(out_csv)

topk = fi_sorted.head(args.topk)
plt.figure(figsize=(10, max(6, 0.35*len(topk))))
topk["consensus"][::-1].plot(kind="barh")
plt.title(f"Combined Feature Importance (consensus, top {args.topk})")
plt.xlabel("Normalized importance (avg across models)")
plt.tight_layout()
out_png = rep / "fi_combined_top.png"
plt.savefig(out_png, dpi=150)
print(f"Wrote {out_csv} and {out_png}")
