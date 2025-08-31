import argparse, sys, os
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy import stats

def read_single_col(path, col=None):
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if col is None:
        # use the first non-index column
        col = [c for c in df.columns if c not in ["index","Unnamed: 0"]][0]
    return df[col].to_numpy(), col

def mae_per_fold(y, yhat, folds):
    out = []
    for f in sorted(pd.unique(folds)):
        m = folds == f
        out.append(mean_absolute_error(y[m], yhat[m]))
    return np.array(out)

def bootstrap_delta(y, a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        deltas[i] = mean_absolute_error(y[idx], b[idx]) - mean_absolute_error(y[idx], a[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return deltas.mean(), (lo, hi)

def main():
    ap = argparse.ArgumentParser(description="Compare two OOF prediction files for significance.")
    ap.add_argument("--proc", default="data/processed/train_proc.parquet", help="Processed train parquet with target & fold.")
    ap.add_argument("--y-col", default="logerror")
    ap.add_argument("--fold-col", default="fold")
    ap.add_argument("--a", required=True, help="Path to model A OOF (baseline).")
    ap.add_argument("--a-col", default=None, help="Column in A file (defaults to first).")
    ap.add_argument("--b", required=True, help="Path to model B OOF (candidate).")
    ap.add_argument("--b-col", default=None, help="Column in B file (defaults to first).")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_parquet(args.proc)
    if args.y_col not in df.columns or args.fold_col not in df.columns:
        sys.exit(f"ERROR: {args.proc} must contain '{args.y_col}' and '{args.fold_col}'")

    y = df[args.y_col].to_numpy()
    folds = df[args.fold_col].to_numpy()

    a, a_name = read_single_col(args.a, args.a_col)
    b, b_name = read_single_col(args.b, args.b_col)

    if len(a) != len(y) or len(b) != len(y):
        sys.exit(f"ERROR: Length mismatch: y={len(y)}, A={len(a)}, B={len(b)}")

    # Overall MAE
    mae_a = mean_absolute_error(y, a)
    mae_b = mean_absolute_error(y, b)

    # Per-fold stats
    folds_a = mae_per_fold(y, a, folds)
    folds_b = mae_per_fold(y, b, folds)

    # Paired tests
    tstat, p_t = stats.ttest_rel(folds_a, folds_b)
    # Wilcoxon requires >0 diffs present; handle edge cases
    try:
        wstat, p_w = stats.wilcoxon(folds_a, folds_b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        wstat, p_w = np.nan, np.nan

    # Bootstrap CI of delta (B - A): negative is improvement
    delta_mean, (delta_lo, delta_hi) = bootstrap_delta(y, a, b, n_boot=args.nboot)

    # Row-wise win-rate: abs error improved
    win_rate = np.mean(np.abs(b - y) < np.abs(a - y))

    # Print
    print("="*72)
    print(f"Y: {args.proc} [{args.y_col}]  Folds: '{args.fold_col}'  n={len(y)}  k={len(np.unique(folds))}")
    print(f"A (baseline): {args.a} [{a_name}]   MAE={mae_a:.6f}")
    print(f"B (candidate): {args.b} [{b_name}]   MAE={mae_b:.6f}")
    print("-"*72)
    print("Per-fold MAE (A):", np.round(folds_a, 6))
    print("Per-fold MAE (B):", np.round(folds_b, 6))
    print("-"*72)
    print(f"Paired t-test:      p={p_t:.6g}  (H0: mean fold MAE equal)")
    print(f"Wilcoxon signed-rank: p={p_w:.6g}  (nonparametric)")
    print(f"Bootstrap Δ(B-A) MAE: mean={delta_mean:+.6f}, 95% CI=({delta_lo:+.6f}, {delta_hi:+.6f})")
    print(f"Row-wise win-rate (|B-y| < |A-y|): {win_rate*100:.2f}%")
    print("="*72)

if __name__ == "__main__":
    main()
