import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/external")
OUT.mkdir(parents=True, exist_ok=True)

def convert_csv(name: str, parse_dates=None):
    src = RAW / name
    if not src.exists():
        print(f"[skip] {src} not found")
        return
    dst = OUT / (src.stem + ".parquet")
    if dst.exists():
        print(f"[ok] {dst} already exists")
        return

    print(f"[convert] {src} -> {dst}")
    df = pd.read_csv(src, parse_dates=parse_dates, low_memory=False)
    df.to_parquet(dst, engine="pyarrow", index=False)
    print(f"[done] wrote {dst}")

if __name__ == "__main__":
    # Train files: parse transactiondate
    convert_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
    convert_csv("train_2017.csv",    parse_dates=["transactiondate"])
    # Properties files
    convert_csv("properties_2016.csv")
    convert_csv("properties_2017.csv")