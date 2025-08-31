from __future__ import annotations
import numpy as np
import pandas as pd

def mem_str(bytes_: float) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if bytes_ < 1024: return f"{bytes_:.2f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.2f} PB"

def report_memory(df: pd.DataFrame, label: str = "df") -> None:
    print(f"[memory] {label}: {mem_str(df.memory_usage(deep=True).sum())} "
          f"({df.shape[0]:,} rows × {df.shape[1]} cols)")

def optimize_dtypes(df: pd.DataFrame, 
                    downcast_float: str = "float32",
                    downcast_int: str = "integer",
                    convert_obj_to_cat: bool = True,
                    cat_threshold: float = 0.5) -> pd.DataFrame:
    """
    Downcast numerics and optionally convert low-cardinality object columns to 'category'.
    - cat_threshold: if unique_ratio <= cat_threshold -> category
    """
    before = df.memory_usage(deep=True).sum()

    # Numerics
    for col, dt in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(dt):
            # 'float32' reduces mem; keep float64 if extreme precision needed
            if downcast_float == "float32":
                df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_bool_dtype(dt):
            df[col] = df[col].astype('bool')  # already minimal
        elif pd.api.types.is_object_dtype(dt) and convert_obj_to_cat:
            # Convert to category if reasonably low-cardinality
            n_unique = df[col].nunique(dropna=True)
            ratio = n_unique / max(1, len(df))
            if ratio <= cat_threshold:
                df[col] = df[col].astype('category')

    after = df.memory_usage(deep=True).sum()
    print(f"[memory] optimized: {mem_str(before)} -> {mem_str(after)} "
          f"({(1 - after/max(before,1)) * 100:.1f}% saved)")
    return df
