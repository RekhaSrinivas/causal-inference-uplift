"""
Downloads and caches the full Criteo Uplift Modeling Dataset (~14M rows).
Saves as parquet for fast reloads.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _download_criteo():
    from sklift.datasets import fetch_criteo

    print("[data_loader] Downloading Criteo Uplift dataset (~300 MB)...")
    data = fetch_criteo()

    df = data["data"].copy()
    df["treatment"] = data["treatment"].values
    df["visit"] = data["target"].values
    return df


def load_data():
    cache = DATA_DIR / "criteo_full.parquet"
    DATA_DIR.mkdir(exist_ok=True)

    if cache.exists():
        print("[data_loader] Loading cached dataset...")
        return pd.read_parquet(cache)

    df = _download_criteo()
    df.to_parquet(cache, index=False)
    print(f"[data_loader] Cached {len(df):,} rows to {cache}")
    return df
