# src/ingest.py
import pandas as pd
from pathlib import Path
from src.utils import ensure_dirs, load_config

def ingest(csv_path: str):
    cfg = load_config()
    
    date_col = cfg.get("date_column", "Date")
    processed_path = cfg.get("processed_data_path", "data/processed/kpi.parquet")

    print(f"[ingest] Loading CSV → {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate essential columns exist
    required = [
        date_col,
        cfg.get("kpi_column", "Revenue_d"),
        cfg.get("category_column", "Category"),
        cfg.get("product_column", "Product_Name")
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardize date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows without valid date
    df = df.dropna(subset=[date_col])

    # Clean whitespace in string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Sort
    df = df.sort_values(date_col).reset_index(drop=True)

    # Create output folders
    ensure_dirs()

    # Save parquet
    df.to_parquet(processed_path, index=False)

    print(f"[ingest] Processed data saved → {processed_path}")
    return {"processed_path": processed_path}
