import pandas as pd
from pathlib import Path
from src.utils import ensure_dirs, load_config

def ingest(csv_path: str):
    cfg = load_config()
    
    date_col = cfg.get("date_column", "Date")
    processed_path = cfg.get("processed_data_path", "data/processed/kpi.parquet")

    print(f"[ingest] Loading CSV → {csv_path}")
    df = pd.read_csv(csv_path)

    required = [
        date_col,
        cfg.get("kpi_column", "Revenue_d"),
        cfg.get("category_column", "Category"),
        cfg.get("product_column", "Product_Name")
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.sort_values(date_col).reset_index(drop=True)

    ensure_dirs()

    df.to_parquet(processed_path, index=False)

    print(f"[ingest] Processed data saved → {processed_path}")
    return {"processed_path": processed_path}
