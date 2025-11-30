import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_config(path="config/config.yaml"):
    return yaml.safe_load(open(path))

def ensure_dirs():
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def save_csv(df, path):
    df.to_csv(path, index=False)

def timestamp():
    return datetime.utcnow().isoformat()

def read_kpi_csv(path: str):
    df = pd.read_csv(path, parse_dates=True)
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c: "date"})
                break
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")
