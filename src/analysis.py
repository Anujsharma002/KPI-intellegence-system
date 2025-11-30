# src/analysis.py — FIXED FOR CUSTOM DATE COLUMN

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.utils import load_config
from datetime import timedelta
import math

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:latest"


def call_ollama(prompt: str, model: Optional[str] = None, timeout: int = 180):
    model_to_use = model or DEFAULT_MODEL
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model_to_use, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "(no response)")
    except Exception as e:
        raise RuntimeError(f"LLM error: {e}")


def _compute_pre_post_means(df, feature, alert_date, date_col, window_days):
    pre_start = alert_date - timedelta(days=window_days)
    pre_mask = (df[date_col] >= pre_start) & (df[date_col] < alert_date)
    post_mask = (df[date_col] == alert_date)

    pre = df.loc[pre_mask, feature]
    post = df.loc[post_mask, feature]

    return {
        "pre_mean": float(pre.mean()) if len(pre) else float("nan"),
        "post_mean": float(post.mean()) if len(post) else float("nan"),
    }


def _fit_feature_importance(df, features, kpi_col):
    if df.shape[0] < 20:
        return {f: 0.0 for f in features}

    X = df[features].fillna(0)
    y = df[kpi_col].fillna(0)

    if np.isclose(y.std(), 0):
        return {f: 0.0 for f in features}

    m = RandomForestRegressor(n_estimators=80, random_state=42)
    m.fit(X, y)
    imps = dict(zip(features, m.feature_importances_))
    total = sum(imps.values()) or 1
    return {k: v / total for k, v in imps.items()}


def analyze_causes(processed_path: str, alerts: List[Dict[str, Any]], top_n: int = 5):
    cfg = load_config()

    kpi_col = cfg.get("kpi_column", "Revenue_d")
    date_col = cfg.get("date_column", "Date")
    window = cfg.get("rolling_window_days", 7)
    mode = cfg.get("mode", "llm")

    if not Path(processed_path).exists():
        raise FileNotFoundError(processed_path)

    df = pd.read_parquet(processed_path)

    if date_col not in df.columns:
        raise ValueError(f"Processed data must contain '{date_col}' column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Candidate numeric features
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_candidates = [c for c in numeric_candidates if c != kpi_col]

    extra = cfg.get("extra_numeric_candidates", [])
    numeric_candidates = list(set(numeric_candidates + extra))

    # Global feature importances
    global_imps = _fit_feature_importance(df, numeric_candidates, kpi_col)

    results = []
    for alert in alerts:
        alert_date = pd.to_datetime(alert["date"])

        cause_rows = []
        for feat in numeric_candidates:
            if feat not in df.columns:
                continue

            pm = _compute_pre_post_means(df, feat, alert_date, date_col, window)
            pre, post = pm["pre_mean"], pm["post_mean"]
            if math.isnan(pre) or math.isnan(post):
                continue

            delta = (post - pre) / (pre if pre != 0 else 1)
            importance = global_imps.get(feat, 0.0)

            cause_rows.append({
                "feature": feat,
                "delta": float(delta),
                "importance": float(importance),
                "score": abs(delta) * (importance + 1e-6)
            })

        ranked = sorted(cause_rows, key=lambda x: x["score"], reverse=True)[:top_n]

        # Explanation
        if mode == "fast":
            expl = f"Alert {alert['alert_id']} likely due to: " + ", ".join(
                [c["feature"] for c in ranked]
            )
        else:
            prompt = f"Alert on date {alert_date.date()}. KPI deviation: {alert.get('pct_change'):.2%}. Causes: {ranked}"
            try:
                expl = call_ollama(prompt)
            except Exception:
                expl = f"(LLM failed) Likely causes: {', '.join([c['feature'] for c in ranked])}"

        results.append({
            "alert_id": alert["alert_id"],
            "date": alert["date"],
            "ranked_causes": ranked,
            "explanation": expl,
        })

    # Save JSON for dashboard
    Path("output").mkdir(exist_ok=True)
    with open("output/causes.json", "w") as f:
        json.dump({"causes": results}, f, indent=4)

    return {"causes": results}
