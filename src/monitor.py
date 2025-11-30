# src/monitor.py
import pandas as pd
from pathlib import Path
from src.utils import load_config, save_csv
from datetime import datetime
import requests
import json


def call_llm_for_alert(cfg, text):
    """Optional LLM message for natural-language interpretation."""
    if not cfg.get("use_llm_analysis", False):
        return None

    try:
        payload = {
            "model": cfg["ollama_model"],
            "prompt": f"Explain this KPI deviation in simple business language:\n{text}",
            "stream": False
        }
        response = requests.post(cfg["ollama_url"], json=payload, timeout=60)
        return response.json().get("response", "")
    except Exception as e:
        return f"LLM error: {e}"


def classify_severity(pct_change, zscore, cfg):
    """Return Low / Medium / High severity label."""
    drop_pct = abs(pct_change) * 100
    z = abs(zscore)

    if drop_pct > (cfg["pct_drop_threshold"] * 100) * 2 or z > (cfg["zscore_threshold"] + 1):
        return "HIGH"
    elif drop_pct > (cfg["pct_drop_threshold"] * 100):
        return "MEDIUM"
    return "LOW"


def detect_alerts(processed_path: str):
    cfg = load_config()

    date_col = cfg["date_column"]
    kpi_col = cfg["kpi_column"]
    window = cfg["rolling_window_days"]
    alerts_out = cfg["alerts_output_path"]

    enable_pct = cfg["enable_pct_change"]
    pct_threshold = cfg["pct_drop_threshold"]
    enable_z = cfg["enable_zscore"]
    z_th = cfg["zscore_threshold"]

    if not Path(processed_path).exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    # Load data
    df = pd.read_parquet(processed_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Rolling stats
    df["rolling_mean"] = df[kpi_col].rolling(window).mean()
    df["rolling_std"] = df[kpi_col].rolling(window).std()

    alerts = []

    for idx, row in df.iterrows():
        if pd.isna(row["rolling_mean"]):
            continue  # warm-up window

        value = row[kpi_col]
        baseline = row["rolling_mean"]
        std = row["rolling_std"]

        pct_change = (value - baseline) / baseline
        zscore = (value - baseline) / std if std and std > 0 else 0

        is_alert = False
        reasons = []

        # Percent-based deviation
        if enable_pct and pct_change < -pct_threshold:
            is_alert = True
            reasons.append(f"Percent drop: {pct_change:.1%}")

        # Z-score deviation
        if enable_z and zscore < -z_th:
            is_alert = True
            reasons.append(f"Z-score: {zscore:.2f}")

        if not is_alert:
            continue

        severity = classify_severity(pct_change, zscore, cfg)

        # Generate human-readable text
        alert_text = (
            f"{severity} ALERT — {kpi_col} deviated on {row[date_col].date()}\n"
            f"Actual: {value:.2f} | Baseline: {baseline:.2f}\n"
            f"Drop: {pct_change:.1%} | Z-score: {zscore:.2f}\n"
            f"Reasons: {', '.join(reasons)}"
        )

        # Optional LLM interpretation
        llm_msg = call_llm_for_alert(cfg, alert_text)

        full_message = alert_text
        if llm_msg:
            full_message += f"\n\nLLM Interpretation:\n{llm_msg}"

        alerts.append({
            "alert_id": f"A{idx}",
            "date": str(row[date_col].date()),
            "value": float(value),
            "baseline": float(baseline),
            "pct_change": float(pct_change),
            "zscore": float(zscore),
            "severity": severity,
            "alert_message": full_message
        })

    alerts_df = pd.DataFrame(alerts)
    save_csv(alerts_df, alerts_out)

    print(f"[monitor] {len(alerts)} readable alerts saved → {alerts_out}")
    return {"alerts": alerts}
