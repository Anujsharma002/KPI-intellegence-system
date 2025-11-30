
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import timedelta
from src.utils import load_config
from crew_agents import (
    ingest_agent_fn,
    monitor_agent_fn,
    analysis_agent_fn,
    actions_agent_fn,
)

cfg = load_config()

DATE_COL = cfg.get("date_column", "Date")
KPI_COL = cfg.get("kpi_column", "Revenue_d")
CAT_COL = cfg.get("category_column", "Category")
PROD_COL = cfg.get("product_column", "Product_Name")

PARQUET_PATH = cfg.get("processed_data_path", "data/processed/kpi.parquet")
ALERTS_PATH = cfg.get("alerts_output_path", "output/alerts_log.csv")
RECS_PATH = cfg.get("recommendations_output_path", "output/recommendations.csv")
CAUSES_FILE = "output/causes.json"

def load_df():
    if Path(PARQUET_PATH).exists():
        df = pd.read_parquet(PARQUET_PATH)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        return df.sort_values(DATE_COL)
    return pd.DataFrame()

def load_alerts():
    return pd.read_csv(ALERTS_PATH) if Path(ALERTS_PATH).exists() else pd.DataFrame()

def load_recs():
    return pd.read_csv(RECS_PATH) if Path(RECS_PATH).exists() else pd.DataFrame()

def run_full_pipeline():
    st.write("🚀 Running full KPI pipeline...")

    state = {"raw_path": cfg.get("raw_data_path", "data/raw/kpi_data.csv")}

    # Ingest
    st.write("📥 Ingesting dataset...")
    state = ingest_agent_fn(state)

    # Monitor
    st.write("📉 Detecting deviations...")
    state = monitor_agent_fn(state)

    # Analysis (FAST by config)
    st.write("🧠 Running causal analysis...")
    state = analysis_agent_fn(state)

    # Save causes.json
    import json
    Path("output").mkdir(exist_ok=True)
    with open(CAUSES_FILE, "w") as f:
        json.dump({"causes": state.get("causes", [])}, f, indent=4)

    # Actions
    st.write("💡 Generating recommendations...")
    state = actions_agent_fn(state)

    # Save recommendations
    recs = state.get("recommendations", [])
    if recs:
        pd.DataFrame(recs).to_csv(RECS_PATH, index=False)

    st.success("✅ Full pipeline completed!")
    return True

st.set_page_config(page_title="KPI Intelligence Dashboard", layout="wide")
st.title("📈 KPI Intelligence System")
st.caption("Monitoring • Causal Analysis • Recommendations")

st.sidebar.header("⚙️ Pipeline Controls")

if st.sidebar.button("🚀 Run Full KPI Pipeline"):
    run_full_pipeline()
    st.rerun()   # NEW recommended function

# Reload after pipeline run
df = load_df()
alerts_df = load_alerts()
recs_df = load_recs()

st.header("📊 KPI Trends")

if df.empty:
    st.info("Dataset not found. Please run the pipeline.")
else:
    cat_filter = st.selectbox(
        "Select Category:",
        ["All"] + sorted(df[CAT_COL].dropna().unique().tolist())
    )

    df_f = df if cat_filter == "All" else df[df[CAT_COL] == cat_filter]

    fig = px.line(
        df_f,
        x=DATE_COL,
        y=KPI_COL,
        color=None if cat_filter != "All" else CAT_COL,
        title=f"{KPI_COL} Trend ({cat_filter})"
    )
    st.plotly_chart(fig, width="stretch")

st.header("🚨 KPI Alerts")

def color_severity(level):
    if level.upper() == "HIGH":
        return "🔴 HIGH"
    if level.upper() == "MEDIUM":
        return "🟠 MEDIUM"
    return "🟡 LOW"

if alerts_df.empty:
    st.info("No alerts detected. Run pipeline.")
else:
    st.dataframe(alerts_df)

    selected_alert = st.selectbox(
        "Select Alert:",
        alerts_df["alert_id"].tolist()
    )

    if selected_alert:
        alert = alerts_df[alerts_df["alert_id"] == selected_alert].iloc[0]

        st.subheader(f"🔍 Alert Details — {selected_alert}")

        st.write(f"**Date:** `{alert['date']}`")
        st.write(f"**Severity:** {color_severity(alert['severity'])}")
        st.write(f"**KPI Value:** `{alert['value']}`")
        st.write(f"**Baseline:** `{alert['baseline']}`")
        st.write(f"**Deviation:** `{alert['pct_change']*100:.1f}%`")
        st.write(f"**Z-score:** `{alert['zscore']:.2f}`")

        # NEW readable alert message
        st.info(alert["alert_message"])

        st.subheader("🧠 Causal Analysis")

        import json
        if Path(CAUSES_FILE).exists():
            causes = json.load(open(CAUSES_FILE)).get("causes", [])
            item = next((x for x in causes if x["alert_id"] == selected_alert), None)

            if item:
                st.write("### Ranked Causes")
                st.write(item["ranked_causes"])
                st.write("### Explanation")
                st.write(item["explanation"])
            else:
                st.info("No causal analysis for this alert.")
        else:
            st.info("Run pipeline to generate causal analysis.")

        st.divider()

        st.subheader("💡 Recommendations")

        match = recs_df[recs_df["alert_id"] == selected_alert]
        if not match.empty:
            st.markdown(match.iloc[0]["recommendation_text"])
        else:
            st.info("Run pipeline to generate recommendations.")
