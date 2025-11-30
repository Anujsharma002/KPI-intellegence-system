import re
import pandas as pd
import streamlit as st
import json
import requests
from datetime import timedelta
from src.utils import load_config

# Load config
CFG = load_config("config/config.yaml")

# Map config fields
DATE_COL = CFG["date_column"]
CAT_COL = CFG["category_column"]
SUBCAT_COL = CFG["subcategory_column"]
PRODUCT_COL = CFG["product_column"]
KPI_COL = CFG["kpi_column"]

PARQUET_PATH = CFG["processed_data_path"]
OLLAMA_URL = CFG["ollama_url"]
OLLAMA_MODEL = CFG["ollama_model"]
DEFAULT_DAYS = CFG["default_trend_days"]


# ------------------------------------------------------
#   1. OLLAMA CALL
# ------------------------------------------------------
def call_llm(prompt):
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=CFG["llm_timeout_seconds"])
        return res.json().get("response", "")
    except Exception as e:
        return f"⚠️ LLM unavailable: {e}"


# ------------------------------------------------------
#   2. ADVANCED NLP PARSER (LLM + fallback)
# ------------------------------------------------------
def parse_query_llm(question, df):
    """
    Ask the LLM to extract structured parameters.
    """

    categories = df[CAT_COL].dropna().unique().tolist()
    subcats = df[SUBCAT_COL].dropna().unique().tolist()
    products = df[PRODUCT_COL].dropna().unique().tolist()

    prompt = f"""
You are a KPI query parser. Extract meaning as structured JSON.

Metrics available: {CFG["extra_numeric_candidates"]}

Categories: {categories}
Subcategories: {subcats}
Products: {products}

Return JSON ONLY:
{{
 "metric": "...",
 "categories": ["..."],
 "products": ["..."],
 "subcategories": ["..."],
 "days": 14,
 "intent": "trend | alert | compare | cause | summary"
}}

User query: "{question}"
"""

    try:
        parsed = json.loads(call_llm(prompt))
        return parsed
    except:
        return None


# ------------------------------------------------------
#   3. Fallback Pattern Parser
# ------------------------------------------------------
def parse_query_fallback(q, df):
    q_lower = q.lower()

    # Days
    match = re.search(r"last\s+(\d+)\s+days?", q_lower)
    days = int(match.group(1)) if match else DEFAULT_DAYS

    # Metric detection
    metric = KPI_COL
    for m in CFG["extra_numeric_candidates"]:
        if m.lower().split("_")[0] in q_lower:
            metric = m

    # Category detection
    found_categories = []
    for c in df[CAT_COL].dropna().unique().tolist():
        if c.lower() in q_lower:
            found_categories.append(c)

    # Handle composite categories
    if "home & kitchen" in q_lower:
        found_categories = ["Home", "Kitchen"]

    return {
        "metric": metric,
        "categories": found_categories,
        "products": [],
        "subcategories": [],
        "days": days,
        "intent": "trend"
    }


# ------------------------------------------------------
#   4. Main Parser
# ------------------------------------------------------
def parse_query(q):
    df = pd.read_parquet(PARQUET_PATH)

    parsed = parse_query_llm(q, df)
    if parsed:
        return parsed

    return parse_query_fallback(q, df)


# ------------------------------------------------------
#   5. Trend + Alert Logic
# ------------------------------------------------------
def analyze(parsed):
    df = pd.read_parquet(PARQUET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Category filters
    if parsed["categories"]:
        df = df[df[CAT_COL].isin(parsed["categories"])]

    # Product filters
    if parsed["products"]:
        df = df[df[PRODUCT_COL].isin(parsed["products"])]

    days = parsed["days"]
    metric = parsed["metric"]

    recent = df.tail(days)
    if recent.empty:
        return "❌ No data found for this filter."

    current = recent[metric].sum()

    baseline = df.tail(CFG["rolling_window_days"])[metric].mean()
    pct_change = ((current - baseline) / baseline) * 100

    if abs(pct_change) < CFG["pct_change_threshold"] * 100:
        return f"📊 KPI stable. Change: {pct_change:.1f} %"

    if pct_change < 0:
        return f"🚨 KPI dropped by {abs(pct_change):.1f}% vs baseline."

    return f"📈 KPI increased by {pct_change:.1f}% vs baseline."


# ------------------------------------------------------
#   6. STREAMLIT UI
# ------------------------------------------------------
st.title("🤖 KPI Conversational Assistant")

q = st.text_input("Ask a question:")

if q:
    st.write("⏳ Processing...")

    parsed = parse_query(q)

    st.subheader("🔍 Parsed Query")
    st.json(parsed)

    st.subheader("📊 Answer")
    st.write(analyze(parsed))
