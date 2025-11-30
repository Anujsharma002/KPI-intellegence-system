
import json
import re
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from src.utils import load_config
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

CFG = load_config("config/config.yaml")

DATE_COL = CFG["date_column"]
CAT_COL = CFG["category_column"]
PRODUCT_COL = CFG["product_column"]
SUBCAT_COL = CFG["subcategory_column"]
KPI_COL = CFG["kpi_column"]

PARQUET_PATH = CFG["processed_data_path"]
OLLAMA_URL = CFG["ollama_url"]
OLLAMA_MODEL = CFG["ollama_model"]

MEMORY_LIMIT = CFG.get("chat_memory_length", 5)
chat_memory = []

def call_llm(prompt):
    """Call Ollama with streaming enabled."""
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)

        full = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                j = json.loads(line.decode())
                full += j.get("response", "")
            except:
                pass

        return full.strip()

    except Exception as e:
        return f"[red]⚠️ LLM error: {e}[/red]"

def detect_intent(q):
    q = q.lower()

    if q.startswith("a") and q[1:].isdigit():
        return "alert"

    if any(k in q for k in ["fix", "actions", "recommend", "what should we do"]):
        return "actions"

    if any(k in q for k in ["why", "cause", "driver", "explain", "drop", "decline"]):
        return "cause"

    if "full" in q and "pipeline" in q:
        return "full_pipeline"

    return "trend"

def parse_query_llm(question, df):
    """Let LLM extract metric/category/product."""
    categories = df[CAT_COL].dropna().unique().tolist()
    products = df[PRODUCT_COL].dropna().unique().tolist()

    prompt = f"""
Extract meaning from this KPI question.

Return JSON with fields:
{{
 "metric": "...",
 "categories": ["..."],
 "products": ["..."],
 "days": 14
}}

Metrics: {CFG["extra_numeric_candidates"]}
Categories: {categories[:20]}
Products: {products[:20]}

Query: "{question}"
"""

    try:
        return json.loads(call_llm(prompt))
    except:
        return None


def parse_query_fallback(q, df):
    """Regex and keyword fallback parser."""
    q = q.lower()
    days = 14

    match = re.search(r"last\s+(\d+)\s+days?", q)
    if match:
        days = int(match.group(1))

    metric = KPI_COL
    for m in CFG["extra_numeric_candidates"]:
        if m.lower() in q:
            metric = m

    cats = [c for c in df[CAT_COL].dropna().unique() if c.lower() in q]
    prods = [p for p in df[PRODUCT_COL].dropna().unique() if p.lower() in q]

    return {
        "metric": metric,
        "categories": cats,
        "products": prods,
        "days": days
    }


def parse_query(q):
    df = pd.read_parquet(PARQUET_PATH)
    parsed = parse_query_llm(q, df) or parse_query_fallback(q, df)
    parsed["intent"] = detect_intent(q)
    parsed["raw"] = q
    return parsed


def analyze(parsed):
    df = pd.read_parquet(PARQUET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Filter
    if parsed["categories"]:
        df = df[df[CAT_COL].isin(parsed["categories"])]
    if parsed["products"]:
        df = df[df[PRODUCT_COL].isin(parsed["products"])]

    # NO DATA FOUND
    if df.empty:
        return None, None, None, "empty_filter"

    metric = parsed["metric"]

    # WRONG METRIC
    if metric not in df.columns:
        return None, None, None, "bad_metric"

    # Trend period
    days = parsed["days"]
    recent = df.tail(days)

    if recent.empty:
        return None, None, None, "no_recent_data"

    # Safe KPI calculations
    current_val = recent[metric].mean()

    baseline_window = CFG["rolling_window_days"]
    baseline_series = df.tail(baseline_window)[metric]

    if baseline_series.empty or baseline_series.mean() == 0 or pd.isna(baseline_series.mean()):
        return current_val, None, None, "no_baseline"

    baseline = baseline_series.mean()
    pct_change = ((current_val - baseline) / baseline) * 100

    return pct_change, current_val, baseline, None


def show_alert(alert_id):
    alerts_path = CFG["alerts_output_path"]
    if not Path(alerts_path).exists():
        console.print("[red]No alerts logged yet![/red]")
        return

    alerts = pd.read_csv(alerts_path)

    if alert_id not in alerts["alert_id"].values:
        console.print("[red]Alert not found.[/red]")
        return

    row = alerts[alerts["alert_id"] == alert_id].iloc[0]

    console.print(Panel(
        f"[bold]🔍 ALERT {alert_id}[/bold]\n\n"
        f"📅 Date: {row['date']}\n"
        f"📉 Value: {row['value']}\n"
        f"📈 Deviation: {row['pct_change']:.2%}\n"
        f"🔎 Reason: {row.get('reason','N/A')}",
        border_style="red"
    ))


def causal_analysis(parsed):
    df = pd.read_parquet(PARQUET_PATH).tail(parsed["days"])
    stats = df.describe().to_string()

    prompt = f"""
We detected possible KPI deviation.

Dataset (last {parsed['days']} days) stats:
{stats}

Explain top 3 likely causal factors in bullet points.
"""
    return call_llm(prompt)


def recommend_actions(causes):
    prompt = f"""
Based on these causes:

{causes}

Generate 3 prioritized business actions.
"""
    return call_llm(prompt)

def sparkline(series):
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(series), max(series)
    span = mx - mn or 1
    return "".join(blocks[int((v - mn) / span * 7)] for v in series)


def run_full_pipeline(parsed):
    pct, cur, base, err = analyze(parsed)

    # Handle errors
    if err:
        error_messages = {
            "empty_filter": "[red]No matching data for selected category/product.[/red]",
            "bad_metric": "[red]Metric not found in dataset.[/red]",
            "no_recent_data": "[red]Not enough recent data to compute trend.[/red]",
            "no_baseline": "[red]Cannot compute baseline — insufficient data.[/red]",
        }
        console.print(Panel(error_messages[err], expand=False))
        return

    df = pd.read_parquet(PARQUET_PATH)[parsed["metric"]].tail(30)

    console.print(Panel(
        f"[bold cyan]📊 TREND SUMMARY[/bold cyan]\n\n"
        f"Current: {cur:.2f}\nBaseline: {base:.2f}\n"
        f"Change: {pct:.1f}%\n\n"
        f"[green]{sparkline(df.tolist())}[/green]",
        expand=False
    ))

    if pct < 0:
        console.print(Panel("[yellow]🔍 CAUSAL ANALYSIS[/yellow]"))
        causes = causal_analysis(parsed)
        console.print(causes)

        console.print(Panel("[green]🧭 RECOMMENDATIONS[/green]"))
        console.print(recommend_actions(causes))
    else:
        console.print("[green]KPI stable — no decline detected.[/green]")


def run_cli():
    console.print(Panel(
        "[bold magenta]🤖 KPI Intelligence CLI[/bold magenta]\n"
        "Ask questions like:\n"
        "• Show revenue trend for Kitchen last 14 days\n"
        "• Why did Electronics decline?\n"
        "• Give actions to fix Home category\n"
        "• A14 (see alert)\n",
        expand=False
    ))

    while True:
        q = console.input("[bold blue]🧑‍💻 You:[/bold blue] ").strip()

        if q.lower() in ["exit", "quit"]:
            console.print("[red]Exiting...[/red]")
            break

        parsed = parse_query(q)
        intent = parsed["intent"]

        # ---- ALERT MODE ----
        if intent == "alert":
            show_alert(q.upper())
            continue

        # ---- FULL PIPELINE ----
        if intent == "full_pipeline":
            run_full_pipeline(parsed)
            continue

        # ---- TREND ----
        if intent == "trend":
            pct, cur, base, err = analyze(parsed)

            if err:
                console.print(Panel(f"[red]Error: {err}[/red]", expand=False))
                continue

            console.print(Panel(
                f"📈 Trend Change: {pct:.1f}% vs baseline.\n"
                f"Current={cur:.0f}   Baseline={base:.0f}",
                expand=False
            ))
            continue

        # ---- CAUSE ----
        if intent == "cause":
            console.print(Panel("🔍 CAUSAL ANALYSIS"))
            console.print(causal_analysis(parsed))
            continue

        # ---- ACTIONS ----
        if intent == "actions":
            console.print(Panel("🧭 RECOMMENDATIONS"))
            causes = causal_analysis(parsed)
            console.print(recommend_actions(causes))
            continue


if __name__ == "__main__":
    run_cli()
