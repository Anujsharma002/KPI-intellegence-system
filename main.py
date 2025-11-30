import typer
import sys
import os
import subprocess
from src.utils import ensure_dirs

ensure_dirs()

from crew_agents import (
    ingest_agent_fn,
    monitor_agent_fn,
    analysis_agent_fn,
    actions_agent_fn,
)

USE_CREW = True
try:
    import crewai
    from crewai import Crew
    from crew_agents import (
        data_ingest_agent,
        monitor_agent,
        analysis_agent,
        actions_agent,
    )
except Exception as e:
    print("[main] CrewAI not usable, switching to fallback.\n", e)
    USE_CREW = False


app = typer.Typer(help="🔹 KPI Intelligence System CLI")


def run_fallback_pipeline():
    print("[main] Running fallback sequential pipeline...")
    state = {"raw_path": "data/raw/kpi_data.csv"}

    state = ingest_agent_fn(state)
    state = monitor_agent_fn(state)
    state = analysis_agent_fn(state)
    state = actions_agent_fn(state)

    print("[main] Fallback pipeline complete.")
    print("[main] Outputs written to output/*.csv")
    return state

def run_crewai_pipeline():
    print("[main] Running CrewAI pipeline...")

    crew = Crew(
        agents=[
            data_ingest_agent,
            monitor_agent,
            analysis_agent,
            actions_agent,
        ]
    )

    result = crew.run({"raw_path": "data/raw/kpi_data.csv"})
    print("[main] CrewAI pipeline complete.")
    print("[main] Outputs written to output/*.csv")
    return result



@app.command()
def pipeline():
    """
    🚀 Run full KPI pipeline:
    1. Ingestion
    2. Alert Detection
    3. Causal Analysis
    4. Action Recommendations
    """

    if USE_CREW and data_ingest_agent:
        run_crewai_pipeline()
    else:
        run_fallback_pipeline()


@app.command()
def ingest():
    """📥 Run only data ingestion."""
    state = {"raw_path": "data/raw/kpi_data.csv"}
    print(ingest_agent_fn(state))


@app.command()
def detect():
    """🚨 Run only KPI deviation detection."""
    state = {"processed_path": "data/processed/kpi.parquet"}
    print(monitor_agent_fn(state))


@app.command()
def analyze():
    """🧠 Run only causal analysis."""
    state = {"processed_path": "data/processed/kpi.parquet", "alerts": []}
    print(analysis_agent_fn(state))


@app.command()
def recommend():
    """🛠 Run only action recommendation generation."""
    state = {"causes": []}
    print(actions_agent_fn(state))


@app.command()
def chat():
    """
    💬 Conversational Querying
    Starts the Streamlit Chatbot-style conversational UI.
    """
    print("Starting chatbot...")
    subprocess.run(["uv", "run", "chatbot_cli.py"])


@app.command()
def dashboard():
    """
    📊 Launch KPI Intelligence Dashboard (Streamlit).
    """
    print("Launching dashboard...")
    subprocess.run(["uv", "run", "streamlit", "run", "dashboard.py"])


@app.command()
def info():
    """ℹ Display system info."""
    typer.echo("KPI Intelligence System")
    typer.echo(f"CrewAI Available: {USE_CREW}")
    typer.echo("Modes: pipeline | chat | dashboard | ingest | detect | analyze | recommend")


if __name__ == "__main__":
    app()
