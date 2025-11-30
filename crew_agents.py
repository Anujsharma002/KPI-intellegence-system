from src.ingest import ingest
from src.monitor import detect_alerts
from src.analysis import analyze_causes
from src.actions import generate_recommendations


def ingest_agent_fn(state):
    result = ingest(state.get("raw_path", "data/raw/kpi_data.csv"))
    state.update(result)
    return state

def monitor_agent_fn(state):
    processed_path = state.get("processed_path", "data/processed/kpi.parquet")
    result = detect_alerts(processed_path)
    state.update(result)
    return state

def analysis_agent_fn(state):
    processed_path = state.get("processed_path", "data/processed/kpi.parquet")
    alerts = state.get("alerts", [])
    result = analyze_causes(processed_path, alerts)
    state.update(result)
    return state

def actions_agent_fn(state):
    causes = state.get("causes", [])
    result = generate_recommendations({"causes": causes})
    state.update(result)
    return state



try:
    from crewai import Agent, LLM

    llm = LLM(
        provider="ollama",
        model="mistral",
        base_url="http://localhost:11434"
    )

    data_ingest_agent = Agent(
        role="Ingestion Agent",
        goal="Load and validate the KPI CSV and produce parquet.",
        backstory="You load the KPI file and prepare it for analysis.",
        llm=llm,
        verbose=True,
        functions=[ingest_agent_fn],
    )

    monitor_agent = Agent(
        role="Monitoring Agent",
        goal="Detect deviations in KPIs using rolling windows.",
        backstory="You analyze KPI values and detect anomalies.",
        llm=llm,
        verbose=True,
        functions=[monitor_agent_fn],
    )

    analysis_agent = Agent(
        role="Causal Analysis Agent",
        goal="Analyze root causes of KPI deviations.",
        backstory="You compare deltas and feature importance.",
        llm=llm,
        verbose=True,
        functions=[analysis_agent_fn],
    )

    actions_agent = Agent(
        role="Action Recommendation Agent",
        goal="Generate prioritized business recommendations.",
        backstory="You translate causes into recommendations.",
        llm=llm,
        verbose=True,
        functions=[actions_agent_fn],
    )

except Exception as e:
    print("[crew_agents] CrewAI unavailable, falling back.", e)
    data_ingest_agent = None
    monitor_agent = None
    analysis_agent = None
    actions_agent = None
