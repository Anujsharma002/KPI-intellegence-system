# Provides task definitions for CrewAI if you want more structured tasks.
# CrewAI might accept Task objects; we include conceptual tasks.

try:
    from crewai import Task
except Exception:
    Task = None

ingest_task = {
    "name": "ingest",
    "description": "Load csv and save parquet"
}
monitor_task = {
    "name": "monitor",
    "description": "Detect alerts"
}
analysis_task = {
    "name": "analysis",
    "description": "Analyze causes for alerts"
}
actions_task = {
    "name": "actions",
    "description": "Generate recommendations"
}
