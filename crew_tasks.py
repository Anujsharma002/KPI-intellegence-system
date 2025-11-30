
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
