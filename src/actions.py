# src/actions.py
"""
Lever 3 — Intelligent Action Agent (Recommendation Engine)

Given the ranked causes from the causal engine, this module:
 - maps each cause to a likely business intervention
 - prioritizes actions by cause severity (score/delta)
 - generates concise business language
 - produces an email-ready recommended action summary
 - optionally uses Ollama LLM for polished tone (controlled by config.mode)
"""

from typing import Dict, List, Any, Optional
from src.utils import load_config
import requests
import json

# Default Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:latest"

# -----------------------------------------------------
# Low-level helper for LLM refinement (optional)
# -----------------------------------------------------
def call_ollama(prompt: str, model: Optional[str] = None, base_url: str = OLLAMA_URL, timeout: int = 180) -> str:
    model_to_use = model or DEFAULT_MODEL
    try:
        payload = {"model": model_to_use, "prompt": prompt, "stream": False}
        r = requests.post(base_url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Prefer "response"
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        # fallback
        return json.dumps(data)
    except Exception as e:
        raise RuntimeError(f"LLM failure: {e}")


# -----------------------------------------------------
# Rule-based mapping from cause → business action
# -----------------------------------------------------
def action_mapping(feature: str, delta: float) -> str:
    """
    Deterministic rule-based action suggestions.
    You can expand this list with company-specific rules.
    """
    feature_l = feature.lower()

    # Sales volume drop
    if "sales" in feature_l:
        return "Increase targeted marketing and promotions to recover lost volume."

    # Discount changed
    if "discount" in feature_l:
        if delta < 0:
            return "Re-evaluate discount strategy; consider temporary increase to boost conversions."
        else:
            return "Review discount margins to ensure profitability while sustaining volume."

    # Rating decline
    if "rating" in feature_l or "review" in feature_l:
        return "Investigate customer complaints; improve product quality or customer service."

    # Trend score or demand index
    if "trend" in feature_l or "demand" in feature_l:
        return "Boost demand via influencer campaigns or social ads to regain visibility."

    # Supply chain or operations impact
    if "supply" in feature_l or "logistic" in feature_l or "inventory" in feature_l:
        return "Optimize supply chain bottlenecks; improve fulfillment speed and inventory planning."

    # Pricing changes
    if "price" in feature_l or "cost" in feature_l:
        return "Re-evaluate pricing strategy to align with competitor benchmarks."

    # Revenue correlated but unknown driver
    return "Investigate operational or market factors impacting this metric."


# -----------------------------------------------------
# Deterministic fallback recommendation block
# -----------------------------------------------------
def deterministic_recommendation(causes: List[Dict[str, Any]], alert_id: Any, date: str) -> str:
    """
    Build an email-ready paragraph + bullets summarizing recommended actions.
    """
    lines = []
    lines.append(f"Subject: Recommended Actions for Alert {alert_id} ({date})\n")
    lines.append("Based on the identified root causes, the following actions are recommended:\n")

    for c in causes:
        feat = c["feature"]
        delta = c.get("delta", 0.0)
        action = action_mapping(feat, delta)
        try:
            dstr = f"{delta:.2%}"
        except Exception:
            dstr = str(delta)
        lines.append(f"• {feat}: {action} (change: {dstr})")

    return "\n".join(lines)


# -----------------------------------------------------
# LLM-based refinement (if mode: llm or silent)
# -----------------------------------------------------
def llm_refine_recommendations(alert_id: Any, date: str, causes: List[Dict[str, Any]],
                               fallback_text: str, model_name: str) -> str:
    """
    Uses LLM to polish and enhance the recommendations. Falls back automatically.
    """
    try:
        cause_lines = []
        for c in causes:
            feat = c["feature"]
            delta = c.get("delta", 0.0)
            try:
                dstr = f"{delta:.2%}"
            except Exception:
                dstr = str(delta)
            cause_lines.append(f"- {feat} changed by {dstr}")

        prompt = f"""
You are a business strategy assistant.

A KPI alert occurred:

Alert ID: {alert_id}
Date: {date}

Key contributing causes:
{chr(10).join(cause_lines)}

Task:
Write a professional, email-ready set of recommended actions.
Requirements:
- Use concise business language
- Provide 3–5 actionable, prioritized bullet points
- No long paragraphs
- Keep bullets to 8–14 words each
- Output only the final email text

Start now.
        """

        return call_ollama(prompt, model=model_name)

    except Exception:
        return fallback_text


# -----------------------------------------------------
# Main entrypoint
# -----------------------------------------------------
def generate_recommendations(causes_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: {"causes": [ { alert_id, date, ranked_causes, explanation } ] }
    Output: {"recommendations": [ { alert_id, date, actions } ] }
    """
    cfg = load_config()
    mode = cfg.get("mode", "llm")
    model_name = cfg.get("ollama_model", DEFAULT_MODEL)

    if "causes" not in causes_payload:
        return {"recommendations": []}

    output = []

    for item in causes_payload["causes"]:
        alert_id = item.get("alert_id")
        date = item.get("date")
        ranked = item.get("ranked_causes", [])

        # Use top 3 drivers for recommendations
        top_drivers = ranked[:3]

        # Deterministic fallback recommendations
        fallback_text = deterministic_recommendation(top_drivers, alert_id, date)

        # Use LLM if allowed
        if mode == "fast":
            final_text = fallback_text
        else:
            final_text = llm_refine_recommendations(alert_id, date, top_drivers, fallback_text, model_name)

        output.append({
            "alert_id": alert_id,
            "date": date,
            "actions": top_drivers,
            "recommendation_text": final_text
        })

    print(f"[actions] produced recommendations for {len(output)} alerts")
    return {"recommendations": output}
