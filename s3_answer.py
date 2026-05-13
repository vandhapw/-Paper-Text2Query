"""
Stage 3 – Answer Generation
Passes the retrieved data context + original question to an LLM
to generate a coherent natural language answer.
"""

from __future__ import annotations
from langchain_ollama import ChatOllama
from s1_decomposer import DecomposedQuery

# ── Time Scope Description ─────────────────────────────────────────────────────

def _time_description(dq: DecomposedQuery) -> str:
    label_map = {
        "today":      "data from today",
        "yesterday":  "data from yesterday",
        "this_week":  "data from this week",
        "last_week":  "data from last week",
        "this_month": "data from this month",
        "last_hour":  "data from the last hour",
        "last_24h":   "data from the last 24 hours",
        "last_7d":    "data from the last 7 days",
        "last_30d":   "data from the last 30 days",
        "global":     "ALL historical data in the database (full collection)",
    }
    return label_map.get(dq.time_type, "recent data")


# ── System Prompt Builder ──────────────────────────────────────────────────────

def _build_system_prompt(dq: DecomposedQuery, context: str) -> str:
    scope = _time_description(dq)
    collections_hint = ", ".join(dq.collections)
    fields_hint      = ", ".join(dq.fields) if dq.fields else "various sensor fields"
    operation_hint   = dq.operation

    return f"""You are a helpful IoT sensor data assistant.
You have been given retrieved sensor data from MongoDB collections: {collections_hint}.
The data scope is: {scope}.
The operation performed was: {operation_hint} on fields: {fields_hint}.

Rules:
- Answer ONLY based on the data context provided below. Do not hallucinate values.
- If the context says "(no data found)", state clearly that no data is available for the requested period.
- Present numerical values with their units where known:
    temperature → °C, humidity → %, co2 → ppm, dust → µg/m³,
    wind_kph → km/h, precip_mm → mm, pm2_5/pm10 → µg/m³
- Format timestamps in a human-readable way (e.g., "March 11, 2026 at 14:30 UTC").
- If multiple collections are present, clearly label which reading belongs to which location
  (Klaen = indoor unit, Company = office sensor, WeatherAPI = outdoor/weather).
- Be concise, factual, and helpful.

=== RETRIEVED DATA CONTEXT ===
{context}
==============================
"""


# ── Main Stage 3 Entry Point ───────────────────────────────────────────────────

def run_stage3(
    question: str,
    context:  str,
    dq:       DecomposedQuery,
    ollama_llm: ChatOllama,
) -> str:
    """
    Generate a natural language answer from the retrieved data context.

    Args:
        question   : Original user question
        context    : Formatted data string from Stage 2
        dq         : DecomposedQuery from Stage 1 (for metadata)
        ollama_llm : Configured ChatOllama instance

    Returns:
        Natural language answer string
    """
    system_prompt = _build_system_prompt(dq, context)

    messages = [
        ("system", system_prompt),
        ("human",  question),
    ]

    try:
        response = ollama_llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"❌ Answer generation failed: {e}\n\nRaw data context:\n{context}"
