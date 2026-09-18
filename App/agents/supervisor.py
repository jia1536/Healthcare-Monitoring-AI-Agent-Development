"""
Supervisor: routes an incoming user message to the right domain agent
(medical_info / medication / fitness).

Routing is keyword-based rather than an extra LLM call. This is a deliberate
choice for a health app: it's instant, free, fully deterministic/testable
without an API key, and avoids a routing hallucination sending a medication
question to the wrong agent. Ambiguous or general messages default to
medical_info, which is the safest fallback (it just answers from the RAG
knowledge base rather than taking any tracking action).
"""

from typing import Literal

MEDICATION_KEYWORDS = [
    "medication", "medicine", "pill", "dose", "dosage", "prescription",
    "adherence", "interaction", "drug", "refill", "missed dose",
]

FITNESS_KEYWORDS = [
    "steps", "sleep", "fitness", "workout", "exercise", "calorie",
    "heart rate", "goal", "activity", "walk", "run", "weight",
]

Route = Literal["medical_info", "medication", "fitness"]


def classify_intent(text: str) -> Route:
    lowered = text.lower()

    med_hits = sum(1 for kw in MEDICATION_KEYWORDS if kw in lowered)
    fit_hits = sum(1 for kw in FITNESS_KEYWORDS if kw in lowered)

    if med_hits == 0 and fit_hits == 0:
        return "medical_info"
    if med_hits >= fit_hits:
        return "medication"
    return "fitness"


def supervisor_node(state: dict) -> dict:
    """LangGraph node: reads the latest human message and stashes the routing
    decision in state['route'] for the conditional edge to consume."""
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if getattr(m, "type", None) == "human"),
        messages[-1] if messages else None,
    )
    text = getattr(last_human, "content", "") or ""
    route = classify_intent(text)
    return {"route": route}
