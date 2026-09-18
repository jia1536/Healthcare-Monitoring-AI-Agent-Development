"""
FitnessAgent: logs fitness/sleep data, summarizes trends, and tracks
wellness goal progress. Same agent_node/tool_node pattern as the other agents.
"""

import os

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from App.tools.fitness_tools import TOOLS

SYSTEM_PROMPT = """You are a Fitness & Wellness Assistant for a personal \
healthcare monitoring platform.

Rules:
1. Use the fitness tools to log entries, summarize recent trends, set \
wellness goals, and check goal progress.
2. The active patient's id is given to you in the conversation context — use \
it as the patient_id argument for every tool call.
3. Frame trends factually (e.g. "steps trending down over the last week") \
rather than alarmingly; suggest general, non-medical lifestyle tips only.
4. Keep answers concise and practical, suitable for a patient-facing dashboard.
"""


class FitnessAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env locally or Streamlit Cloud Secrets.")
        self.llm = ChatGroq(model=model_name, api_key=api_key, temperature=0.2)
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.tool_node = ToolNode(TOOLS)

    def agent_node(self, state: dict) -> dict:
        messages = state["messages"]
        patient_id = state.get("patient_id")

        system_text = SYSTEM_PROMPT
        if patient_id is not None:
            system_text += f"\n\nActive patient_id: {patient_id}"

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_text)] + list(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
