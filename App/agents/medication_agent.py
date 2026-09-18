"""
MedicationAgent: tracks medications, logs doses, reports adherence, and
flags basic interaction concerns. Same agent_node/tool_node pattern as
MedicalInfoAgent so it drops into the LangGraph workflow identically.
"""

import os

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from App.tools.medication_tools import TOOLS

SYSTEM_PROMPT = """You are a Medication Tracking Assistant for a personal \
healthcare monitoring platform.

Rules:
1. Use the medication tools to add medications, log doses, check adherence, \
list active medications, and check for basic drug interaction concerns.
2. The active patient's id is given to you in the conversation context — use \
it as the patient_id argument for every tool call.
3. Never tell a patient to stop or change a medication yourself; if an \
interaction concern is found, tell them to confirm with a pharmacist or doctor.
4. Keep answers concise and practical, suitable for a patient-facing dashboard.
"""


class MedicationAgent:
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
