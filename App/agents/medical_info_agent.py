"""
MedicalInfoAgent: the first single agent for the healthcare monitoring
platform (Track B). Answers general health-information questions by
retrieving from a local RAG knowledge base before responding.

This mirrors the CareerAssessmentAgent pattern:
- agent_node: invokes the LLM (with tools bound) on the current message state
- tool_node: executes any tool calls the LLM requested and appends results
"""

import os

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from App.tools.medical_tools import TOOLS

SYSTEM_PROMPT = """You are a Medical Information Assistant for a personal \
healthcare monitoring platform.

Rules:
1. Always use the search_health_documents tool to ground your answer in the \
local knowledge base before responding to a health question.
2. Never provide a diagnosis. Present general information only.
3. Always include a brief disclaimer that this is not a substitute for \
professional medical advice.
4. If the local documents don't cover the question, say so honestly instead \
of inventing information.
5. Keep answers clear and concise, suitable for a patient-facing dashboard.
"""


class MedicalInfoAgent:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model=model_name, api_key=api_key, temperature=0.2)
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.tool_node = ToolNode(TOOLS)

    def agent_node(self, state: dict) -> dict:
        """Invoke the LLM (with the system prompt prepended) on current messages."""
        messages = state["messages"]

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
