import os
from typing import TypedDict, Annotated, Literal, Optional

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from App.agents.medical_info_agent import MedicalInfoAgent
from App.agents.medication_agent import MedicationAgent
from App.agents.fitness_agent import FitnessAgent
from App.agents.supervisor import supervisor_node
from App.tools.analytics_tools import detect_health_risks, build_health_report
from App.db import database as db


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: Optional[int]
    route: Optional[str]
    health_report: str
    alerts_raised: list


def _make_router():
    """Shared route_* function for every domain agent's tool-call check —
    each agent's response is inspected identically, so one closure is reused
    across all three conditional edges instead of duplicating the branch."""

    def _route(state: AgentState) -> Literal["tools", "finalize"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "finalize"

    return _route


def route_from_supervisor(state: AgentState) -> Literal["medical_info", "medication", "fitness"]:
    return state.get("route", "medical_info")


def finalize_node(state: AgentState) -> dict:
    """Shared exit node for every domain agent: runs risk detection and
    builds/saves the aggregated health report. Skips analytics gracefully
    when there is no patient_id (e.g. a one-off medical-info question)."""
    last_message = state["messages"][-1]
    latest_response = getattr(last_message, "content", "") or ""
    patient_id = state.get("patient_id")

    alerts_raised = []
    report = latest_response
    if patient_id is not None:
        alerts_raised = detect_health_risks(patient_id)
        report = build_health_report(patient_id, latest_response)

    return {"health_report": report, "alerts_raised": alerts_raised}


class ExecuteWorkflow:
    """Track B multi-agent workflow.

    START -> supervisor (keyword-based intent routing)
          -> one of: medical_info | medication | fitness
               (each loops through its own ToolNode until it has a final answer)
          -> finalize (risk detection + health report generation, shared)
          -> END
    """

    def __init__(self):
        db.init_db()

        medical_agent = MedicalInfoAgent()
        medication_agent = MedicationAgent()
        fitness_agent = FitnessAgent()

        builder = StateGraph(AgentState)

        builder.add_node("supervisor", supervisor_node)

        builder.add_node("medical_info", medical_agent.agent_node)
        builder.add_node("medical_tools", medical_agent.tool_node)

        builder.add_node("medication", medication_agent.agent_node)
        builder.add_node("medication_tools", medication_agent.tool_node)

        builder.add_node("fitness", fitness_agent.agent_node)
        builder.add_node("fitness_tools", fitness_agent.tool_node)

        builder.add_node("finalize", finalize_node)

        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "medical_info": "medical_info",
                "medication": "medication",
                "fitness": "fitness",
            },
        )

        route = _make_router()

        builder.add_conditional_edges(
            "medical_info", route, {"tools": "medical_tools", "finalize": "finalize"},
        )
        builder.add_edge("medical_tools", "medical_info")

        builder.add_conditional_edges(
            "medication", route, {"tools": "medication_tools", "finalize": "finalize"},
        )
        builder.add_edge("medication_tools", "medication")

        builder.add_conditional_edges(
            "fitness", route, {"tools": "fitness_tools", "finalize": "finalize"},
        )
        builder.add_edge("fitness_tools", "fitness")

        builder.add_edge("finalize", END)

        self.workflow = builder.compile()

    def run_workflow(self, user_query: str, patient_id: Optional[int] = None) -> dict:
        from langchain_core.messages import HumanMessage

        input_state = {
            "messages": [HumanMessage(content=user_query)],
            "patient_id": patient_id,
        }
        return self.workflow.invoke(input_state)


if __name__ == "__main__":
    print("\n\n--------------------------OUTPUT STARTS HERE--------------\n\n")

    workflow = ExecuteWorkflow()

    # Demo patient so the medication/fitness branches have something to work with
    patients = db.list_patients()
    demo_patient_id = patients[0]["id"] if patients else db.create_patient("Demo Patient", 35, "Other")

    for query in [
        "What are common symptoms of high blood pressure?",
        "Did I take all my medications this week?",
        "How are my step counts trending lately?",
    ]:
        print(f"\n>>> {query}")
        output = workflow.run_workflow(query, patient_id=demo_patient_id)
        print(output["messages"][-1].content)

    print("\n\n--------------------------OUTPUT ENDS HERE-----------------\n\n")
