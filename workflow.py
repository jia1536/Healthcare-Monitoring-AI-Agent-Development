import os
from typing import TypedDict, Annotated, Literal

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from App.agents.medical_info_agent import MedicalInfoAgent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    health_report: str


def route_medical_info(state: AgentState) -> Literal["medical_tools", "__end__"]:
    """Decide whether the last LLM message requested a tool call, or is a
    final answer ready to be saved to state."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "medical_tools"

    state["health_report"] = last_message.content
    print(f"\n\nHEALTH REPORT: {state['health_report']}\n\n")
    return "__end__"


class ExecuteWorkflow:
    """Track B, Week 3-4 checkpoint: single-agent workflow.

    Currently wires up ONE agent (MedicalInfoAgent) end-to-end with its RAG
    tool. Additional agents (medication tracking, fitness data, etc.) will be
    added as their own nodes once this single-agent path is stable, per the
    project's "single agent before workflows" requirement.
    """

    def __init__(self):
        medical_agent = MedicalInfoAgent()

        workflow_builder = StateGraph(AgentState)

        workflow_builder.add_node("medical_info", medical_agent.agent_node)
        workflow_builder.add_node("medical_tools", medical_agent.tool_node)

        workflow_builder.add_edge(START, "medical_info")
        workflow_builder.add_conditional_edges(
            "medical_info",
            route_medical_info,
            {
                "medical_tools": "medical_tools",
                "__end__": END,
            },
        )
        workflow_builder.add_edge("medical_tools", "medical_info")

        self.workflow = workflow_builder.compile()

    def run_workflow(self, user_query: str) -> dict:
        from langchain_core.messages import HumanMessage

        input_state = {"messages": [HumanMessage(content=user_query)]}
        result = self.workflow.invoke(input_state)
        return result


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("\n\n--------------------------OUTPUT STARTS HERE--------------\n\n")

    workflow = ExecuteWorkflow()
    output = workflow.run_workflow("What are common symptoms of high blood pressure?")

    print(output["messages"][-1].content)

    print("\n\n--------------------------OUTPUT ENDS HERE-----------------\n\n")
