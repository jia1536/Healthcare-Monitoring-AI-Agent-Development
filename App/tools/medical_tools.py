"""
Tools available to the MedicalInfoAgent.
"""

from langchain_core.tools import tool

from App.rag.rag_chain import HealthRAGChain

# Built once at import time so the vectorstore isn't rebuilt on every call
_rag_chain = HealthRAGChain()


@tool
def search_health_documents(query: str) -> str:
    """Search the local health knowledge base for information relevant to a
    user's health question (e.g. symptoms, general condition info, medication
    safety, nutrition basics). Returns the most relevant document excerpts
    with their source file. Always use this before answering a medical
    information question so the response is grounded in the local documents
    rather than invented."""
    return _rag_chain.retrieve_as_context(query)


TOOLS = [search_health_documents]
