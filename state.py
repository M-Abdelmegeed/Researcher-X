from typing import TypedDict, List


class ResearchState(TypedDict):
    """State definition for LangGraph workflow."""
    query: str
    retrieved_docs: list
    response: str
    classification: str
    chat_history: List[str]
    chat_summary: str
    session_id: str
    generated_queries: List[str]
    reranked_docs: str
    research_result: str