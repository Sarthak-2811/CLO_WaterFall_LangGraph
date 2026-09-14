"""
Chatbot Agent -- Interactive AI Assistant for CLO Workflows.

CHANGE from the original: `get_workflow_analysis` used to rebuild the entire
LangGraph app (and open a fresh sqlite3 connection) on EVERY tool call. It's
now built once and cached at module import time. It also now surfaces the
new grounded fields (current_tranches, iteration_history,
macro_scenario_results) instead of just parsed_waterfall/simulation_results.
"""

import os
import json
import sqlite3
from typing import Optional

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.tools.retriever import search_indenture, get_or_create_retriever

_ANALYSIS_APP_CACHE: dict = {}


def _get_analysis_app():
    if "app" not in _ANALYSIS_APP_CACHE:
        from src.main import build_clo_graph
        conn = sqlite3.connect("data/workflow_threads.db", check_same_thread=False)
        cp = SqliteSaver(conn)
        _ANALYSIS_APP_CACHE["app"] = build_clo_graph(cp)
    return _ANALYSIS_APP_CACHE["app"]


@tool
def search_document(query: str, pdf_path: str) -> str:
    """Search the currently active CLO Indenture PDF for specific keywords or clauses.
    Use this to answer questions about the legal document itself."""
    try:
        retriever = get_or_create_retriever(pdf_path, k=4)
        results = search_indenture(retriever, query)
        return results or "No matching clauses found in the document."
    except Exception as e:
        return f"Error searching document: {str(e)}"


@tool
def get_workflow_analysis(thread_id: str) -> str:
    """Get the current tranche structure, structural iteration history, base-case
    Monte Carlo results, and macro scenario stress-test results for the current
    deal. Use this to answer questions about loss probabilities, tranche sizes,
    how the structure changed across iterations, or recession scenarios."""
    try:
        app = _get_analysis_app()
        config = {"configurable": {"thread_id": thread_id}}
        snap = app.get_state(config)
        if not snap or not snap.values:
            return "No data found for this workflow thread."

        data = {
            "current_tranches": snap.values.get("current_tranches"),
            "simulation_results": snap.values.get("simulation_results"),
            "iteration_history": snap.values.get("iteration_history"),
            "macro_scenario_results": snap.values.get("macro_scenario_results"),
            "status": snap.values.get("status"),
        }
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching workflow analysis: {str(e)}"


def build_chatbot_graph(checkpointer: Optional[SqliteSaver] = None):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, max_tokens=2048)
    tools = [search_document, get_workflow_analysis]

    system_prompt = (
        "You are an expert AI CLO (Collateralized Loan Obligation) Assistant. "
        "Your role is to help users understand the structure, tranches, rules, and simulation results "
        "of the active CLO workflow.\n\n"
        "You have access to:\n"
        "1. `get_workflow_analysis`: call this when asked about simulation results, loss probabilities, "
        "how the structure changed across iterations, or recession/stress-scenario behavior. You MUST pass "
        "the `thread_id` from the System Context.\n"
        "2. `search_document`: call this for specific legal clauses or raw indenture text. You MUST pass the "
        "exact `pdf_path` from the System Context.\n\n"
        "Combine both when useful. Be concise, professional, and explain complex financial concepts clearly."
    )

    return create_react_agent(model=llm, tools=tools, prompt=system_prompt, checkpointer=checkpointer)
