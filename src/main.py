"""
Main entry point for the CLO Waterfall Simulator LangGraph workflow.

Graph shape (new stress_test_node added vs. the original):

  START -> parser_node -> quant_node <-> quant_node (code-error retries)
                              |
                              v
                          critic_node --(rejected, budget left)--> quant_node
                              |
                     (approved)|  (exhausted budget)
                              v                 \
                      stress_test_node       reporter_node
                              |                 /
                              v                /
                         reporter_node <------
                              |
                              v
                             END
"""

import os
import sys
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from src.state import GraphState
from src.agents.parser_agent import parser_agent
from src.agents.quant_agent import quant_agent
from src.agents.critic_agent import critic_agent
from src.agents.stress_test_agent import stress_test_agent
from src.agents.reporter_agent import reporter_agent

load_dotenv()

MAX_ITERATIONS = 5  # structural (Critic) loop budget -- separate from the Quant's own code-retry budget


def route_after_parser(state: GraphState) -> str:
    if state.get("status") == "failed":
        return END
    return "quant_node"


def route_after_quant(state: GraphState) -> str:
    status = state.get("status")
    if status == "quant_error":
        print(f"[!] Code error on structural iteration {state.get('iteration_count', 0)}, "
              f"retry {state.get('code_retry_count', 0)}. Retrying Quant...")
        return "quant_node"
    if status == "failed":
        print("[!] Quant exceeded its code-retry budget. Terminating graph.")
        return END
    return "critic_node"


def route_after_critic(state: GraphState) -> str:
    status = state.get("status")
    iteration = state.get("iteration_count", 0)

    if status == "approved":
        print(f"[OK] Model approved by rating-agency Critic at structural iteration {iteration}. Running macro stress tests...")
        return "stress_test_node"

    if iteration >= MAX_ITERATIONS:
        print("[!] Convergence limit reached without approval. Skipping stress test, writing report as-is.")
        return "reporter_node"

    print(f"[->] Model rejected. Routing back to Quant with structural feedback (loop {iteration + 1})...")
    return "quant_node"


def build_clo_graph(checkpointer=None):
    workflow = StateGraph(GraphState)

    workflow.add_node("parser_node", parser_agent)
    workflow.add_node("quant_node", quant_agent)
    workflow.add_node("critic_node", critic_agent)
    workflow.add_node("stress_test_node", stress_test_agent)
    workflow.add_node("reporter_node", reporter_agent)

    workflow.add_edge(START, "parser_node")
    workflow.add_conditional_edges("parser_node", route_after_parser)
    workflow.add_conditional_edges("quant_node", route_after_quant)
    workflow.add_conditional_edges("critic_node", route_after_critic)
    workflow.add_edge("stress_test_node", "reporter_node")
    workflow.add_edge("reporter_node", END)

    return workflow.compile(checkpointer=checkpointer)


def make_initial_state(pdf_path: str, default_rate: float = 0.02, recovery_rate: float = 0.70, correlation: float = 0.20) -> GraphState:
    """Single source of truth for a fresh run's initial state -- both src/main.py's
    CLI entrypoint and app.py's Streamlit runner call this, so the two can never
    drift out of sync with GraphState's schema (the original bug: app.py hand-rolled
    its own copy of this dict)."""
    return {
        "pdf_path": pdf_path,
        "extracted_text_chunks": [],
        "vector_store_id": None,
        "parsed_waterfall": None,
        "current_tranches": None,
        "current_default_rate": default_rate,
        "default_correlation": correlation,
        "base_recovery_rate": recovery_rate,
        "generated_code": None,
        "simulation_results": None,
        "execution_error": None,
        "critic_feedback": None,
        "iteration_count": 0,
        "code_retry_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "status": "pending",
        "iteration_history": [],
        "macro_scenario_results": None,
        "final_report": None,
        "parser_data_quality": None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.main <path_to_indenture_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    if not os.path.exists(pdf_file):
        print(f"Error: File not found at {pdf_file}")
        sys.exit(1)

    app = build_clo_graph()
    print(f"Starting CLO Waterfall Analysis on {pdf_file}...")

    final_state = app.invoke(make_initial_state(pdf_file))

    print("\n--- Final Workflow Results ---")
    print("Status:", final_state.get("status"))
    print("Structural Iterations:", final_state.get("iteration_count"), "/", final_state.get("max_iterations"))
    print("Final Tranches:", json.dumps(final_state.get("current_tranches"), indent=2))
    print("Base-Case Loss Probabilities:", final_state.get("simulation_results"))
    print("\n--- Macro Scenario Results ---")
    print(json.dumps(final_state.get("macro_scenario_results"), indent=2))
    print("\n--- Executive Summary ---")
    print(final_state.get("final_report"))
