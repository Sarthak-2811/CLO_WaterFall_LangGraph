"""
Reporter Agent: Generates the executive summary from GROUNDED data -- the
actual iteration history the graph produced and the actual macro stress-test
results -- rather than letting the LLM invent an iteration log or scenario
numbers, which is what the original design forced it to do (it only ever
received the latest snapshot, never a history, and never more than one
simulation run).
"""
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState


def reporter_agent(state: GraphState) -> Dict[str, Any]:
    rules = state.get("parsed_waterfall")
    final_tranches = state.get("current_tranches") or (rules or {}).get("tranches", [])
    results = state.get("simulation_results")
    iterations = state.get("iteration_count")
    max_iterations = state.get("max_iterations")
    history = state.get("iteration_history", [])
    macro = state.get("macro_scenario_results") or {}
    converged = state.get("status") == "approved"

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

    system_prompt = """You are a Lead Quantitative Structurer at a top-tier investment bank writing a post-execution report for a multi-agent CLO simulation run.

You are given the ACTUAL iteration log and ACTUAL macro-scenario simulation results as JSON below. Ground every number in that data -- do not invent figures, and do not invent iterations that are not in the log. If a scenario's "success" field is false, say plainly that it could not be simulated rather than making up numbers for it. If the deal never converged (did not reach "approved" status), say so plainly in the Executive Summary and adjust sections 2-5 accordingly (e.g. there may be no approved final structure to stress-test).

Follow this exact structure:

## 1. Executive Summary & Final Verdict
- Deal Status (Approved / Failed to Converge), Total Iterations, final loss probabilities per tranche, one paragraph bottom line.

## 2. The Delta: Initial vs. Final Structure
Markdown table comparing tranche sizes from the FIRST entry in the iteration log vs the LAST entry. Note the single most consequential structural pivot.

## 3. Iteration-by-Iteration Breakdown
For each entry in the iteration log, in order: starting tranche sizes, which tranches breached their rating threshold and by how much, and what the Critic's feedback prescribed.

## 4. Macroeconomic Scenario Analysis
Use the macro_scenario_results JSON. For each scenario report the actual loss probabilities produced and describe how cliff effects / cash-sweep mechanics behaved as defaults rose scenario to scenario.

## 5. Strategic Recommendations & Areas for Improvement
2-3 actionable, data-grounded insights.

Maintain a professional, analytical, objective tone."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "Deal Converged: {converged}\n"
            "Final Tranches:\n{tranches}\n\n"
            "Final Base-Case Simulation Results:\n{results}\n\n"
            "Total Iterations: {iterations} / {max_iterations} allowed\n\n"
            "Full Iteration Log (JSON):\n{history}\n\n"
            "Macro Scenario Results (JSON):\n{macro}\n\n"
            "Write the Executive Summary Report."
        ))
    ])

    chain = prompt | llm
    response = chain.invoke({
        "converged": converged,
        "tranches": json.dumps(final_tranches, indent=2),
        "results": json.dumps(results, indent=2) if results else "{}",
        "iterations": iterations,
        "max_iterations": max_iterations,
        "history": json.dumps(history, indent=2),
        "macro": json.dumps(macro, indent=2),
    })

    return {"final_report": response.content}
