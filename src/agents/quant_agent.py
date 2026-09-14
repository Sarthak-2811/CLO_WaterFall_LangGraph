"""
Quant Agent: Reads the LIVE capital structure (post-Critic resizing, if any)
plus explicit Monte Carlo assumptions from state, and generates + executes a
Python simulation.

CHANGE from the original: `default_correlation` and `base_recovery_rate`
existed in GraphState but were never read -- the prompt hardcoded "2% base
annual default probability... 70% recovery" regardless of what the state
said. That's fixed here: the three assumptions come from state, are forced
into the generated script as named constants, and are what the stress-test
node later substitutes to run real macro scenarios against the SAME approved
code path.
"""
import json
import logging
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState
from src.tools.python_repl import execute_simulation_code

logger = logging.getLogger(__name__)

MAX_CODE_RETRIES = 3


def quant_agent(state: GraphState) -> Dict[str, Any]:
    rules = state.get("parsed_waterfall")
    if not rules:
        return {"status": "failed", "execution_error": "No indenture rules found in state."}

    tranches = state.get("current_tranches") or rules.get("tranches", [])
    if not tranches:
        return {"status": "failed", "execution_error": "No tranches found in parsed waterfall rules. Cannot run simulation."}

    working_rules = {**rules, "tranches": tranches}
    rules_json = json.dumps(working_rules, indent=2)

    critic_feedback = state.get("critic_feedback") or "None. This is the initial run."
    execution_error = state.get("execution_error") or "None."
    iteration = state.get("iteration_count", 0)
    code_retry = state.get("code_retry_count", 0)

    default_rate = state.get("current_default_rate", 0.02)
    recovery_rate = state.get("base_recovery_rate", 0.70)
    correlation = state.get("default_correlation", 0.20)

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)

    system_prompt = """You are an elite Quantitative Developer at a top-tier Investment Bank.
Write a standalone Python script to run a Monte Carlo simulation of a CLO Waterfall.

RULES:
1. Use `numpy` and `pandas`. Simulate 10,000 paths (n_paths=10000).
2. Use the provided JSON capital structure for tranche sizes, triggers, and spreads.
3. Define these three constants EXACTLY as named, using the numbers given below -- do NOT invent your own values, and do not rename them:
   ANNUAL_DEFAULT_PROB = {default_rate}
   RECOVERY_RATE = {recovery_rate}
   DEFAULT_CORRELATION = {correlation}
4. Model correlated defaults with a single-factor Gaussian copula: draw one systemic factor per path and one idiosyncratic factor per obligor, combine with weight sqrt(DEFAULT_CORRELATION) on the systemic factor, and map through the normal CDF (using `norm.cdf` and `norm.ppf` from `scipy.stats`) against ANNUAL_DEFAULT_PROB to get each obligor's default indicator. Do NOT simulate defaults as independent iid draws -- that materially understates tail risk for a rated structure.
5. Calculate the probability of principal loss (dollar loss > 0) for EACH tranche.
6. WATERFALL LOSS ORDER -- losses are absorbed BOTTOM-UP: Equity/Subordinated takes the FIRST losses, then Mezzanine/Junior tranches (e.g. Class C, then Class B), Senior tranches (e.g. Class A-1/AAA) take the LAST losses. Never reverse this.
7. If the JSON includes coverage_tests (OC/IC), implement them: when a test breaches on a given path, halt further Equity distributions on that path for the remainder of its life and redirect that cash to pay down Senior principal instead, per this Critic guidance:
   {critic_feedback}
8. DO NOT make network calls.
9. CRITICAL: You MUST print the final result to stdout as the LAST line via `print(json.dumps(results_dict))`. Keys = tranche class_name, values = probability of loss (float 0..1). If you fail to print a valid JSON object, the system will crash.
10. Do not embed the raw rules JSON string in your code -- extract the numbers you need and define them as plain Python variables.
11. Output ONLY pure Python code. No markdown fences.
12. Include `import json`, `import numpy as np`, `import pandas as pd`, and `from scipy.stats import norm` at the top. You may also use `import math` and `import random` if needed -- no other imports are permitted.
13. NO HARDCODED ASSUMPTIONS beyond the three named constants above: if a trigger/fee/spread is null in the JSON, treat it as 0 / excluded -- never invent a market-standard number for it.
14. DO NOT use `if __name__ == '__main__':` guards. All code runs at the module level inside exec(); place your simulation logic and final print() call at the top level, not inside any guard block.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Capital Structure & Rules:\n{rules}\n\nExecution errors from the previous attempt (fix these if present):\n{errors}\n\nWrite the Python simulation code.")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "rules": rules_json, 
        "errors": execution_error,
        "default_rate": default_rate,
        "recovery_rate": recovery_rate,
        "correlation": correlation,
        "critic_feedback": critic_feedback
    })
    generated_code = response.content.replace("```python", "").replace("```", "").strip()

    success, results, error_msg = execute_simulation_code(generated_code)

    if not success:
        next_retry = code_retry + 1
        if next_retry >= MAX_CODE_RETRIES:
            return {
                "generated_code": generated_code,
                "execution_error": f"Exceeded {MAX_CODE_RETRIES} code-retry attempts on this structural iteration. Last error: {error_msg}",
                "status": "failed",
                "code_retry_count": next_retry,
            }
        return {
            "generated_code": generated_code,
            "execution_error": error_msg,
            "status": "quant_error",
            "code_retry_count": next_retry,
        }

    return {
        "generated_code": generated_code,
        "simulation_results": results,
        "execution_error": None,
        "iteration_count": iteration + 1,
        "code_retry_count": 0,
        "status": "simulated",
    }
