"""
Stress-Test Agent (NEW): This is the piece the original codebase was missing
entirely. The original Reporter agent was asked to write a "Macroeconomic
Scenario Analysis" section, but nothing in the graph ever actually simulated
more than one scenario -- the LLM was inventing that whole section from
nothing.

This node runs the FINAL, rating-agency-approved simulation code across a
suite of macro scenarios (Base / Mild / Severe / Extreme recession) by
substituting the ANNUAL_DEFAULT_PROB / RECOVERY_RATE constants the Quant
agent was required to define, and re-executing the exact same approved code
path for each one. That keeps the stress test faithful to the precise
waterfall logic that passed the rating check, instead of asking the LLM to
write four separate scripts (and risking four separate sets of bugs).
"""
import re
import logging
from typing import Dict, Any
from src.state import GraphState
from src.tools.python_repl import execute_simulation_code

logger = logging.getLogger(__name__)

# (annual_default_prob, recovery_rate) per scenario -- illustrative, not
# calibrated to any specific asset class. Tune these to match your use case.
MACRO_SCENARIOS = {
    "base_case":        (0.02, 0.70),
    "mild_recession":   (0.05, 0.60),
    "severe_recession": (0.10, 0.40),
    "extreme_stress":   (0.20, 0.25),
}

_CONST_RE = re.compile(r"^\s*(ANNUAL_DEFAULT_PROB|RECOVERY_RATE)\s*=\s*[\d.eE+-]+\s*$", re.MULTILINE)


def _substitute_constants(code: str, default_rate: float, recovery_rate: float) -> str:
    values = {"ANNUAL_DEFAULT_PROB": default_rate, "RECOVERY_RATE": recovery_rate}

    def _replace(match):
        name = match.group(1)
        return f"{name} = {values[name]}"

    return _CONST_RE.sub(_replace, code)


def stress_test_agent(state: GraphState) -> Dict[str, Any]:
    code = state.get("generated_code")
    if not code:
        return {"macro_scenario_results": {}}

    scenario_results = {}
    for scenario_name, (default_rate, recovery_rate) in MACRO_SCENARIOS.items():
        scenario_code = _substitute_constants(code, default_rate, recovery_rate)
        success, results, error_msg = execute_simulation_code(scenario_code)
        scenario_results[scenario_name] = {
            "assumptions": {"annual_default_prob": default_rate, "recovery_rate": recovery_rate},
            "success": success,
            "results": results if success else None,
            "error": None if success else error_msg,
        }
        if not success:
            logger.warning(f"[StressTest] Scenario '{scenario_name}' failed: {error_msg}")

    return {"macro_scenario_results": scenario_results}
