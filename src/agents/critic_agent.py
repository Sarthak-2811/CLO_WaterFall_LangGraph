"""
Critic Agent: Evaluates simulation metrics against rating-agency loss
constraints for EVERY rated tranche, not just the senior-most one -- that was
the original design's biggest gap; a Class B or Class C tranche could be
badly underwater and the graph would still approve the deal as long as the
AAA note happened to be fine.

When a structure fails, the Critic now returns a STRUCTURED, validated
resize (see src/schemas/adjustments.py) instead of only prose. Zero-sum is
enforced deterministically in code rather than trusted from the LLM's
arithmetic.
"""
import json
import logging
from typing import Dict, Any, List, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState
from src.schemas.adjustments import StructuralAdjustment
from src.config.rating_thresholds import threshold_for_tranche

logger = logging.getLogger(__name__)

ZERO_SUM_TOLERANCE = 1.0  # dollars


def _current_tranches(state: GraphState) -> List[dict]:
    return state.get("current_tranches") or (state.get("parsed_waterfall") or {}).get("tranches", [])


def _evaluate_tranches(tranches: List[dict], results: Dict[str, float]) -> List[dict]:
    breaches = []
    for t in tranches:
        threshold, rating, inferred = threshold_for_tranche(t)
        if threshold is None:
            continue  # equity/unrated: no hard cap, it's meant to absorb loss
        loss_prob = results.get(t["class_name"])
        if loss_prob is None:
            continue
        if loss_prob > threshold:
            breaches.append({
                "class_name": t["class_name"],
                "rating": rating,
                "rating_inferred": inferred,
                "observed_loss_prob": loss_prob,
                "threshold": threshold,
            })
    return breaches


def _apply_zero_sum_resize(tranches: List[dict], adjustment: StructuralAdjustment, total_target_par) -> Tuple[List[dict], str]:
    """Applies the Critic's proposed resize, then force-corrects any rounding
    drift by distributing it proportionally across the tranches that were
    actually touched, so the structure always sums to total_target_par
    exactly regardless of the LLM's arithmetic."""
    by_name = {t["class_name"]: dict(t) for t in tranches}
    touched = []
    for adj in adjustment.tranche_adjustments:
        if adj.class_name in by_name:
            by_name[adj.class_name]["principal_amount"] = adj.new_principal_amount
            touched.append(adj.class_name)

    correction_note = ""
    if total_target_par and touched:
        current_total = sum((t.get("principal_amount") or 0) for t in by_name.values())
        drift = total_target_par - current_total
        if abs(drift) > ZERO_SUM_TOLERANCE:
            share = drift / len(touched)
            for name in touched:
                by_name[name]["principal_amount"] = max(0.0, (by_name[name].get("principal_amount") or 0) + share)
            correction_note = (
                f" [Auto-corrected a ${drift:,.0f} zero-sum drift in the Critic's proposal "
                f"by redistributing across {', '.join(touched)} so the total stays fixed.]"
            )

    return list(by_name.values()), correction_note


def critic_agent(state: GraphState) -> Dict[str, Any]:
    results = state.get("simulation_results")
    rules = state.get("parsed_waterfall")
    iteration = state.get("iteration_count", 1)

    if not results or not rules:
        return {"status": "failed", "execution_error": "Missing simulation results or indenture rules in Critic node."}

    tranches = _current_tranches(state)
    if not tranches:
        return {"status": "failed", "execution_error": "No tranches found in parsed waterfall rules."}

    breaches = _evaluate_tranches(tranches, results)
    history_entry = {"iteration": iteration, "tranches_snapshot": tranches, "simulation_results": results, "breaches": breaches}

    if not breaches:
        summary = "; ".join(f"{t['class_name']}: {results.get(t['class_name'], 0):.4%}" for t in tranches if t["class_name"] in results)
        feedback = f"Approved: all rated tranches are within their loss thresholds. {summary}"
        history_entry.update(status="approved", critic_feedback=feedback)
        return {
            "status": "approved",
            "critic_feedback": feedback,
            "current_tranches": tranches,
            "iteration_history": state.get("iteration_history", []) + [history_entry],
        }

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    structured_llm = llm.with_structured_output(StructuralAdjustment)

    system_prompt = """You are the Lead Structuring Quant and Credit Evaluator for a major investment bank, stress-testing a CLO structure against rating agency criteria.

CRITICAL RULES:
1. NO HARDCODED ASSUMPTIONS: only work with the tranche sizes, ratings, and triggers given to you.
2. ZERO-SUM: your tranche_adjustments must keep the total capital structure equal to the Total Target Par. If you shrink a tranche, grow another (typically the Equity/Subordinated tranche) by the same amount.
3. Prefer dynamic cash-flow mechanics over pure resizing: an interest-diversion/cash-sweep that halts Equity distributions and pays down Senior principal when OC/IC tests breach is usually a better fix than just shrinking Senior.
4. Only include tranches you are actually changing in tranche_adjustments.
"""
    total_par = (rules or {}).get("total_target_par")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            f"Total Target Par (fixed): {total_par}\n"
            f"Current Tranches:\n{json.dumps(tranches, indent=2)}\n\n"
            f"Rating Threshold Breaches:\n{json.dumps(breaches, indent=2)}\n\n"
            f"Full Simulation Results: {json.dumps(results)}\n"
            f"Iteration: {iteration}\n\n"
            "Propose the structural adjustment."
        ))
    ])

    try:
        adjustment: StructuralAdjustment = (prompt | structured_llm).invoke({})
    except Exception as e:
        logger.error(f"[Critic] Structured output failed: {e}")
        feedback = f"Critic LLM failed to produce a structured adjustment ({e}). Retrying next iteration with structure unchanged."
        history_entry.update(status="rejected", critic_feedback=feedback)
        return {
            "status": "rejected",
            "critic_feedback": feedback,
            "current_tranches": tranches,
            "iteration_history": state.get("iteration_history", []) + [history_entry],
        }

    new_tranches, correction_note = _apply_zero_sum_resize(tranches, adjustment, total_par)

    feedback = (
        f"Rejected: {len(breaches)} tranche(s) breached their rating threshold "
        f"({', '.join(b['class_name'] for b in breaches)}). Strategy: {adjustment.strategy}. "
        f"{adjustment.rationale} Cash-sweep instructions for the Quant: {adjustment.cash_sweep_instructions}"
        f"{correction_note}"
    )
    history_entry.update(status="rejected", critic_feedback=feedback, proposed_tranches=new_tranches)

    return {
        "status": "rejected",
        "critic_feedback": feedback,
        "current_tranches": new_tranches,
        "iteration_history": state.get("iteration_history", []) + [history_entry],
    }
