"""
Structured output schema for the Critic agent.

Replaces free-text-only feedback (the original design) with a validated
payload the rest of the graph can consume programmatically: exact tranche
resizes, exact trigger tightening, and separate qualitative instructions for
the dynamic cash-sweep mechanic the Quant has to encode in Python.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


class TrancheAdjustment(BaseModel):
    class_name: str = Field(description="Exact class_name of the tranche being resized; must match an existing tranche")
    new_principal_amount: float = Field(description="The new principal balance in USD for this tranche")


class TriggerAdjustment(BaseModel):
    applies_to_class: str = Field(description="Which class/group this coverage test applies to; must match an existing coverage test")
    new_trigger_ratio: float = Field(description="The tightened trigger ratio as a percentage, e.g. 125.0")


class StructuralAdjustment(BaseModel):
    """The Critic's prescribed fix when a structure fails one or more rating thresholds."""
    strategy: Literal["interest_diversion_cash_sweep", "senior_amortization", "resize_only"] = Field(
        description="Which dynamic structuring strategy the Quant must implement in code"
    )
    tranche_adjustments: List[TrancheAdjustment] = Field(
        default_factory=list,
        description="Zero-sum resizing of tranche principal amounts. Omit tranches that are unchanged."
    )
    trigger_adjustments: List[TriggerAdjustment] = Field(
        default_factory=list,
        description="Tightened OC/IC trigger ratios, if any."
    )
    cash_sweep_instructions: str = Field(
        description="Precise, step-by-step instructions for the Quant to encode the cash-sweep/diversion mechanic in the Monte Carlo script."
    )
    rationale: str = Field(description="One or two sentence explanation of why this fix should work.")
