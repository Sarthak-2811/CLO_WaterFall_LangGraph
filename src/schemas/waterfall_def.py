from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Tranche(BaseModel):
    """Represents a single debt or equity tranche in the CLO capital structure."""
    class_name: str = Field(description="Class identifier, e.g., 'Class A-1', 'Class B', 'Subordinated/Equity'")
    target_rating: str = Field(default="", description="Rating agency credit rating, e.g., 'AAA', 'AA', 'BBB', 'NR/Equity'")
    principal_amount: Optional[float] = Field(default=None, description="Initial par or principal balance in USD. Null if not found.")
    coupon_type: Literal["floating", "fixed"] = Field(default="floating")
    spread_bps: Optional[float] = Field(default=0.0, description="Spread over benchmark in bps")
    fixed_rate: Optional[float] = Field(default=None, description="Fixed coupon %, if not floating")
    is_equity: bool = Field(default=False, description="True if this is the residual equity/subordinated tranche")


class CoverageTest(BaseModel):
    test_type: Literal["OC", "IC"] = Field(description="Overcollateralization or Interest Coverage test")
    applies_to_class: str = Field(description="The tranche or combined class tested, e.g. 'Class A/B'")
    trigger_ratio: Optional[float] = Field(default=None, description="Minimum ratio threshold as a percentage")
    cure_action: str = Field(default="Pay down senior principal until test is satisfied")


class FeeStructure(BaseModel):
    senior_admin_fee_cap: Optional[float] = Field(default=None)
    senior_mgmt_fee_rate: Optional[float] = Field(default=None)
    subordinated_mgmt_fee_rate: Optional[float] = Field(default=None)
    incentive_fee_hurdle_irr: Optional[float] = Field(default=None)
    incentive_fee_share: Optional[float] = Field(default=None)


class WaterfallStep(BaseModel):
    priority: int
    payee: str
    payment_type: Literal["fees", "interest", "principal", "oc_cure", "residual_equity"]
    condition: Optional[str] = Field(default=None)


class IndentureRules(BaseModel):
    """Complete parsed financial model extracted from the legal indenture."""
    deal_name: str = Field(default="Unknown Deal")
    total_target_par: Optional[float] = Field(default=None)
    tranches: List[Tranche] = Field(default_factory=list)
    coverage_tests: List[CoverageTest] = Field(default_factory=list)
    fees: FeeStructure = Field(default_factory=FeeStructure)
    ccc_bucket_limit: Optional[float] = Field(default=None)
    interest_waterfall: List[WaterfallStep] = Field(default_factory=list)

    def effective_total_par(self) -> Optional[float]:
        if self.total_target_par:
            return self.total_target_par
        principals = [t.principal_amount for t in self.tranches if t.principal_amount]
        return sum(principals) if principals else None
