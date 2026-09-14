from typing import TypedDict, List, Optional, Dict, Any


class GraphState(TypedDict):
    """Shared state for the CLO Waterfall LangGraph workflow."""
    pdf_path: str
    extracted_text_chunks: List[str]
    vector_store_id: Optional[str]
    parsed_waterfall: Optional[Dict[str, Any]]

    # Live capital structure. Starts as a copy of parsed_waterfall['tranches']
    # and is overwritten by the Critic's zero-sum resize on every rejected
    # iteration. This is what was MISSING in the original design -- without
    # it, nothing tracks how the structure actually evolved across the loop.
    current_tranches: Optional[List[Dict[str, Any]]]

    # Monte Carlo assumptions as EXPLICIT state -- read by the Quant prompt,
    # never invented inline by the LLM, and reusable by the stress-test node.
    current_default_rate: float
    default_correlation: float
    base_recovery_rate: float

    generated_code: Optional[str]
    simulation_results: Optional[Dict[str, Any]]
    execution_error: Optional[str]

    critic_feedback: Optional[str]
    iteration_count: int      # structural (Critic) loop counter
    code_retry_count: int     # Quant syntax/runtime retry counter (separate budget)
    max_iterations: int
    status: str

    # Grounded record of every structural iteration -- the actual data source
    # for the Reporter's "Delta" and "Iteration-by-Iteration" sections.
    iteration_history: List[Dict[str, Any]]

    # Base/Mild/Severe/Extreme results for the FINAL, approved structure.
    macro_scenario_results: Optional[Dict[str, Any]]

    final_report: Optional[str]
    parser_data_quality: Optional[Dict[str, Any]]
