# CLO Waterfall Simulator — Agentic AI Structuring Desk

A LangGraph multi-agent pipeline that reads a CLO indenture PDF, writes and
runs a Monte Carlo simulation of the waterfall, rating-checks every tranche,
autonomously resizes the structure until it passes, then stress-tests the
approved deal across recession scenarios and writes a grounded executive
report. Runs entirely on free Groq inference.

## What changed from the original draft

The original codebase had the right shape (Parser → Quant → Critic → loop →
Reporter) but several load-bearing pieces were either dead code or missing
outright:

| Problem | Fix |
|---|---|
| `default_correlation` / `base_recovery_rate` existed in state but the Quant prompt hardcoded 2%/70% anyway | Quant now reads `current_default_rate`, `base_recovery_rate`, `default_correlation` from state and forces them into the generated script as named constants (`ANNUAL_DEFAULT_PROB`, `RECOVERY_RATE`, `DEFAULT_CORRELATION`) |
| Critic only ever checked the single senior tranche against one hardcoded 0.01% threshold | `src/config/rating_thresholds.py` gives every rating bucket (AAA/AA/A/BBB/BB/B) its own loss threshold; `critic_agent.py` checks **every rated tranche**, not just the senior one |
| Critic's tranche resize was free text baked into the next script — `parsed_waterfall` in state never actually changed | New `current_tranches` field in `GraphState`, updated by a **structured** Critic output (`StructuralAdjustment` Pydantic model) with a deterministic zero-sum correction pass, not trusted LLM arithmetic |
| No iteration history — Reporter had to invent the "Delta" and "Iteration-by-Iteration" sections | `iteration_history` accumulates a real snapshot every structural loop; Reporter is told to use it as its only source of truth |
| No real macro scenario data — the "Macroeconomic Scenario Analysis" section was pure hallucination since only one simulation ever ran | New `stress_test_agent.py` node re-executes the exact **approved** code with base/mild/severe/extreme default & recovery assumptions substituted in, so the Reporter has real numbers |
| Sandbox was a regex blocklist over source text (`_check_for_dangerous_imports`) — bypassable with `__import__('os')` etc. | `python_repl.py` rewritten to run with a restricted `__builtins__` and a whitelist `__import__`, plus a wall-clock timeout |
| One shared iteration counter for both LLM code-errors and structural rejections | Split into `iteration_count` (structural/Critic loop, capped at `MAX_ITERATIONS`) and `code_retry_count` (Quant syntax/runtime retries, capped separately) |
| `app.py` hand-rolled its own copy of the initial state dict, separate from `src/main.py`, and had already drifted | Both now build from `src.main.make_initial_state(...)` |

## Architecture

```
START -> Parser -> Quant <-> Quant (code-error retries)
                     |
                     v
                  Critic --(rejected, budget left)--> Quant
                     |
            (approved) |  (exhausted budget)
                     v              \
              Stress Test        Reporter
                     |               ^
                     +---------------+
```

- **Parser** — pdfplumber table scan (tranche sizes, ratings, spreads — deterministic,
  no LLM) + a small RAG/LLM pass for coverage tests, fees, and waterfall metadata.
- **Quant** — writes a fresh Monte Carlo script each structural iteration, using a
  single-factor Gaussian copula for correlated defaults, executed in the sandboxed
  REPL.
- **Critic** — checks every rated tranche's loss probability against its rating
  threshold; on failure, returns a structured, zero-sum-validated resize plus
  cash-sweep instructions for the next Quant pass.
- **Stress Test** — once approved, reruns the *same* approved code under four
  macro scenarios (base/mild/severe/extreme) by substituting the default-rate and
  recovery-rate constants.
- **Reporter** — writes the executive summary strictly from the accumulated
  iteration log and scenario results (no invented numbers).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your free Groq key: https://console.groq.com/keys
```

## Run

```bash
# CLI, single PDF:
python -m src.main path/to/indenture.pdf

# Full UI, multi-thread, chat-with-your-deal:
streamlit run app.py
```

## Notes / known limits

- The Python sandbox (`src/tools/python_repl.py`) restricts builtins and imports,
  but it's still `exec()` in-process. Fine for your own test PDFs; if you ever
  point this at untrusted documents, move execution to a separate locked-down
  process or container.
- Rating thresholds in `src/config/rating_thresholds.py` are illustrative
  approximations of published idealized-loss tables, not the real agency numbers
  — swap in your own if you need this to be more than a demo.
- When a tranche's rating isn't stated verbatim in the PDF, it's inferred from
  class-name convention (Class A → AAA, Class B → AA, ...) and flagged as
  `rating_inferred: true` everywhere it's used, so it's never silently treated
  as an extracted fact.
