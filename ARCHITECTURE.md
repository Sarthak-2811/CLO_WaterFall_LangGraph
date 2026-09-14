# System Architecture & Multi-Agent Workflow

The CLO Waterfall Simulator leverages a stateful, multi-agent architecture powered by **LangGraph**. Instead of relying on a single large prompt, the system divides the complex task of financial structuring into specialized agents, each constrained by strict tools, schemas, and routing logic.

## 🗺️ The State Machine (Graph Flow)

The workflow operates as a cyclic graph with two distinct iterative loops: an inner code-correction loop and an outer financial-structuring loop.

```text
  START
    │
    ▼
[ Parser Agent ] ────────────────────────────────┐
    │                                            │ (If extraction fails)
    ▼                                            │
[ Quant Agent ] ◄──────┐                         │
    │                  │ (Syntax/Runtime Error)  │
    ▼                  │                         │
[ Code Sandbox ] ──────┘                         │
    │ (Success)                                  │
    ▼                                            │
[ Critic Agent ] ────────────────┐               │
    │                            │ (Rejected:    │
    │ (Approved)                 │  Structure    │
    ▼                            │  Fails)       │
[ Stress-Test Agent ]            │               │
    │                            │               │
    ▼                            │               │
[ Reporter Agent ] ◄─────────────┘               │
    │               (Max Iterations Reached)     │
    ▼                                            ▼
   END ◄─────────────────────────────────────────┘
```

## 🧠 Core Components

### 1. `GraphState` (The Central Memory)
The entire graph is governed by a unified `TypedDict` state object. It explicitly separates raw document data (`parsed_waterfall`), the live mutable data (`current_tranches`), financial assumptions, and strict execution history logs. 

### 2. The Specialists (Agents)

* **Parser Agent (The Reader):** Employs a hybrid architecture. It bypasses LLM hallucinations for critical numbers by using deterministic Python regex over `pdfplumber` table extractions. It only delegates to an LLM (via a localized ChromaDB RAG pipeline) to parse semi-structured text like coverage tests and fee waterfalls. Output is strictly validated via Pydantic.
* **Quant Agent (The Math Engine):** A generative coder. It reads the current capital structure and macro assumptions and writes a bespoke `numpy`/`pandas` Monte Carlo simulation script. It forces the use of a single-factor Gaussian copula to ensure defaults are properly correlated (tail risk).
* **Critic Agent (The Risk Officer):** Evaluates the Quant's output against a hardcoded dictionary of idealized rating-agency loss thresholds (e.g., AAA = 0.01% max loss). If the structure fails, it uses a Pydantic `StructuralAdjustment` schema to propose a zero-sum resize (enforced deterministically in Python to prevent LLM arithmetic drift) and kicks the graph back to the Quant Agent.
* **Stress-Test Agent (The Economist):** Runs the final, Critic-approved simulation across Base, Mild, Severe, and Extreme scenarios. Crucially, it does **not** use an LLM. It uses deterministic regex to inject new default/recovery variables into the approved Python string, ensuring the exact same financial logic is tested without introducing generative bugs.
* **Reporter Agent (The Analyst):** Generates the final Markdown summary. It is explicitly fed the literal `iteration_history` and `macro_scenario_results` dicts, forcing it to ground its report in the actual data produced by the graph rather than hallucinating an optimization journey.

### 3. The Tooling layer

* **Execution Sandbox (`python_repl.py`):** Protects the host machine from the Quant's generated code. It overrides `__import__` to strictly whitelist data science libraries, lobotomizes `__builtins__` to prevent file/network access, and enforces a strict OS-level `signal` wall-clock timeout to prevent infinite loops.
* **Threaded Checkpointer:** Utilizes LangGraph's `SqliteSaver` to persist every node execution to a local `.db` file. This allows the Streamlit UI to instantly reload past analysis threads and provides context to the Chatbot Agent.