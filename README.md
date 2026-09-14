# CLO Waterfall Simulator

An agentic AI workflow built with LangGraph, LangChain, and Streamlit to extract, simulate, optimize, and stress-test Collateralized Loan Obligation (CLO) indentures from raw PDFs.

## 🚀 Overview

The CLO Waterfall Simulator is a multi-agent system designed to replace the manual, error-prone process of modeling structured finance vehicles. By ingesting a 300+ page legal indenture, the system automatically:
1. Extracts the capital structure and waterfall rules.
2. Generates and executes a custom Python Monte Carlo simulation.
3. Iteratively resizes the deal structure until it passes rating-agency loss thresholds.
4. Stress-tests the approved structure across severe macroeconomic scenarios.
5. Generates a fully grounded, hallucination-free executive report.

## ✨ Key Features

* **Hybrid Document Parsing:** Combines deterministic regex for numerical accuracy (tranche sizes, spreads) with a local HuggingFace ChromaDB RAG pipeline for complex legal rules (coverage tests, fees).
* **Generative Quant Engine:** Dynamically writes Pandas/Numpy simulation scripts utilizing a single-factor Gaussian copula to model correlated obligor defaults.
* **Sandboxed Execution:** Safely executes AI-generated code locally with strict module whitelisting, stripped built-ins, and wall-clock timeouts.
* **Automated Risk Structuring (Critic):** Evaluates loss probabilities against hardcoded Moody's/S&P limits. Automatically proposes zero-sum capital structure adjustments if tranches fail.
* **Macro Stress Testing:** Re-executes the *exact* approved simulation code under Base, Mild, Severe, and Extreme recession scenarios using deterministic regex variable injection.
* **Persistent Streaming UI:** A Streamlit interface featuring real-time agent execution streaming, SQLite threaded session memory, and a ReAct Chatbot Copilot to query live simulation data.

## 📦 Prerequisites

* Python 3.10+
* Groq API Key (for `openai/gpt-oss-120b` or equivalent models)
* (Optional) HuggingFace token

## 🛠️ Installation

1. **Clone the repository and set up a virtual environment:**
   ```bash
   git clone <your-repo-url>
   cd clo_simulator
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure `langgraph-checkpoint-sqlite` and `pypdf` are included in your environment).*

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   CLO_CHROMA_ROOT=./data/vector_store
   ```

4. **Disable Streamlit File Watcher (Optional but recommended to prevent Torchvision warnings):**
   Create `.streamlit/config.toml`:
   ```toml
   [server]
   fileWatcherType = "none"
   ```

## 🏃‍♂️ Usage

### Web Interface (Recommended)
Launch the persistent, multi-threaded Streamlit application:
```bash
streamlit run app.py
```
* Upload your CLO Indenture PDF via the sidebar.
* Adjust base macroeconomic assumptions.
* Click **Run Analysis** to watch the agents stream their progress.
* Use the **AI Assistant** tab to chat with the document and simulation results.

### Command Line Interface
Run the analysis headless from the terminal:
```bash
python -m src.main path/to/your/indenture.pdf
```

## 📁 Repository Structure

* `app.py`: Streamlit frontend and Chatbot UI.
* `src/main.py`: LangGraph state machine definition and routing logic.
* `src/state.py`: TypedDict defining the shared memory for the agents.
* `src/agents/`:
  * `parser_agent.py`: PDF extraction and RAG querying.
  * `quant_agent.py`: Monte Carlo Python code generation.
  * `critic_agent.py`: Rating threshold evaluation and zero-sum restructuring.
  * `stress_test_agent.py`: Multi-scenario execution.
  * `reporter_agent.py`: Executive summary generation.
  * `chatbot_agent.py`: ReAct assistant for the Streamlit UI.
* `src/tools/`:
  * `python_repl.py`: Secure Python execution sandbox.
  * `retriever.py`: PDF table extraction and ChromaDB management.
* `src/schemas/`: Pydantic models enforcing agent I/O contracts.

## 🛡️ Architecture
For a deep dive into the multi-agent workflow and data pipeline, see [ARCHITECTURE.md](ARCHITECTURE.md).