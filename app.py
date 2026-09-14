"""
CLO Waterfall Simulator -- Streaming Multi-Thread Interface
==========================================================

Features:
- Each CLO run gets its own persistent thread (UUID -> SQLite checkpointer)
- Real-time streaming: live per-agent status as the pipeline executes
- Thread sidebar: browse, switch, and reload any past analysis
- Rich result cards: extracted rules, tranche resize history, generated code,
  base-case simulation JSON, macro stress-test table, and the final report
- Per-PDF isolated vector store with inline management controls

CHANGE from the original: the initial-state dict used to be hand-rolled here
AND separately in src/main.py -- they had already drifted (this file never
set the default_correlation/base_recovery_rate the Quant prompt now actually
reads). Both now build their initial state from the single
`src.main.make_initial_state` factory.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Optional

import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langgraph.checkpoint.sqlite import SqliteSaver
from src.main import build_clo_graph, make_initial_state, MAX_ITERATIONS
from src.agents.chatbot_agent import build_chatbot_graph
from src.tools.retriever import get_collection_info, delete_collection

st.set_page_config(page_title="CLO Waterfall Simulator", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-approved { background: rgba(52,211,153,0.15); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
.badge-rejected { background: rgba(239,68,68,0.15);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.badge-running  { background: rgba(251,191,36,0.15); color: #fde68a; border: 1px solid rgba(251,191,36,0.3); }
.badge-error    { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-pending  { background: rgba(148,163,184,0.10);color: #94a3b8; border: 1px solid rgba(148,163,184,0.2);}
.metric-tile { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 14px 18px; text-align: center; }
.metric-val  { font-size: 1.6rem; font-weight: 700; }
.metric-label{ font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }
.section-title { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #64748b; margin: 16px 0 6px 0; }
.hero { text-align: center; padding: 60px 20px 40px; }
.hero h1 { font-size: 2.6rem; font-weight: 700; margin-bottom: 8px; }
.hero p  { font-size: 1.05rem; color: #94a3b8; max-width: 560px; margin: 0 auto 32px; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data/raw_indentures"
DB_PATH = "data/workflow_threads.db"
KNOWN_NODES = {"parser_node", "quant_node", "critic_node", "stress_test_node", "reporter_node"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


@st.cache_resource
def _get_app_and_checkpointer():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cp = SqliteSaver(conn=conn)
    app = build_clo_graph(checkpointer=cp)
    chat = build_chatbot_graph(checkpointer=cp)
    return app, chat, cp


clo_app, chatbot_app, checkpointer = _get_app_and_checkpointer()


def new_thread_id() -> str:
    return str(uuid.uuid4())


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def list_all_thread_ids() -> list:
    seen = set()
    try:
        for cp in checkpointer.list(None):
            tid = cp.config["configurable"]["thread_id"]
            if not str(tid).endswith("-chat"):
                seen.add(tid)
    except Exception:
        pass
    return list(seen)


def delete_analysis_thread(tid: str):
    try:
        if hasattr(checkpointer, "delete_thread"):
            checkpointer.delete_thread(tid)
            checkpointer.delete_thread(f"{tid}-chat")
        elif hasattr(checkpointer, "conn"):
            cur = checkpointer.conn.cursor()
            for table in ("checkpoints", "checkpoint_writes"):
                cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
                cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (f"{tid}-chat",))
            checkpointer.conn.commit()

        if tid in st.session_state.all_thread_ids:
            st.session_state.all_thread_ids.remove(tid)
        st.session_state.thread_meta.pop(tid, None)
        if st.session_state.active_thread == tid:
            st.session_state.active_thread = None
    except Exception as e:
        st.error(f"Failed to delete thread: {e}")


def load_thread_state(thread_id: str) -> Optional[dict]:
    try:
        snap = clo_app.get_state(config=make_config(thread_id))
        return dict(snap.values) if snap and snap.values else None
    except Exception:
        return None


def fmt_money(val) -> str:
    if val is None:
        return "N/A"
    val = float(val)
    if val >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    if val >= 1_000_000:
        return f"${val/1_000_000:.0f}M"
    return f"${val:,.0f}"


def status_badge_html(status: str) -> str:
    cls = {"approved": "badge-approved", "rejected": "badge-rejected", "simulated": "badge-running",
           "quant_error": "badge-error", "failed": "badge-error"}.get(status, "badge-pending")
    return f'<span class="badge {cls}">{status}</span>'


if "active_thread" not in st.session_state: st.session_state.active_thread = None
if "pending_pdf" not in st.session_state: st.session_state.pending_pdf = None
if "all_thread_ids" not in st.session_state: st.session_state.all_thread_ids = list_all_thread_ids()
if "thread_meta" not in st.session_state:
    st.session_state.thread_meta = {}
    for tid in st.session_state.all_thread_ids:
        state = load_thread_state(tid)
        if state:
            pw = state.get("parsed_waterfall") or {}
            pdf_path = state.get("pdf_path", "")
            st.session_state.thread_meta[tid] = {
                "deal_name": pw.get("deal_name", f"Thread {tid[:8]}"),
                "status": state.get("status", "completed"),
                "pdf_name": os.path.basename(pdf_path) if pdf_path else "",
            }

with st.sidebar:
    st.markdown("## 📊 CLO Simulator")
    st.caption("Agentic AI · Multi-Thread · Streaming")
    st.divider()

    if st.button("➕  New Analysis", use_container_width=True, type="primary"):
        st.session_state.active_thread = None
        st.session_state.pending_pdf = None
        st.rerun()

    st.markdown('<p class="section-title">📁 Document</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed", key="pdf_upload")
    if uploaded:
        save_path = os.path.join(DATA_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.pending_pdf = save_path
        st.success(f"✅ {uploaded.name}")

    existing = sorted(os.listdir(DATA_DIR))
    if existing:
        pick = st.selectbox("Or select existing", ["—"] + existing, key="pdf_pick", label_visibility="collapsed")
        if pick != "—":
            st.session_state.pending_pdf = os.path.join(DATA_DIR, pick)

    if st.session_state.pending_pdf:
        info = get_collection_info(st.session_state.pending_pdf)
        st.markdown(f"<small>📄 `{os.path.basename(st.session_state.pending_pdf)}`</small>", unsafe_allow_html=True)
        if info["indexed"]:
            st.success(f"🗄️ Indexed · {info.get('index_size_mb', '?')} MB")
            if st.button("🗑️ Clear Index", key="clear_idx"):
                delete_collection(st.session_state.pending_pdf)
                st.warning("Index cleared — will re-embed on next run.")
                st.rerun()
        else:
            st.info("⏳ Not yet indexed")

        with st.expander("⚙️ Macro assumptions for this run"):
            default_rate = st.slider("Base annual default probability", 0.0, 0.15, 0.02, 0.005, format="%.3f")
            recovery_rate = st.slider("Base recovery rate", 0.0, 1.0, 0.70, 0.05)
            correlation = st.slider("Default correlation", 0.0, 0.9, 0.20, 0.05)
            st.session_state["_run_assumptions"] = (default_rate, recovery_rate, correlation)

    st.markdown('<p class="section-title">🕐 Past Analyses</p>', unsafe_allow_html=True)
    all_tids = st.session_state.all_thread_ids
    if not all_tids:
        st.caption("No analyses yet.")
    else:
        for tid in reversed(all_tids):
            meta = st.session_state.thread_meta.get(tid, {})
            deal, status, pdf_n, ts = meta.get("deal_name", f"Thread {tid[:8]}"), meta.get("status", ""), meta.get("pdf_name", ""), meta.get("timestamp", "")
            icon = {"approved": "✅", "rejected": "❌", "quant_error": "⚠️"}.get(status, "⏺")
            label = f"{icon}  {deal[:28]}" + (f"\n`{pdf_n[:22]}`" if pdf_n else "") + (f"\n{ts[:16]}" if ts else "")
            is_active = tid == st.session_state.active_thread
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(label, key=f"th-{tid}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state.active_thread = tid
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del-{tid}", help="Delete this analysis"):
                    delete_analysis_thread(tid)
                    st.rerun()


def _view_json_modal(title: str, data):
    @st.dialog(title)
    def show(): st.json(data)
    show()


def _view_code_modal(title: str, code: str):
    @st.dialog(title)
    def show(): st.code(code, language="python")
    show()


def _parser_card(state: dict):
    pw = state.get("parsed_waterfall") or {}
    deal, par, n_tr, n_ct = pw.get("deal_name") or "—", pw.get("total_target_par") or 0, len(pw.get("tranches") or []), len(pw.get("coverage_tests") or [])
    with st.container(border=True):
        st.markdown(f"#### 📄 Parser Agent &nbsp; {status_badge_html('approved')}", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deal Name", deal[:22] + ("…" if len(deal) > 22 else ""))
        c2.metric("Total Par", fmt_money(par))
        c3.metric("Tranches", n_tr)
        c4.metric("Coverage Tests", n_ct)
        if st.button("📋 View Extracted IndentureRules JSON", key="btn_parser_json"):
            _view_json_modal("Extracted IndentureRules JSON", pw)

        dq = state.get("parser_data_quality") or {}
        if dq and not dq.get("principals_extracted", True):
            st.warning("⚠️ Principal amounts could not be reliably extracted — results use fallbacks. Try clearing the index and re-running.", icon="⚠️")
        elif dq:
            st.success("✅ Tranche principal amounts extracted from the document.")
            inferred = dq.get("ratings_will_be_inferred_from_class_name", 0)
            if inferred:
                st.caption(f"ℹ️ {inferred} tranche rating(s) not found verbatim in the PDF — inferred from class-name convention for threshold checks.")
        st.caption(f"🗄️ Vector Store Collection: `{state.get('vector_store_id', '—')}`")


def _quant_card(state: dict, iteration: int):
    code, results, err = state.get("generated_code", ""), state.get("simulation_results") or {}, state.get("execution_error")
    badge_cls, badge_text = ("badge-approved", "✓ Executed") if not err else ("badge-error", "⚠ Error")
    with st.container(border=True):
        st.markdown(f"#### 📊 Quant Agent &nbsp; <span class='badge {badge_cls}'>{badge_text}</span> "
                    f"&nbsp; <small style='color:#64748b'>Iteration {iteration}</small>", unsafe_allow_html=True)
        if err:
            st.error(f"**Runtime Error:** {err}")
        cols = st.columns(2)
        with cols[0]:
            if code and st.button("💻 View Generated Python Code", key="btn_quant_code"):
                _view_code_modal("Generated Python Code", code)
        with cols[1]:
            if results and st.button("📈 View Monte Carlo JSON", key="btn_quant_json"):
                _view_json_modal("Monte Carlo Raw JSON (base case)", results)
        if results and isinstance(results, dict):
            import pandas as pd
            st.markdown("**Base-Case Loss Probability per Tranche**")
            df = pd.DataFrame([{"Tranche": k, "Loss Probability": f"{v:.2%}"} for k, v in results.items()])
            st.dataframe(df, hide_index=True, use_container_width=True)


def _critic_card(state: dict, iteration: int):
    feedback, status = state.get("critic_feedback", "—"), state.get("status", "")
    badge_cls, badge_text = ("badge-approved", "✅ APPROVED") if status == "approved" else ("badge-rejected", "↩ REJECTED")
    with st.container(border=True):
        st.markdown(f"#### ⚖️ Critic Agent &nbsp; <span class='badge {badge_cls}'>{badge_text}</span> "
                    f"&nbsp; <small style='color:#64748b'>Iteration {iteration}</small>", unsafe_allow_html=True)
        st.info(feedback or "No feedback provided.")


def _history_card(state: dict):
    history = state.get("iteration_history") or []
    if not history:
        return
    with st.container(border=True):
        st.markdown("#### 🔁 Structural Iteration Log", unsafe_allow_html=True)
        for entry in history:
            label = f"Iteration {entry['iteration']} — {entry.get('status', '').upper()}"
            with st.expander(label):
                breaches = entry.get("breaches") or []
                if breaches:
                    st.write("**Breaches:**")
                    for b in breaches:
                        st.write(f"- {b['class_name']} ({b.get('rating', '?')}): {b['observed_loss_prob']:.4%} > {b['threshold']:.4%}")
                else:
                    st.write("No breaches.")
                st.caption(entry.get("critic_feedback", ""))
                if st.button("View tranche snapshot", key=f"hist-{entry['iteration']}"):
                    _view_json_modal(f"Tranches at Iteration {entry['iteration']}", entry.get("tranches_snapshot"))


def _scenario_card(state: dict):
    macro = state.get("macro_scenario_results")
    if not macro:
        return
    with st.container(border=True):
        st.markdown(f"#### 🌩️ Macro Scenario Stress Test &nbsp; {status_badge_html('approved')}", unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for scenario, payload in macro.items():
            assumptions = payload.get("assumptions", {})
            row = {"Scenario": scenario.replace("_", " ").title(),
                   "Default Rate": f"{assumptions.get('annual_default_prob', 0):.1%}",
                   "Recovery Rate": f"{assumptions.get('recovery_rate', 0):.0%}"}
            if payload.get("success") and payload.get("results"):
                for tranche, loss in payload["results"].items():
                    row[f"{tranche} Loss"] = f"{loss:.2%}"
            else:
                row["Status"] = "Simulation failed"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _reporter_card(state: dict):
    report = state.get("final_report", "")
    with st.container(border=True):
        st.markdown(f"#### 📑 Reporter Agent &nbsp; {status_badge_html('approved')}", unsafe_allow_html=True)
        st.markdown(report) if report else st.caption("No report generated.")


def display_full_results(thread_id: str, state: dict):
    pw = state.get("parsed_waterfall") or {}
    status, iters = state.get("status", "unknown"), state.get("iteration_count", 0)
    meta = st.session_state.thread_meta.get(thread_id, {})

    st.markdown(f"## {pw.get('deal_name', 'CLO Analysis')}")
    h1, h2, h3, h4 = st.columns(4)
    for col, val, label in [(h1, status.upper(), "Status"), (h2, iters, "Iterations"),
                             (h3, fmt_money(pw.get("total_target_par", 0)), "Total Par"),
                             (h4, f"{thread_id[:8]}…", "Thread")]:
        col.markdown(f'<div class="metric-tile"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    if meta.get("timestamp"):
        st.caption(f"Run at: {meta['timestamp']}")
    st.divider()

    _parser_card(state)
    _quant_card(state, iters)
    _critic_card(state, iters)
    _history_card(state)
    if state.get("macro_scenario_results"):
        _scenario_card(state)
    if state.get("final_report"):
        _reporter_card(state)


def display_chat_interface(thread_id: str, workflow_state: dict):
    st.markdown("### 💬 Chat with your CLO Data")
    st.caption("Ask about the document, tranches, iteration history, or scenario results.")

    chat_thread_id = f"{thread_id}-chat"
    config = {"configurable": {"thread_id": chat_thread_id}}

    try:
        chat_state = chatbot_app.get_state(config)
        messages = chat_state.values.get("messages", [])
    except Exception:
        messages = []

    if not messages:
        pw = workflow_state.get("parsed_waterfall") or {}
        system_injection = (
            f"[System Context]\nActive PDF Path: `{workflow_state.get('pdf_path') or ''}`\n"
            f"Active Workflow Thread ID: `{thread_id}`\nDeal Name: `{pw.get('deal_name') or 'Unknown Deal'}`\n"
        )
        chatbot_app.update_state(config, {"messages": [("system", system_injection)]})
        chat_state = chatbot_app.get_state(config)
        messages = chat_state.values.get("messages", [])

    for msg in messages:
        if msg.type == "system":
            continue
        if msg.type == "human":
            with st.chat_message("user"): st.write(msg.content)
        elif msg.type == "ai":
            with st.chat_message("assistant"):
                if msg.content: st.write(msg.content)
                for tc in (getattr(msg, "tool_calls", None) or []):
                    with st.expander(f"🛠️ Executed Tool: `{tc['name']}`"): st.json(tc["args"])
        elif msg.type == "tool":
            with st.chat_message("tool"):
                with st.expander(f"🔧 Tool Result: `{msg.name}`"): st.write(msg.content)

    if prompt := st.chat_input("Ask about the indenture or results..."):
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    for chunk in chatbot_app.stream({"messages": [("user", prompt)]}, config=config, stream_mode="updates"):
                        if "agent" in chunk:
                            msg = chunk["agent"]["messages"][-1]
                            if msg.content: st.write(msg.content)
                            for tc in (getattr(msg, "tool_calls", None) or []):
                                with st.status(f"🛠️ Executing `{tc['name']}`...", state="running") as s:
                                    st.json(tc["args"]); s.update(state="complete")
                        elif "tools" in chunk:
                            msg = chunk["tools"]["messages"][-1]
                            with st.expander(f"🔧 Tool Result: {msg.name}"): st.write(msg.content)
                except Exception as e:
                    st.error(f"Error: {e}")
        st.rerun()


def run_streaming_analysis(pdf_path: str):
    thread_id = new_thread_id()
    st.session_state.active_thread = thread_id
    if thread_id not in st.session_state.all_thread_ids:
        st.session_state.all_thread_ids.append(thread_id)

    config = make_config(thread_id)
    pdf_basename = os.path.basename(pdf_path)
    default_rate, recovery_rate, correlation = st.session_state.get("_run_assumptions", (0.02, 0.70, 0.20))
    initial_state = make_initial_state(pdf_path, default_rate=default_rate, recovery_rate=recovery_rate, correlation=correlation)

    info = get_collection_info(pdf_path)
    phase_label = "📥 Embedding PDF into Vector Store (first time — this may take 1-2 min)…" if not info["indexed"] else "🤖 Running Multi-Agent CLO Analysis Pipeline…"

    collected, final_state = {}, {}
    with st.status(phase_label, expanded=True) as workflow_status:
        st.write(f"📄 Document: **{pdf_basename}**")
        st.write(f"🔑 Thread ID: `{thread_id[:16]}…`")
        st.markdown("---")

        for chunk in clo_app.stream(initial_state, config=config, stream_mode="updates"):
            node_name = next(iter(chunk))
            output = chunk[node_name] or {}
            if node_name not in KNOWN_NODES:
                continue
            collected.update(output)
            final_state.update(output)

            if node_name == "parser_node":
                pw = output.get("parsed_waterfall") or {}
                st.write(f"✅ **Parser Agent** — *{pw.get('deal_name', 'Unknown Deal')}* · "
                         f"{fmt_money(pw.get('total_target_par') or 0)} par · {len(pw.get('tranches') or [])} tranches")
            elif node_name == "quant_node":
                it, st_, err = output.get("iteration_count", "?"), output.get("status", ""), output.get("execution_error")
                if st_ == "quant_error":
                    st.write(f"⚠️ **Quant Agent** — Code error, retrying (structural iteration {it})…")
                    if err: st.code(err[:300], language="text")
                else:
                    st.write(f"✅ **Quant Agent** — Simulation executed (iteration {it})")
            elif node_name == "critic_node":
                it, st_ = output.get("iteration_count", "?"), output.get("status", "")
                st.write(f"✅ **Critic Agent** — ✨ Model **APPROVED** on iteration {it}!" if st_ == "approved"
                         else f"🔄 **Critic Agent** — Rejected (iteration {it}), sending back to Quant…")
                fb = output.get("critic_feedback", "")
                if fb: st.caption(fb[:220] + ("…" if len(fb) > 220 else ""))
            elif node_name == "stress_test_node":
                n_ok = sum(1 for v in (output.get("macro_scenario_results") or {}).values() if v.get("success"))
                st.write(f"✅ **Stress-Test Agent** — Ran {n_ok}/4 macro scenarios against the approved structure")
            elif node_name == "reporter_node":
                st.write("✅ **Reporter Agent** — Executive summary written")

        final_status = final_state.get("status", "unknown")
        workflow_status.update(label=f"{'✅' if final_status == 'approved' else '⚠️'} Pipeline Complete — **{final_status.upper()}**",
                                state="complete", expanded=False)

    pw = final_state.get("parsed_waterfall") or {}
    st.session_state.thread_meta[thread_id] = {
        "deal_name": pw.get("deal_name", f"Thread {thread_id[:8]}"),
        "pdf_name": pdf_basename,
        "status": final_state.get("status", "unknown"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    st.divider()
    display_full_results(thread_id, final_state)


def show_welcome():
    st.markdown("""
    <div class="hero">
        <h1>📊 CLO Waterfall Simulator</h1>
        <p>Upload a CLO Indenture PDF — AI agents extract the legal rules, write and run a
        Monte Carlo simulation, rating-check every tranche, resize the structure until it
        passes, then stress-test the approved deal across base/mild/severe/extreme recession
        scenarios and write an executive summary grounded in the actual run.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for col, emoji, label, color in [
        (cols[0], "📄", "Parser", "#818cf8"), (cols[1], "📊", "Quant", "#22d3ee"),
        (cols[2], "⚖️", "Critic", "#f59e0b"), (cols[3], "🌩️", "Stress Test", "#fb7185"),
        (cols[4], "📑", "Reporter", "#34d399"),
    ]:
        col.markdown(f'<div class="metric-tile"><div class="metric-val" style="color:{color}">{emoji}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.pending_pdf:
        pdf_name = os.path.basename(st.session_state.pending_pdf)
        st.markdown(f"**Selected:** `{pdf_name}`")
        info = get_collection_info(st.session_state.pending_pdf)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🚀  Run Analysis", type="primary", use_container_width=True):
                run_streaming_analysis(st.session_state.pending_pdf)
        with col_info:
            st.success(f"🗄️ Already indexed ({info.get('index_size_mb', '?')} MB).") if info["indexed"] else st.info("⏳ First run will embed the PDF before analysis begins.")
    else:
        st.info("← Select or upload a CLO Indenture PDF from the sidebar, then click **Run Analysis**.")


active = st.session_state.active_thread
if active:
    state = load_thread_state(active)
    if state:
        if st.button("← New Analysis", key="back_btn"):
            st.session_state.active_thread = None
            st.rerun()
        st.divider()
        tab1, tab2 = st.tabs(["📊 Pipeline Results", "💬 AI Assistant"])
        with tab1: display_full_results(active, state)
        with tab2: display_chat_interface(active, state)
    else:
        st.warning(f"Could not load state for thread `{active}`.")
        if st.button("← Back"):
            st.session_state.active_thread = None
            st.rerun()
else:
    show_welcome()
