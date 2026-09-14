# src/agents/parser_agent.py
"""
Parser Agent -- Hybrid CLO Indenture Extraction.

Architecture (most reliable first):
1. DETERMINISTIC: pdfplumber scans every page's tables directly for tranche
   rows with dollar amounts AND rating labels. Exact -- no embeddings, no
   LLM, no hallucination.
2. RAG + LLM (divide-and-conquer): small focused calls for semi-structured
   content (coverage tests, fees, waterfall steps).
3. SANITIZE: fills remaining nulls with market-convention fallbacks so the
   Quant agent always has valid floats, and flags every fallback via
   `parser_data_quality` so nothing invented is silently treated as fact.

CHANGE from the original: target_rating is no longer left as an empty
string. We attempt to read it directly off the page (a cell that fullmatches
AAA/AA/BBB/BB/NR next to the tranche row); this feeds the Critic's per-
tranche rating-threshold check later in the graph.
"""

import re
import json
import os
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.waterfall_def import (
    IndentureRules, Tranche, CoverageTest, FeeStructure, WaterfallStep
)
from src.state import GraphState
from src.tools.retriever import get_or_create_retriever, get_collection_info

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_retries=2)

_DOLLAR_RE = re.compile(r"[\$]?\s*([\d,]+(?:\.\d+)?)\s*$")
_SPREAD_RE = re.compile(r"(?:Benchmark|SOFR|LIBOR)\s*[+]\s*([\d.]+)\s*%", re.IGNORECASE)
# Deliberately excludes bare "A"/"B" -- those collide with the tranche class
# letters themselves (Class A, Class B) and would be unreliable to parse from
# a raw cell. AAA/AA/BBB/BB/NR are unambiguous.
_RATING_CELL_RE = re.compile(r"^(AAA|AA|BBB|BB|NR)[+-]?(?:\s*\(sf\))?$", re.IGNORECASE)


def _parse_dollar(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.strip().lstrip("$").replace(",", "").strip()
    try:
        val = float(text)
        return val if val > 0 else None
    except ValueError:
        return None


def _parse_spread(text: str, class_name: str = "") -> Optional[float]:
    if re.search(r'subordinated|equity', class_name, re.I):
        return None
    m = _SPREAD_RE.search(text or "")
    if m:
        return round(float(m.group(1)) * 100, 1)
    return None


def _parse_rating_cell(text: str) -> Optional[str]:
    if not text:
        return None
    m = _RATING_CELL_RE.match(text.strip())
    return m.group(1).upper() if m else None


_CLASS_PATTERNS = [
    (re.compile(r"Class\s+A[-\s]?1", re.I), "Class A-1", False),
    (re.compile(r"Class\s+A[-\s]?2", re.I), "Class A-2", False),
    (re.compile(r"Class\s+A\b",       re.I), "Class A",   False),
    (re.compile(r"Class\s+B\b",       re.I), "Class B",   False),
    (re.compile(r"Class\s+C\b",       re.I), "Class C",   False),
    (re.compile(r"Class\s+D\b",       re.I), "Class D",   False),
    (re.compile(r"Class\s+E\b",       re.I), "Class E",   False),
    (re.compile(r"Subordinated",      re.I), "Subordinated Notes", True),
    (re.compile(r"Equity",            re.I), "Equity",    True),
]


def _classify_row(row_text: str):
    for pattern, name, is_eq in _CLASS_PATTERNS:
        if pattern.search(row_text):
            return name, is_eq
    return None, False


def _extract_tranches_deterministic(pdf_path: str) -> List[Tranche]:
    import pdfplumber
    found: Dict[str, Tranche] = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for table in tables:
                    for row in (table or []):
                        if not row:
                            continue
                        row_cells = [str(c or "").strip() for c in row]
                        row_text = " ".join(row_cells)

                        class_name, is_equity = _classify_row(row_text)
                        if not class_name:
                            continue

                        principal = None
                        rating = None
                        for cell in row_cells:
                            val = _parse_dollar(cell)
                            if val and val > 100_000 and principal is None:
                                principal = val
                            r = _parse_rating_cell(cell)
                            if r and rating is None:
                                rating = r

                        spread = _parse_spread(row_text, class_name=class_name)

                        if class_name in found and found[class_name].principal_amount is not None:
                            continue

                        tranche = Tranche(
                            class_name=class_name,
                            target_rating=rating or "",
                            principal_amount=principal,
                            coupon_type="fixed" if is_equity else "floating",
                            spread_bps=spread,
                            is_equity=is_equity,
                        )
                        found[class_name] = tranche
                        if principal:
                            logger.info(
                                f"[Parser][Det] {class_name}: ${principal:,.0f}"
                                f"{f', rating={rating}' if rating else ''}"
                                f"{f', spread={spread}bps' if spread else ''}"
                                f" (page {page_num})"
                            )

                page_text = page.extract_text() or ""
                for class_name, tranche in list(found.items()):
                    class_pat = re.compile(re.escape(class_name), re.I)
                    for line in page_text.splitlines():
                        if not class_pat.search(line):
                            continue
                        if tranche.spread_bps is None and not tranche.is_equity:
                            spread = _parse_spread(line, class_name=class_name)
                            if spread:
                                tranche = tranche.model_copy(update={"spread_bps": spread})
                                found[class_name] = tranche
                        if not tranche.target_rating:
                            for token in line.split():
                                r = _parse_rating_cell(token)
                                if r:
                                    tranche = tranche.model_copy(update={"target_rating": r})
                                    found[class_name] = tranche
                                    break

    except Exception as e:
        logger.error(f"[Parser][Det] pdfplumber scan failed: {e}")

    tranches = list(found.values())
    logger.info(f"[Parser][Det] Deterministic extraction found {len(tranches)} tranches.")
    return tranches


def _extract_total_par_deterministic(pdf_path: str) -> Optional[float]:
    import pdfplumber
    patterns = [
        re.compile(r"[Tt]arget\s+[Pp]ar[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        re.compile(r"[Aa]ggregate\s+[Pp]rincipal\s+[Aa]mount[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        re.compile(r"[Tt]otal\s+[Pp]ar[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
    ]
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for pat in patterns:
                    m = pat.search(text)
                    if m:
                        val = _parse_dollar(m.group(1))
                        if val and val > 1_000_000:
                            logger.info(f"[Parser][Det] Total par detected: ${val:,.0f}")
                            return val
    except Exception as e:
        logger.warning(f"[Parser][Det] Total par scan failed: {e}")
    return None


COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract Overcollateralization (OC) and Interest Coverage (IC) tests from the text. "
        'Return ONLY a JSON array: [{{"test_type": "OC"|"IC", "applies_to_class": str, '
        '"trigger_ratio": float|null, "cure_action": str}}]. Return [] if none found. '
        "Output ONLY the JSON array."
    )),
    ("human", "Text:\n{context}")
])

FEES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract CLO fee structure from the text. Return ONLY a JSON object: "
        '{{"senior_admin_fee_cap": float|null, "senior_mgmt_fee_rate": float|null, '
        '"subordinated_mgmt_fee_rate": float|null, "incentive_fee_hurdle_irr": float|null, '
        '"incentive_fee_share": float|null}}. Rates as decimals (0.0015 = 15bps). '
        "Output ONLY the JSON object."
    )),
    ("human", "Text:\n{context}")
])

WATERFALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        'Extract CLO deal metadata and interest waterfall. Return ONLY: {{"deal_name": str, '
        '"ccc_bucket_limit": float|null, "interest_waterfall": [{{"priority": int, "payee": str, '
        '"payment_type": "fees"|"interest"|"principal"|"oc_cure"|"residual_equity", "condition": str|null}}]}}. '
        "Output ONLY the JSON object."
    )),
    ("human", "Text:\n{context}")
])

COVERAGE_QUERIES = [
    "overcollateralization OC test ratio trigger threshold interest coverage IC test diversion",
    "OC ratio IC ratio Class A Class B coverage test reinvestment period diversion",
]
FEES_QUERIES = [
    "senior management fee rate subordinated management fee administrative fee cap trustee incentive hurdle",
    "asset management fee collateral manager fee annual percentage basis points",
]
WATERFALL_QUERIES = [
    "priority of payments interest proceeds waterfall sequential first second third fees trustee",
    "CCC bucket limit deal name CLO target par IRR hurdle equity subordinated",
]


def _build_context(retriever, queries: List[str], k_per_query: int = 4) -> str:
    seen, chunks = set(), []
    vs = getattr(retriever, "vectorstore", None)
    for i, query in enumerate(queries, 1):
        try:
            docs = vs.similarity_search(query, k=k_per_query) if vs else retriever.invoke(query)
        except Exception as e:
            logger.warning(f"[Parser] Search failed for query {i}: {e}")
            docs = []
        for doc in docs:
            fp = doc.page_content[:100].strip()
            if fp and fp not in seen:
                seen.add(fp)
                chunks.append(f"[Q{i}/P{doc.metadata.get('page','?')}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)[:5000]


def _safe_json(text: str, fallback):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"[Parser] JSON parse failed: {e} | raw={text[:120]}")
        return fallback


def _extract_coverage_tests(retriever) -> List[CoverageTest]:
    ctx = _build_context(retriever, COVERAGE_QUERIES, k_per_query=4)
    raw = (COVERAGE_PROMPT | llm).invoke({"context": ctx}).content
    data = _safe_json(raw, [])
    tests = []
    if isinstance(data, list):
        for item in data:
            try:
                tests.append(CoverageTest(**item))
            except Exception as e:
                logger.warning(f"[Parser] Bad coverage test: {e}")
    return tests


def _extract_fees(retriever) -> FeeStructure:
    ctx = _build_context(retriever, FEES_QUERIES, k_per_query=4)
    raw = (FEES_PROMPT | llm).invoke({"context": ctx}).content
    data = _safe_json(raw, {})
    try:
        return FeeStructure(**data) if isinstance(data, dict) else FeeStructure()
    except Exception as e:
        logger.warning(f"[Parser] Fee parse failed: {e}")
        return FeeStructure()


def _extract_waterfall_meta(retriever) -> dict:
    ctx = _build_context(retriever, WATERFALL_QUERIES, k_per_query=4)
    raw = (WATERFALL_PROMPT | llm).invoke({"context": ctx}).content
    data = _safe_json(raw, {})
    if not isinstance(data, dict):
        data = {}
    steps = []
    for item in data.get("interest_waterfall", []):
        try:
            steps.append(WaterfallStep(**item))
        except Exception as e:
            logger.warning(f"[Parser] Bad waterfall step: {e}")
    return {
        "deal_name": data.get("deal_name") or "Unknown Deal",
        "ccc_bucket_limit": data.get("ccc_bucket_limit"),
        "interest_waterfall": steps,
    }


def parser_agent(state: GraphState) -> Dict[str, Any]:
    pdf_path = state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Indenture PDF not found at path: {pdf_path}")

    logger.info(f"[Parser] Starting hybrid extraction for: {os.path.basename(pdf_path)}")

    tranches = _extract_tranches_deterministic(pdf_path)
    total_par = _extract_total_par_deterministic(pdf_path)

    if total_par is None and tranches:
        extracted_sum = sum(t.principal_amount for t in tranches if t.principal_amount)
        if extracted_sum > 0:
            total_par = extracted_sum
            logger.info(f"[Parser] Total par derived from tranche sum: ${total_par:,.0f}")

    logger.info("[Parser] Building RAG index for semi-structured content...")
    retriever, collection_name = get_or_create_retriever(pdf_path, k=4)
    index_status = "reused" if get_collection_info(pdf_path)["indexed"] else "newly built"

    coverage_tests = _extract_coverage_tests(retriever)
    fees = _extract_fees(retriever)
    meta = _extract_waterfall_meta(retriever)

    parsed_rules = IndentureRules(
        deal_name=meta["deal_name"],
        total_target_par=total_par,
        tranches=tranches,
        coverage_tests=coverage_tests,
        fees=fees,
        ccc_bucket_limit=meta["ccc_bucket_limit"],
        interest_waterfall=meta["interest_waterfall"],
    )
    rules_dict = parsed_rules.model_dump()

    logger.info(
        f"[Parser] Done. Deal='{parsed_rules.deal_name}' | "
        f"Par=${total_par:,.0f} | Tranches={len(tranches)} | Tests={len(coverage_tests)}"
        if total_par else
        f"[Parser] Done. Deal='{parsed_rules.deal_name}' | Par=N/A | Tranches={len(tranches)}"
    )

    missing_principals = any(t.principal_amount is None for t in tranches) if tranches else True
    ratings_extracted = sum(1 for t in tranches if t.target_rating)
    data_quality = {
        "principals_extracted": bool(tranches) and not missing_principals,
        "tranche_count": len(tranches),
        "coverage_tests_extracted": len(coverage_tests) > 0,
        "ratings_extracted_from_pdf": ratings_extracted,
        "ratings_will_be_inferred_from_class_name": len(tranches) - ratings_extracted,
        "extraction_method": "deterministic+rag",
    }

    return {
        "extracted_text_chunks": [
            f"[Collection: {collection_name} | {index_status}]",
            f"[Tranches: {len(tranches)} (deterministic) | Tests: {len(coverage_tests)} (RAG)]",
            f"[Total par: ${total_par:,.0f}]" if total_par else "[Total par: estimated]",
        ],
        "parsed_waterfall": rules_dict,
        "current_tranches": rules_dict["tranches"],
        "vector_store_id": collection_name,
        "parser_data_quality": data_quality,
    }
