"""
Retriever Tool: RAG pipeline for querying 300-page CLO Indenture PDFs.

Design (unchanged from the original -- this part of the codebase was solid):
- Each PDF gets its own isolated ChromaDB collection (keyed by a stable hash
  of the filename). Switching PDFs = different collection = zero data bleed.
- Same PDF uploaded again = collection already exists -> skip re-embedding.
- Local HuggingFace embeddings (all-MiniLM-L6-v2) run on CPU at zero cost.
- pdfplumber table-aware extraction keeps values like "$399,000,000" attached
  to their row labels (e.g. "Class A-1 Notes") within the same chunk.
"""

import os
import hashlib
import logging
from typing import Optional, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

CHROMA_DB_ROOT = os.environ.get("CLO_CHROMA_ROOT", "./data/vector_store")

_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _EMBEDDINGS


def _collection_name_for_pdf(pdf_path: str) -> str:
    basename = os.path.basename(pdf_path)
    safe_stem = "".join(c if c.isalnum() else "_" for c in os.path.splitext(basename)[0])[:24]
    short_hash = hashlib.sha256(basename.encode()).hexdigest()[:12]
    return f"{safe_stem}_{short_hash}"


def _persist_dir_for_collection(collection_name: str) -> str:
    return os.path.join(CHROMA_DB_ROOT, collection_name)


def _collection_exists(collection_name: str) -> bool:
    persist_dir = _persist_dir_for_collection(collection_name)
    chroma_db_file = os.path.join(persist_dir, "chroma.sqlite3")
    return os.path.isfile(chroma_db_file)


def _table_row_to_text(row: list) -> str:
    return " | ".join(str(cell or "").strip() for cell in row)


def _load_pdf_with_tables(pdf_path: str) -> List[Document]:
    import pdfplumber
    docs: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_texts: List[str] = []
            tables = page.extract_tables()

            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    if row and any(cell for cell in row):
                        rows.append(_table_row_to_text(row))
                if rows:
                    page_texts.append(f"[TABLE]\n" + "\n".join(rows))

            table_bboxes = [t.bbox for t in page.find_tables()] if page.find_tables() else []
            if table_bboxes:
                remaining = page
                for bbox in table_bboxes:
                    try:
                        remaining = remaining.outside_bbox(bbox)
                    except Exception:
                        pass
                prose = remaining.extract_text(x_tolerance=3, y_tolerance=3) or ""
            else:
                prose = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            if prose.strip():
                page_texts.append(prose.strip())

            if page_texts:
                combined = "\n\n".join(page_texts)
                docs.append(Document(page_content=combined, metadata={"page": page_num, "source": os.path.basename(pdf_path)}))

    logger.info(f"[RAG] pdfplumber extracted {len(docs)} pages.")
    return docs


def _load_pdf_fallback(pdf_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info(f"[RAG] PyPDFLoader extracted {len(docs)} pages (fallback).")
    return docs


def _load_pdf(pdf_path: str) -> List[Document]:
    try:
        docs = _load_pdf_with_tables(pdf_path)
        if docs:
            return docs
        logger.warning("[RAG] pdfplumber returned empty -- falling back to PyPDFLoader.")
    except Exception as e:
        logger.warning(f"[RAG] pdfplumber failed ({e}) -- falling back to PyPDFLoader.")
    return _load_pdf_fallback(pdf_path)


def get_or_create_retriever(pdf_path: str, k: int = 4):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    collection_name = _collection_name_for_pdf(pdf_path)
    persist_dir = _persist_dir_for_collection(collection_name)
    embeddings = _get_embeddings()

    if _collection_exists(collection_name):
        logger.info(f"[RAG] Reusing existing index '{collection_name}' for {os.path.basename(pdf_path)}")
        vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_dir)
    else:
        logger.info(f"[RAG] Building new index '{collection_name}' for {os.path.basename(pdf_path)}...")
        os.makedirs(persist_dir, exist_ok=True)
        docs = _load_pdf(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])
        splits = text_splitter.split_documents(docs)
        logger.info(f"[RAG] Split into {len(splits)} chunks.")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name=collection_name, persist_directory=persist_dir)
        logger.info(f"[RAG] Index '{collection_name}' built and persisted to {persist_dir}.")

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 5, "lambda_mult": 0.5})
    return retriever, collection_name


def search_indenture(retriever, query: str) -> str:
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "(No relevant content found for this query)"
    return "\n\n---\n\n".join(f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in relevant_docs)


def delete_collection(pdf_path: str) -> bool:
    import shutil
    import chromadb

    collection_name = _collection_name_for_pdf(pdf_path)
    persist_dir = _persist_dir_for_collection(collection_name)

    if os.path.isdir(persist_dir):
        if hasattr(chromadb.api.client.SharedSystemClient, "clear_system_cache"):
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        shutil.rmtree(persist_dir, ignore_errors=True)
        logger.info(f"[RAG] Deleted collection '{collection_name}' at {persist_dir}.")
        return True

    logger.info(f"[RAG] No collection found for '{collection_name}' -- nothing to delete.")
    return False


def get_collection_info(pdf_path: str) -> dict:
    collection_name = _collection_name_for_pdf(pdf_path)
    persist_dir = _persist_dir_for_collection(collection_name)
    exists = _collection_exists(collection_name)
    info = {"collection_name": collection_name, "persist_dir": persist_dir, "indexed": exists}
    if exists:
        db_file = os.path.join(persist_dir, "chroma.sqlite3")
        info["index_size_mb"] = round(os.path.getsize(db_file) / (1024 * 1024), 2)
    return info
