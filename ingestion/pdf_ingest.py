"""
PDF ingestion pipeline: load -> chunk -> embed -> store in ChromaDB.

Run directly:
    python -m ingestion.pdf_ingest path/to/file.pdf
"""

import os
import sys

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "rag_docs"

# Chunk size chosen so each chunk fits comfortably in the LLM context window
# while still being semantically coherent. Overlap prevents answers being split
# across chunk boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _get_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Open (or create) the persisted ChromaDB collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest(pdf_path: str) -> int:
    """
    Load a PDF, split it into chunks, embed each chunk, and upsert into ChromaDB.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Number of chunks stored.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Load — PyPDFLoader yields one Document per page, with page metadata.
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[ingest] Loaded {len(pages)} pages from '{pdf_path}'")

    # 2. Chunk — split pages into overlapping text windows.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"[ingest] Split into {len(chunks)} chunks")

    # 3. Tag each chunk with the source filename so retrieval can filter by document.
    basename = os.path.basename(pdf_path)
    for chunk in chunks:
        chunk.metadata["source_document"] = basename

    # 4. Embed + store — accumulates alongside any previously ingested docs this session.
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)
    vectorstore.add_documents(chunks)
    print(f"[ingest] Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PERSIST_DIR}'")

    return len(chunks)


def reset_collection() -> None:
    """
    Delete all documents from ChromaDB.
    Called on page load so each browser session starts with a clean slate.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        client.delete_collection(COLLECTION_NAME)
        print(f"[reset] Collection '{COLLECTION_NAME}' deleted")
    except Exception:
        # Collection may not exist yet on first ever load — that's fine
        print(f"[reset] Collection '{COLLECTION_NAME}' did not exist, nothing to clear")


def get_retriever(k: int = 4, source_document: str | None = None) -> Chroma:
    """
    Return a LangChain retriever backed by the persisted ChromaDB collection.

    Uses MMR (Maximal Marginal Relevance) to retrieve k diverse, relevant chunks.
    fetch_k candidates are scored first; MMR selects k from those for diversity.

    Args:
        k:               Number of chunks to return per query.
        source_document: If set, restrict retrieval to chunks from this filename only.
    """
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)
    search_kwargs: dict = {"k": k, "fetch_k": max(k * 5, 20)}
    if source_document:
        search_kwargs["filter"] = {"source_document": source_document}
        print(f"[retriever] source_document filter={source_document!r} k={k} fetch_k={search_kwargs['fetch_k']}")
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


def list_documents() -> list[str]:
    """
    Return distinct source_document values stored in ChromaDB.

    Uses the raw chromadb client (no embedding model needed) so this is fast.
    Returns an empty list if the collection does not yet exist.
    """
    print(f"[list_documents] path={os.path.abspath(CHROMA_PERSIST_DIR)!r} collection={COLLECTION_NAME!r}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        print(f"[list_documents] could not open collection: {type(exc).__name__}: {exc}")
        return []
    result = collection.get(include=["metadatas"])
    metas = result.get("metadatas", [])
    seen: set[str] = set()
    for meta in metas:
        if meta and "source_document" in meta:
            seen.add(meta["source_document"])
    print(f"[list_documents] chunks={len(metas)}, distinct docs={len(seen)}: {sorted(seen)}")
    return sorted(seen)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.pdf_ingest <path/to/file.pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
