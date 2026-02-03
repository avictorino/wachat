"""
Service for processing PDF documents and generating embeddings for RAG.

This module handles the complete pipeline:
1. Extract text from PDF
2. Chunk text into manageable pieces
3. Generate embeddings using sentence-transformers
4. Store embeddings in ChromaDB vector database
"""

import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =========================
# Configuration
# =========================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "knowledge_documents"

# Lazy-loaded globals to avoid loading models at import time
_embedder = None
_collection = None


def _get_embedder():
    """Get or create the embedding model (lazy initialization)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_collection():
    """Get or create the ChromaDB collection (lazy initialization)."""
    global _collection
    if _collection is None:
        chroma_client = chromadb.Client()
        _collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


# =========================
# Main function
# =========================


def embed_pdf_document(file_path: str):
    """
    Pipeline for RAG indexing:
    1. Extract text from PDF
    2. Chunk text
    3. Generate embeddings
    4. Store in vector database

    Args:
        file_path: Absolute path to the PDF file

    Raises:
        ValueError: If PDF contains no extractable text
    """

    # 1. Extract text
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise ValueError("PDF does not contain extractable text")

    # 2. Chunking
    chunks = chunk_text(text)

    # 3. Generate embeddings
    embeddings = generate_embeddings(chunks)

    # 4. Store in vector database
    store_embeddings(chunks=chunks, embeddings=embeddings, source=file_path)

    print(f"✅ PDF indexed successfully: {file_path}")


# =========================
# 1. Text extraction
# =========================


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from all pages of a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Concatenated text from all pages
    """
    reader = PdfReader(file_path)
    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n".join(pages_text)


# =========================
# 2. Chunking
# =========================


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Divide text into smaller chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


# =========================
# 3. Embeddings
# =========================


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Generate embeddings for each chunk.

    Args:
        chunks: List of text chunks

    Returns:
        List of embedding vectors
    """
    embedder = _get_embedder()
    return embedder.encode(chunks).tolist()


# =========================
# 4. Vector storage
# =========================


def store_embeddings(
    *, chunks: list[str], embeddings: list[list[float]], source: str
):
    """
    Store embeddings in ChromaDB.

    Args:
        chunks: Text chunks
        embeddings: Embedding vectors
        source: Source file path for metadata
    """

    collection = _get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source, "chunk_index": idx} for idx in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
