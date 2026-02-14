"""
RAG (Retrieval-Augmented Generation) service for semantic search.

This module provides functions to retrieve relevant context from the RAG chunks
stored in the database based on semantic similarity.
"""

import logging
import os
from typing import List

import requests
from django.db.models import Value
from pgvector.django import CosineDistance, VectorField

from core.models import RagChunk

logger = logging.getLogger(__name__)


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using Ollama API.

    Args:
        text: The text to embed

    Returns:
        List of float values representing the embedding vector

    Raises:
        RuntimeError: If embedding generation fails
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={"model": embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError(f"Invalid embedding response: {data}")
    return embedding


def get_rag_context(user_input: str, limit: int = 3) -> List[str]:
    """
    Retrieve relevant RAG context based on semantic similarity to user input.
    """

    query_embedding = get_embedding(user_input)

    # 🔑 CRITICAL: cast embedding properly
    query_vector = Value(query_embedding, output_field=VectorField())

    CHUNK_FETCH_MULTIPLIER = 2
    MAX_DISTANCE = 0.35  # optional but strongly recommended

    chunks = list(
        RagChunk.objects.filter(embedding__isnull=False)
        .annotate(distance=CosineDistance("embedding", query_vector))
        .filter(distance__lt=MAX_DISTANCE)
        .order_by("distance")
        .values("text")[: limit * CHUNK_FETCH_MULTIPLIER]
    )

    if not chunks:
        logger.warning("No RAG chunks found after similarity filtering")
        return []

    result = [c.get("text") for c in chunks]

    # logger.info(
    #     "RAG retrieved %s chunks (behavior=%s, content=%s)",
    #     len(result)
    # )

    return result
