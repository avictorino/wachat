"""
RAG (Retrieval-Augmented Generation) service for semantic search.

This module provides functions to retrieve relevant context from the RAG chunks
stored in the database based on semantic similarity.
"""

import os
from typing import List

import requests
from django.db.models import Value
from pgvector.django import CosineDistance, VectorField

from core.models import RagChunk


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


class RAGService:
    CHUNK_FETCH_MULTIPLIER = 2
    MAX_DISTANCE = 0.35

    def retrieve(self, query: str, theme_id: str, limit: int = 3) -> List[str]:
        """
        Retrieve relevant RAG context by similarity, constrained to one theme.
        """
        chunks_queryset = RagChunk.objects.filter(
            theme_id=theme_id,
            embedding__isnull=False,
        )
        if not chunks_queryset.exists():
            raise RuntimeError(f"No RAG chunks found for theme '{theme_id}'.")

        query_embedding = get_embedding(query)
        query_vector = Value(query_embedding, output_field=VectorField())

        chunks = list(
            chunks_queryset.annotate(distance=CosineDistance("embedding", query_vector))
            .filter(distance__lt=self.MAX_DISTANCE)
            .order_by("distance")
            .values("text")[: limit * self.CHUNK_FETCH_MULTIPLIER]
        )

        if not chunks:
            raise RuntimeError(
                f"No RAG chunks found after similarity filtering for theme '{theme_id}'."
            )

        return [chunk.get("text") for chunk in chunks]
