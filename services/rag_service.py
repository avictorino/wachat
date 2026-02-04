"""
RAG (Retrieval-Augmented Generation) service for semantic search.

This module provides functions to retrieve relevant context from the RAG chunks
stored in the database based on semantic similarity.
"""

import logging
import os
from typing import List

import requests
from django.db.models import F
from pgvector.django import CosineDistance

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

    try:
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

    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")


def get_rag_context(user_input: str, limit: int = 5) -> List[str]:
    """
    Retrieve relevant RAG context based on semantic similarity to user input.

    This function:
    1. Generates an embedding for the user input
    2. Queries RagChunk ordered by cosine distance
    3. Prefers type="behavior" when available
    4. Returns only the raw text strings

    Args:
        user_input: The user's message text
        limit: Maximum number of chunks to retrieve (default: 5)

    Returns:
        List of text strings from the most relevant chunks
    """
    try:
        # Generate embedding for user input
        query_embedding = get_embedding(user_input)

        # Query chunks ordered by cosine distance
        # We'll fetch more chunks initially to allow for preference filtering
        # Fetch 2× the limit to ensure we have enough chunks after filtering by type
        CHUNK_FETCH_MULTIPLIER = 2
        chunks = list(
            RagChunk.objects.annotate(
                distance=CosineDistance("embedding", query_embedding)
            )
            .order_by("distance")
            .values("text", "type", "distance")[:limit * CHUNK_FETCH_MULTIPLIER]
        )

        if not chunks:
            logger.warning("No RAG chunks found in database")
            return []

        # Prefer behavior chunks when available
        behavior_chunks = [c for c in chunks if c["type"] == "behavior"]
        content_chunks = [c for c in chunks if c["type"] == "content"]

        # Take up to limit chunks, prioritizing behavior
        selected_chunks = []
        
        # Add behavior chunks first (up to limit)
        selected_chunks.extend(behavior_chunks[:limit])
        
        # Fill remaining slots with content chunks if needed
        remaining = limit - len(selected_chunks)
        if remaining > 0:
            selected_chunks.extend(content_chunks[:remaining])

        # Extract just the text
        context_texts = [chunk["text"] for chunk in selected_chunks]

        logger.info(
            f"Retrieved {len(context_texts)} RAG chunks "
            f"({len([c for c in selected_chunks if c['type'] == 'behavior'])} behavior, "
            f"{len([c for c in selected_chunks if c['type'] == 'content'])} content)"
        )

        return context_texts

    except Exception as e:
        logger.error(f"Error retrieving RAG context: {str(e)}", exc_info=True)
        # Return empty list on error to allow conversation to continue
        return []
