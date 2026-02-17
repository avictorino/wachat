import logging
import os
from typing import List

from django.db.models import Value
from openai import OpenAI
from pgvector.django import CosineDistance, VectorField

from core.models import RagChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"


def get_embedding(text: str) -> List[float]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for RAG embeddings.")

    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    if not isinstance(embedding, list):
        raise RuntimeError("Invalid embedding response from OpenAI.")
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
            logger.warning(f"No RAG chunks found for theme '{theme_id}'.")
            return []

        query_embedding = get_embedding(query)
        query_vector = Value(query_embedding, output_field=VectorField(dimensions=3072))

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
