import json
import os
import re
from typing import Iterator, List, Tuple

import fitz  # PyMuPDF
import requests
from django.core.management.base import BaseCommand, CommandError

from config.settings import BASE_DIR
from core.models import RagChunk

# ==========================
# ENV / CONFIG
# ==========================

DEFAULT_PDF_PATH = f"{BASE_DIR}/model/pdfs"
DEFAULT_OUT_DIR = f"{BASE_DIR}/model/rag"

EMBEDDING_DIMENSION = 768  # nomic-embed-text


# ==========================
# TEXT CLEANUP
# ==========================


def repair_text_artifacts(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"(\w+)-?\n(\w+)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ==========================
# SEMANTIC EXTRACTION
# ==========================


def extract_semantic_blocks(pdf_path: str) -> Iterator[Tuple[int, str]]:
    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            raw = block[4]
            if not raw or not raw.strip():
                continue

            cleaned = repair_text_artifacts(raw)
            if cleaned:
                yield page_number, cleaned


# ==========================
# SENTENCE-BASED CHUNKING
# ==========================


def sentence_chunk(
    text: str,
    max_chars: int,
    overlap_sentences: int = 1,
) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[str] = []
    buffer: List[str] = []

    for s in sentences:
        if not s.strip():
            continue

        current_len = sum(len(x) for x in buffer)

        if current_len + len(s) <= max_chars:
            buffer.append(s)
        else:
            chunk = " ".join(buffer).strip()
            if chunk:
                chunks.append(chunk)

            buffer = buffer[-overlap_sentences:] if overlap_sentences else []
            buffer.append(s)

    if buffer:
        chunks.append(" ".join(buffer).strip())

    return chunks


# ==========================
# STRUCTURAL FILTERING
# ==========================


def should_skip_chunk(text: str) -> bool:
    upper = text.upper()
    return (
        len(text) < 300
        or "ISBN" in upper
        or "©" in text
        or "ÍNDICE" in upper
        or "AUTORES" in upper
        or text.isupper()
        or re.match(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ ]+$", text)
    )


# ==========================
# CHUNK CLASSIFICATION
# ==========================


def classify_chunk_type(text: str) -> str:
    """
    Classify chunk as 'behavior' or 'content'.

    Behavior: content about posture, tone, guidance, care, relationship
    Content: informational or doctrinal content

    Args:
        text: The chunk text to classify

    Returns:
        "behavior" or "content"
    """
    text_lower = text.lower()

    # Keywords that indicate behavioral/guidance content
    behavior_keywords = [
        "postura",
        "tom",
        "cuidado",
        "relacionamento",
        "acolhimento",
        "empatia",
        "escuta",
        "presença",
        "acompanhamento",
        "orientação",
        "guia",
        "direção",
        "conselho",
        "apoio",
        "sustentação",
        "companhia",
        "proximidade",
        "atenção",
        "sensibilidade",
        "discernimento",
        "sabedoria pastoral",
        "pastoral",
        "ministério",
    ]

    # Check for behavior keywords
    keyword_count = sum(1 for keyword in behavior_keywords if keyword in text_lower)

    # If multiple behavior keywords found, classify as behavior
    if keyword_count >= 2:
        return "behavior"

    # Default to content
    return "content"


# ==========================
# JSONL OUTPUT
# ==========================


def write_jsonl(chunks: List[RagChunk], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": {
                            "source": chunk.source,
                            "page": chunk.page,
                            "chunk_index": chunk.chunk_index,
                            "type": chunk.type,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ==========================
# OLLAMA EMBEDDINGS
# ==========================


def ollama_embed(text: str, ollama_url: str, embed_model: str) -> List[float]:
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


# ==========================
# DJANGO COMMAND
# ==========================


class Command(BaseCommand):
    help = "Batch semantic PDF ingestion for RAG (directory-based)"

    def add_arguments(self, parser):
        parser.add_argument("--chunk-size", type=int, default=900)
        parser.add_argument("--embed", action="store_true")
        parser.add_argument("--ollama-url", default="http://localhost:11434")
        parser.add_argument("--embed-model", default="nomic-embed-text")

    def handle(self, *args, **options):
        if not os.path.isdir(DEFAULT_PDF_PATH):
            raise CommandError("DEFAULT_PDF_PATH must be a directory")

        pdf_files = [
            f for f in os.listdir(DEFAULT_PDF_PATH) if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            raise CommandError("No PDF files found")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(DEFAULT_PDF_PATH, pdf_file)
            source = os.path.splitext(pdf_file)[0]

            self.stdout.write(self.style.NOTICE(f"Processing: {pdf_file}"))

            chunks_to_create = []

            for page, block in extract_semantic_blocks(pdf_path):
                if page <= 10:
                    continue

                for ci, chunk_text in enumerate(
                    sentence_chunk(block, options["chunk_size"])
                ):
                    if should_skip_chunk(chunk_text):
                        continue

                    chunk_id = f"{source}:p{page}:c{ci}"
                    chunk_type = classify_chunk_type(chunk_text)

                    # Create RagChunk instance (not yet saved)
                    chunk = RagChunk(
                        id=chunk_id,
                        text=chunk_text,
                        source=source,
                        page=page,
                        chunk_index=ci,
                        type=chunk_type,
                        embedding=[0.0] * EMBEDDING_DIMENSION,  # Placeholder
                    )
                    chunks_to_create.append(chunk)

            if not chunks_to_create:
                self.stdout.write(self.style.WARNING(f"No valid chunks in {pdf_file}"))
                continue

            # Generate embeddings if requested
            if options["embed"]:
                for i, chunk in enumerate(chunks_to_create, 1):
                    chunk.embedding = ollama_embed(
                        chunk.text,
                        options["ollama_url"],
                        options["embed_model"],
                    )
                    if i % 50 == 0:
                        self.stdout.write(
                            f"  Embeddings: {i}/{len(chunks_to_create)}"
                        )

            # Use bulk_create for efficiency
            # For upsert behavior, delete existing chunks from this source first
            RagChunk.objects.filter(source=source).delete()
            RagChunk.objects.bulk_create(chunks_to_create, batch_size=500)

            # Write JSONL output
            jsonl_path = os.path.join(DEFAULT_OUT_DIR, f"{source}.jsonl")
            write_jsonl(chunks_to_create, jsonl_path)

            self.stdout.write(
                self.style.SUCCESS(
                    f"{pdf_file}: {len(chunks_to_create)} chunks ingested"
                )
            )

        self.stdout.write(self.style.SUCCESS("All PDFs processed successfully"))
