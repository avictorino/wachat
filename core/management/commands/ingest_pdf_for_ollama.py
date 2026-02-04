import json
import os
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import dotenv
import fitz  # PyMuPDF
import psycopg
import requests
from django.core.management.base import BaseCommand, CommandError
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config.settings import BASE_DIR

# ==========================
# ENV / CONFIG
# ==========================

dotenv.read_dotenv(os.path.join(BASE_DIR, ".env"))

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
# DATA MODEL
# ==========================


@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    embedding: Optional[List[float]] = None


# ==========================
# JSONL OUTPUT
# ==========================


def write_jsonl(records: List[ChunkRecord], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "text": r.text,
                        "metadata": {
                            "source": r.source,
                            "page": r.page,
                            "chunk_index": r.chunk_index,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ==========================
# POSTGRES + PGVECTOR
# ==========================


def get_pg_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise CommandError("DATABASE_URL is not set")

    conn = psycopg.connect(db_url, row_factory=dict_row)
    register_vector(conn)
    return conn


def ensure_postgres_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIMENSION})
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
            ON rag_chunks
            USING hnsw (embedding vector_cosine_ops)
            """
        )

    conn.commit()


def upsert_chunks(conn, records: List[ChunkRecord]):
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO rag_chunks (id, source, page, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
            """,
            [
                (
                    r.id,
                    r.source,
                    r.page,
                    r.chunk_index,
                    r.text,
                    r.embedding,
                )
                for r in records
            ],
        )
    conn.commit()


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

        conn = get_pg_conn()
        ensure_postgres_schema(conn)

        pdf_files = [
            f for f in os.listdir(DEFAULT_PDF_PATH) if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            raise CommandError("No PDF files found")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(DEFAULT_PDF_PATH, pdf_file)
            source = os.path.splitext(pdf_file)[0]

            self.stdout.write(self.style.NOTICE(f"Processing: {pdf_file}"))

            records: List[ChunkRecord] = []

            for page, block in extract_semantic_blocks(pdf_path):
                if page <= 10:
                    continue

                for ci, chunk in enumerate(
                    sentence_chunk(block, options["chunk_size"])
                ):
                    if should_skip_chunk(chunk):
                        continue

                    records.append(
                        ChunkRecord(
                            id=f"{source}:p{page}:c{ci}",
                            text=chunk,
                            source=source,
                            page=page,
                            chunk_index=ci,
                        )
                    )

            if not records:
                self.stdout.write(self.style.WARNING(f"No valid chunks in {pdf_file}"))
                continue

            jsonl_path = os.path.join(DEFAULT_OUT_DIR, f"{source}.jsonl")
            write_jsonl(records, jsonl_path)

            if options["embed"]:
                for i, r in enumerate(records, 1):
                    r.embedding = ollama_embed(
                        r.text,
                        options["ollama_url"],
                        options["embed_model"],
                    )
                    if i % 50 == 0:
                        self.stdout.write(f"  Embeddings: {i}/{len(records)}")

            upsert_chunks(conn, records)

            self.stdout.write(
                self.style.SUCCESS(f"{pdf_file}: {len(records)} chunks ingested")
            )

        conn.close()
        self.stdout.write(self.style.SUCCESS("All PDFs processed successfully"))
