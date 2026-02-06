import hashlib
import os
import re
from typing import Iterator, List, Tuple

import dotenv
import fitz  # PyMuPDF
import requests
from django.core.management.base import BaseCommand, CommandError

from config.settings import BASE_DIR
from core.models import RagChunk

# ==========================
# ENV / CONFIG
# ==========================

dotenv.read_dotenv(BASE_DIR)

DEFAULT_PDF_PATH = "/Users/avictorino/Projects/wachat/model/pdfs"

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768

# 🔥 modelo RAG dedicado
OLLAMA_RAG_CHAT_MODEL = "wachat-rag-v1"

MIN_RAG_CHARS = 100
MAX_RAG_CHARS = 400

# ==========================
# TEXT CLEANUP
# ==========================


def generate_rag_id(source: str, page: int, rag_text: str) -> str:
    base = f"{source}|p{page}|{rag_text.strip().lower()}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"{source}:p{page}:{digest}"


def repair_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"(\w+)-?\n(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


# ==========================
# PDF EXTRACTION
# ==========================


def extract_blocks(pdf_path: str) -> Iterator[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            raw = block[4]
            if raw and raw.strip():
                cleaned = repair_text(raw)
                if cleaned:
                    yield page_number, cleaned


# ==========================
# PARSING MODEL OUTPUT
# ==========================


def parse_conversation(text: str) -> List[List[dict]]:
    conversations, current = [], []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("USUARIO:"):
            if current:
                conversations.append(current)
                current = []
            current.append({"role": "usuario", "text": line[8:].strip()})

        elif line.startswith("CONSELHEIRO:"):
            current.append({"role": "conselheiro", "text": line[12:].strip()})

    if current:
        conversations.append(current)

    return conversations


# ==========================
# FILTERS
# ==========================


def is_reference_like(text: str) -> bool:
    return (
        len(text) < MIN_RAG_CHARS
        and re.search(r"\b(19|20)\d{2}\b", text)
        and "," in text
    )


TITLE_OR_INDEX_PATTERN = re.compile(
    r"""
    (
        ^[IVXLCDM]{1,5}\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ] |
        ^\d{1,3}\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ] |
        ^[A-Z][a-z]{1,20}\s[A-Z][a-z]?,\s |
        ^[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÇ\s/]{3,60}$
    )
    """,
    re.VERBOSE,
)


def is_title_or_index(text: str) -> bool:
    t = text.strip()
    if len(t) < 10 or len(t) > 80:
        return False
    return bool(TITLE_OR_INDEX_PATTERN.match(t))


# ==========================
# NORMALIZATION (Q/A → RAG)
# ==========================


def normalize_qa(user_text: str, counselor_text: str) -> str:
    text = f"{user_text}. {counselor_text}"

    text = re.sub(r"\?", ".", text)
    text = re.sub(r"\b(você|vc|te|seu|sua)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(ah|né|então|eu sinto que|parece que|soa como se)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", text).strip()


# ==========================
# OLLAMA CALLS
# ==========================


def generate_conversation(text: str, ollama_url: str) -> List[List[dict]]:
    """
    Envia SOMENTE o texto cru.
    Todo o comportamento vem do Modelfile (wachat-rag-v1).
    """
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": OLLAMA_RAG_CHAT_MODEL,
            "prompt": text,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()

    raw = resp.json().get("response", "")
    return parse_conversation(raw)


def embed_text(text: str, ollama_url: str) -> List[float]:
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text,
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()

    return resp.json().get("embedding")


# ==========================
# DJANGO COMMAND
# ==========================


class Command(BaseCommand):
    help = "Granular RAG ingestion using dedicated RAG model (semantic hash IDs)"

    def add_arguments(self, parser):
        parser.add_argument("--ollama-url", default="http://localhost:11434")

    def handle(self, *args, **options):
        ollama_url = options["ollama_url"]

        if not os.path.isdir(DEFAULT_PDF_PATH):
            raise CommandError("DEFAULT_PDF_PATH must be a directory")

        for pdf_file in os.listdir(DEFAULT_PDF_PATH):
            if not pdf_file.lower().endswith(".pdf"):
                continue

            source = os.path.splitext(pdf_file)[0]
            pdf_path = os.path.join(DEFAULT_PDF_PATH, pdf_file)

            self.stdout.write(self.style.NOTICE(f"Processing {pdf_file}"))

            for page, block in extract_blocks(pdf_path):
                if (
                    page <= 10
                    or is_reference_like(block)
                    or is_title_or_index(block)
                    or len(block) < MIN_RAG_CHARS
                ):
                    continue

                conversations = generate_conversation(block, ollama_url)

                for convo in conversations:
                    if len(convo) != 2:
                        continue

                    u, c = convo
                    rag_text = normalize_qa(u["text"], c["text"])

                    if not (MIN_RAG_CHARS <= len(rag_text) <= MAX_RAG_CHARS):
                        continue

                    rag_id = generate_rag_id(source, page, rag_text)

                    if RagChunk.objects.filter(id=rag_id).exists():
                        continue

                    embedding = embed_text(rag_text, ollama_url)

                    RagChunk.objects.create(
                        id=rag_id,
                        text=rag_text,
                        raw_text=block,
                        conversations=convo,
                        source=source,
                        page=page,
                        chunk_index=0,
                        type="conversation_pair",
                        embedding=embedding,
                    )

                    self.stdout.write(self.style.SUCCESS(f"Saved {rag_id}"))

        self.stdout.write(self.style.SUCCESS("RAG ingestion completed"))
