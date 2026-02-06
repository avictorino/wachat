import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import dotenv
import fitz  # PyMuPDF
import requests
from django.core.management.base import BaseCommand, CommandError
from django.db import IntegrityError, close_old_connections, transaction

from config.settings import BASE_DIR
from core.models import RagChunk

# ==========================
# ENV / CONFIG
# ==========================

dotenv.read_dotenv(BASE_DIR)

DEFAULT_PDF_PATH = "/Users/avictorino/Projects/wachat/model/pdfs"

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768  # Keep for reference

# Dedicated RAG chat model
OLLAMA_RAG_CHAT_MODEL = "wachat-rag-v1"

MIN_RAG_CHARS = 100
MAX_RAG_CHARS = 400

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_WORKERS = 2

# Per-thread HTTP session (avoid sharing sessions across threads)
_thread_local = threading.local()


def get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


# ==========================
# DATA STRUCTURES
# ==========================


@dataclass(frozen=True)
class BlockTask:
    source: str
    pdf_file: str
    page: int
    block: str


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
    try:
        for page_number, page in enumerate(doc, start=1):
            for block in page.get_text("blocks"):
                raw = block[4]
                if raw and raw.strip():
                    cleaned = repair_text(raw)
                    if cleaned:
                        yield page_number, cleaned
    finally:
        doc.close()


# ==========================
# PARSING MODEL OUTPUT
# ==========================


def parse_conversation(text: str) -> List[List[dict]]:
    conversations: List[List[dict]] = []
    current: List[dict] = []

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
# NORMALIZATION (Q/A -> RAG)
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


def generate_conversation(raw_text: str, ollama_url: str) -> List[List[dict]]:
    """
    Sends ONLY raw text.
    All behavior comes from the Modelfile (wachat-rag-v1).
    """
    session = get_session()
    resp = session.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": OLLAMA_RAG_CHAT_MODEL,
            "prompt": raw_text,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()

    raw = resp.json().get("response", "")
    return parse_conversation(raw)


def embed_text(text: str, ollama_url: str) -> List[float]:
    session = get_session()
    resp = session.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text,
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()

    embedding = resp.json().get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Invalid embedding response from Ollama")
    return embedding


# ==========================
# WORKER
# ==========================


def process_block_task(task: BlockTask, ollama_url: str) -> int:
    """
    Returns number of chunks saved for this block.
    Thread-safe: manages DB connections per thread and handles duplicates via IntegrityError.
    """
    close_old_connections()
    saved = 0

    try:
        conversations = generate_conversation(task.block, ollama_url)

        for convo in conversations:
            if len(convo) != 2:
                continue

            u, c = convo
            rag_text = normalize_qa(u.get("text", ""), c.get("text", ""))

            if not (MIN_RAG_CHARS <= len(rag_text) <= MAX_RAG_CHARS):
                continue

            rag_id = generate_rag_id(task.source, task.page, rag_text)

            # Compute embedding before DB write to reduce open transaction time.
            embedding = embed_text(rag_text, ollama_url)

            try:
                with transaction.atomic():
                    RagChunk.objects.create(
                        id=rag_id,
                        text=rag_text,
                        raw_text=task.block,
                        conversations=convo,
                        source=task.source,
                        page=task.page,
                        chunk_index=0,
                        type="conversation_pair",
                        embedding=embedding,
                    )
                saved += 1
            except IntegrityError:
                # Duplicate id, safe to ignore in parallel runs
                continue

        return saved

    finally:
        close_old_connections()


# ==========================
# DJANGO COMMAND
# ==========================


class Command(BaseCommand):
    help = "Granular RAG ingestion using dedicated RAG model (thread-safe, semantic hash IDs)"

    def add_arguments(self, parser):
        parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
        parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        parser.add_argument("--max-pages", type=int, default=0, help="0 = no limit")
        parser.add_argument("--skip-pages", type=int, default=10)

    def handle(self, *args, **options):
        ollama_url: str = options["ollama_url"]
        workers: int = max(1, int(options["workers"]))
        max_pages: int = int(options["max_pages"])
        skip_pages: int = int(options["skip_pages"])

        if not os.path.isdir(DEFAULT_PDF_PATH):
            raise CommandError("DEFAULT_PDF_PATH must be a directory")

        pdf_files = [
            f for f in os.listdir(DEFAULT_PDF_PATH) if f.lower().endswith(".pdf")
        ]
        if not pdf_files:
            raise CommandError("No PDF files found")

        total_saved = 0
        total_tasks = 0

        for pdf_file in sorted(pdf_files):
            source = os.path.splitext(pdf_file)[0]
            pdf_path = os.path.join(DEFAULT_PDF_PATH, pdf_file)

            self.stdout.write(self.style.NOTICE(f"Processing {pdf_file}"))

            # Build tasks in main thread (safe)
            tasks: List[BlockTask] = []
            pages_seen = 0

            for page, block in extract_blocks(pdf_path):
                if page <= skip_pages:
                    continue

                pages_seen += 1
                if max_pages and pages_seen > max_pages:
                    break

                if len(block) < MIN_RAG_CHARS:
                    continue
                if is_reference_like(block):
                    continue
                if is_title_or_index(block):
                    continue

                tasks.append(
                    BlockTask(source=source, pdf_file=pdf_file, page=page, block=block)
                )

            if not tasks:
                self.stdout.write(self.style.WARNING(f"No tasks for {pdf_file}"))
                continue

            self.stdout.write(
                self.style.NOTICE(
                    f"Queued {len(tasks)} blocks for {pdf_file} (workers={workers})"
                )
            )
            total_tasks += len(tasks)

            # Process tasks in parallel
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(process_block_task, t, ollama_url) for t in tasks
                ]

                for fut in as_completed(futures):
                    try:
                        saved = fut.result()
                        if saved:
                            total_saved += saved
                    except Exception as e:
                        # Hard block behavior: fail fast if desired
                        raise CommandError(f"Worker failed: {e}") from e

            self.stdout.write(
                self.style.SUCCESS(
                    f"Finished {pdf_file}. Total saved so far: {total_saved}"
                )
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"RAG ingestion completed. Blocks processed: {total_tasks}, chunks saved: {total_saved}"
            )
        )
