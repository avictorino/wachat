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

OLLAMA_CHAT_MODEL = "llama3:8b"

MIN_RAG_CHARS = 100
MAX_RAG_CHARS = 400

# ==========================
# CONVERSATION PROMPT
# ==========================

CONVERSATION_PROMPT = """
Leia o texto abaixo e identifique APENAS percepções humanas
que estejam EXPLICITAMENTE presentes ou diretamente implicadas no texto.

IMPORTANTE:
- NÃO interprete além do texto
- NÃO preencha lacunas
- NÃO crie sentido onde não existe
- Se o texto NÃO descreve experiência humana, NÃO invente uma

Esta conversa é apenas uma ETAPA TÉCNICA para gerar memória semântica (RAG).
Ela NÃO será exibida ao usuário.

========================
FORMATO OBRIGATÓRIO
========================
Use SOMENTE este formato.
Uma fala por linha.
Nada fora do formato.

USUARIO: texto
CONSELHEIRO: texto

========================
ESTRUTURA
========================
- No máximo 1 par (USUARIO + CONSELHEIRO)
- Cada fala deve conter UMA única ideia
- Frases curtas e diretas

========================
REGRAS PARA O USUARIO
========================
O USUARIO:
- Só pode expressar algo que esteja claramente ancorado no texto
- Não pode teorizar
- Não pode generalizar
- Não pode criar exemplos
- Não pode fazer perguntas abstratas
- Linguagem simples, factual

Exemplos válidos:
- O texto mostra que diferentes pessoas reagem de formas distintas
- Há contraste entre grupos descritos
- Nem todos respondem da mesma maneira

========================
REGRAS PARA O CONSELHEIRO
========================
O CONSELHEIRO:
- NÃO ensina
- NÃO orienta
- NÃO aconselha
- NÃO espiritualiza
- NÃO amplia
- NÃO conclui

O CONSELHEIRO deve:
- Reescrever a ideia do usuário de forma ainda MAIS neutra
- Eliminar qualquer abstração desnecessária
- Soar como uma nota técnica silenciosa
- Não usar "você", "nós" ou primeira pessoa
- Não usar verbos modais (pode, poderia, tende a)

Exemplos válidos:
- O texto descreve respostas diferentes entre indivíduos
- Há variação observável no comportamento relatado
- O conteúdo aponta diferenças de reação

========================
PROIBIÇÕES ABSOLUTAS
========================
NÃO use:
- Perguntas
- Metáforas
- Linguagem emocional
- Interpretação psicológica
- Termos vagos como: sentido, confiança, confusão, percepção

========================
IMPORTANTE
========================
- Se o texto for apenas técnico, bibliográfico ou estrutural,
  responda com UMA fala curta e neutra, sem humanização excessiva.
- Se não houver percepção humana clara, gere o mínimo possível.

Texto:
"""


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


def extract_blocks(pdf_path: str) -> Iterator[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            raw = block[4]
            if raw and raw.strip():
                cleaned = repair_text(raw)
                if cleaned:
                    yield page_number, cleaned


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


def is_reference_like(text: str) -> bool:
    return (
        len(text) < MIN_RAG_CHARS
        and re.search(r"\b(19|20)\d{2}\b", text)
        and "," in text
    )


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

    text = re.sub(r"\s+", " ", text).strip()
    return text


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
# OLLAMA CALLS
# ==========================


def generate_conversation(text: str, ollama_url: str) -> List[List[dict]]:
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "prompt": CONVERSATION_PROMPT + text,
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
    help = "Granular RAG ingestion with semantic hash IDs"

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
