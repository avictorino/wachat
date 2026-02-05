import json
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
DEFAULT_OUT_DIR = "/Users/avictorino/Projects/wachat/model/rag"

EMBEDDING_DIMENSION = 768  # nomic-embed-text

# ==========================
# CONVERSATIONAL PROMPT
# ==========================

CONVERSATION_PROMPT = """
Transforme o texto abaixo em uma conversa natural, humana e imperfeita entre duas pessoas reais.

O OBJETIVO DESTA CONVERSA
- Esta conversa NÃO é uma resposta final ao usuário
- Ela será usada como material de indexação para RAG
- Portanto, ela DEVE preservar as IDEIAS CENTRAIS do texto
- Use linguagem humana, mas mantenha pistas claras do conteúdo original

========================
FORMATO OBRIGATÓRIO
========================
- Use SOMENTE este formato
- Uma fala por linha
- Não escreva nada fora do formato

FORMATO EXATO:
USUARIO: texto
CONSELHEIRO: texto

Exemplo (apenas formato):
USUARIO: …
CONSELHEIRO: …
USUARIO: …
CONSELHEIRO: …

========================
REGRAS DE LINGUAGEM
========================
- linguagem falada, cotidiana e simples
- frases imperfeitas, com hesitação
- respostas parciais
- uma única ideia por fala
- evite abstrações vagas demais
- prefira imagens simples e concretas

Exemplos de imagens permitidas:
- tudo foi ficando estreito
- sobrou só isso
- o resto foi ficando de lado
- sempre igual, dia após dia
- parece que só gira nisso

========================
REGRAS PARA O USUARIO
========================
O usuário:
- NÃO faz perguntas analíticas ou gerais
- NÃO pergunta "como", "por que", "o que fazer"
- NÃO pede explicações
- NÃO teoriza
- fala a partir da vivência e da percepção
- expressa sensação, confusão, desgaste, repetição, perda de espaço
- usa imagens simples ligadas ao texto
- pode repetir palavras
- pode mencionar fé apenas como sentimento pessoal

========================
REGRAS PARA O CONSELHEIRO
========================
O conselheiro:
- representa presença, escuta e eco emocional
- NÃO orienta
- NÃO aconselha
- NÃO explica
- NÃO ensina
- NÃO conclui
- NÃO fecha a conversa
- NÃO promete ajuda
- NÃO diz o que deve ser feito
- NÃO faz perguntas
- NÃO organiza o pensamento do usuário

O conselheiro DEVE:
- reagir apenas ao que foi dito
- ecoar as MESMAS imagens e palavras do usuário
- manter o foco nas ideias trazidas
- refletir sentimento E percepção, sem explicar
- usar expressões de leveza e incerteza, como:
  parece, soa, dá a impressão, fica a sensação

O conselheiro pode falar um pouco mais que o usuário,
mas nunca para explicar ou interpretar causas.

========================
PROIBIÇÕES ABSOLUTAS
========================
NÃO use:
- termos técnicos, acadêmicos ou institucionais
- linguagem clínica, terapêutica ou moral
- explicações causais
- conclusões
- abstrações genéricas como apenas:
  tristeza, desespero, dor profunda

NÃO use as palavras:
importante, precisa, deve, ajuda, tratar, superar,
caminho, esperança, processo, jornada, solução

========================
IMPORTANTE
========================
- Esta conversa deve continuar aberta
- Não resolva nada
- Não explique o texto original
- Não se afaste das ideias centrais do texto
- Se quebrar qualquer regra, a resposta será descartada
- Não use listas
- Não use aspas
- Não use markdown

========================
Texto:
"""

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


def sentence_chunk(text: str, max_chars: int) -> List[str]:
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
            buffer = [s]

    if buffer:
        chunks.append(" ".join(buffer).strip())

    return chunks


# ==========================
# STRUCTURAL FILTERING
# ==========================


def should_skip_chunk(text: str) -> bool:
    upper = text.upper()
    return (
        len(text) < 100
        or "ISBN" in upper
        or "©" in text
        or "ÍNDICE" in upper
        or "AUTORES" in upper
        or text.isupper()
        or re.match(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ ]+$", text)
    )


# ==========================
# OLLAMA – CONVERSATION
# ==========================


def ollama_generate_conversations(
    text: str,
    ollama_url: str,
) -> Tuple[list[list[dict]], List[float]]:

    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": CONVERSATION_PROMPT + text,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()

    raw = resp.json().get("response", "")

    conversations = parse_conversation_lines(raw)

    if not isinstance(conversations, list):
        raise RuntimeError(f"Missing conversations array: {raw}")

    for convo in conversations:
        if not isinstance(convo, list):
            raise RuntimeError(f"Conversation is not a list: {convo}")

        for turn in convo:
            if not isinstance(turn, dict) or "role" not in turn or "text" not in turn:
                raise RuntimeError(f"Invalid turn format: {turn}")

    embed_input = " ".join(turn["text"] for convo in conversations for turn in convo)
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": embed_input,
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    embedding = data.get("embedding")

    if not isinstance(embedding, list):
        raise RuntimeError(f"Invalid embedding response: {data}")

    return conversations, embedding


# ==========================
# OLLAMA – EMBEDDINGS
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


def parse_conversation_lines(text: str) -> list[list[dict]]:
    conversations = []
    current = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("USUARIO:"):
            if current:
                conversations.append(current)
                current = []
            current.append(
                {"role": "usuario", "text": line[len("USUARIO:") :].strip()}  # noqa
            )

        elif line.startswith("CONSELHEIRO:"):
            current.append(
                {
                    "role": "conselheiro",
                    "text": line[len("CONSELHEIRO:") :].strip(),  # noqa
                }
            )

    if current:
        conversations.append(current)

    return conversations


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
                        "conversation": chunk.conversations,
                        "raw_text": chunk.raw_text,
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
# DJANGO COMMAND
# ==========================


class Command(BaseCommand):
    help = "Batch conversational PDF ingestion for RAG (Ollama + Mistral)"

    def add_arguments(self, parser):
        parser.add_argument("--chunk-size", type=int, default=900)
        parser.add_argument("--ollama-url", default="http://localhost:11434")

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

            page_chunk_counter: dict[int, int] = {}

            for page, block in extract_semantic_blocks(pdf_path):
                if page <= 10:
                    continue

                if page not in page_chunk_counter:
                    page_chunk_counter[page] = 0

                for chunk_text in sentence_chunk(block, options["chunk_size"]):
                    if should_skip_chunk(chunk_text):
                        continue

                    ci = page_chunk_counter[page]
                    page_chunk_counter[page] += 1

                    chunk_id = f"{source}:p{page}:c{ci}"

                    # Skip if chunk already exists (idempotent processing)
                    if RagChunk.objects.filter(id=chunk_id).exists():
                        self.stdout.write(
                            self.style.WARNING(f"Chunk already exists, skipping: {chunk_id}")
                        )
                        continue

                    conversations, embedding = ollama_generate_conversations(
                        chunk_text,
                        options["ollama_url"],
                    )

                    if not conversations:
                        continue

                    conversation_text = " ".join(
                        turn["text"]
                        for convo in conversations
                        for turn in convo
                        if turn.get("text")
                    )

                    if not conversation_text.strip():
                        continue

                    RagChunk(
                        id=chunk_id,
                        text=conversation_text,
                        raw_text=chunk_text,
                        conversations=conversations,
                        source=source,
                        page=page,
                        chunk_index=ci,
                        type="conversations",
                        embedding=embedding,
                    ).save()

                    self.stdout.write(self.style.SUCCESS("Chunk saved: " + chunk_id))

        self.stdout.write(self.style.SUCCESS("All PDFs processed successfully"))
