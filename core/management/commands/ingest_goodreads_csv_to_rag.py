import csv
import hashlib
import os
from typing import Dict

import requests
from django.core.management.base import BaseCommand, CommandError
from openai import OpenAI

from core.models import RagChunk, ThemeV2

THEME_RELACIONAMENTO = "relacionamento"
REQUEST_TIMEOUT_SECONDS = 60
EMBEDDING_MODEL = "text-embedding-3-large"

"""
python3 manage.py ingest_goodreads_csv_to_rag \
  --csv-path ./relationships_quotes.csv \
  --ollama-url http://localhost:11434 \
  --translate-model llama3:8b \
  --source goodreads_relationships
"""


def _build_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]
    return f"{source}:p{page}:c{chunk_index}:{digest}"


def _translate_to_pt_br(
    *,
    ollama_url: str,
    model: str,
    text: str,
) -> str:
    prompt = (
        "Traduza para português brasileiro mantendo o sentido original.\n"
        "Retorne apenas a tradução, sem comentários.\n\n"
        f'Texto: "{text}"'
    )
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    translated = str(payload.get("response", "")).strip()
    if not translated:
        raise CommandError("Ollama translation returned empty content.")
    return translated


def _embed_text(*, client: OpenAI, text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    if not isinstance(embedding, list):
        raise CommandError("OpenAI embeddings response is invalid.")
    return embedding


class Command(BaseCommand):
    help = (
        "Import Goodreads quotes CSV into RagChunk with theme='relacionamento', "
        "translating to pt-BR and generating embeddings via OpenAI."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv-path",
            required=True,
            help="Absolute or relative path to the CSV generated from Goodreads scraping.",
        )
        parser.add_argument(
            "--ollama-url",
            required=True,
            help="Base URL for local Ollama (example: http://localhost:11434).",
        )
        parser.add_argument(
            "--translate-model",
            required=True,
            help="Ollama model name used for translation to pt-BR.",
        )
        parser.add_argument(
            "--source",
            required=True,
            help="Value used in RagChunk.source (example: goodreads_relationships).",
        )
        parser.add_argument(
            "--chunk-type",
            default="content",
            choices=["conversation", "behavior", "content"],
            help="RagChunk.type to persist.",
        )

    def handle(self, *args, **options):
        if not ThemeV2.objects.filter(id=THEME_RELACIONAMENTO).exists():
            raise CommandError(
                "Theme 'relacionamento' is not configured in ThemeV2 table."
            )

        csv_path = str(options["csv_path"]).strip()
        ollama_url = str(options["ollama_url"]).strip()
        translate_model = str(options["translate_model"]).strip()
        source = str(options["source"]).strip()
        chunk_type = str(options["chunk_type"]).strip()
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not csv_path:
            raise CommandError("--csv-path is required.")
        if not ollama_url:
            raise CommandError("--ollama-url is required.")
        if not translate_model:
            raise CommandError("--translate-model is required.")
        if not source:
            raise CommandError("--source is required.")
        if not openai_api_key:
            raise CommandError("OPENAI_API_KEY is required.")

        openai_client = OpenAI(api_key=openai_api_key)

        page_counters: Dict[int, int] = {}
        total_created = 0
        total_updated = 0
        total_rows = 0

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            expected_columns = {"text", "author", "page", "url"}
            missing_columns = expected_columns.difference(reader.fieldnames or [])
            if missing_columns:
                raise CommandError(
                    f"CSV is missing required columns: {sorted(missing_columns)}"
                )

            for row in reader:
                total_rows += 1
                raw_text = str(row.get("text", "")).strip()
                author = str(row.get("author", "")).strip()
                page_raw = str(row.get("page", "")).strip()
                url = str(row.get("url", "")).strip()

                if not raw_text:
                    raise CommandError(f"Row {total_rows}: column 'text' is empty.")
                if not author:
                    raise CommandError(f"Row {total_rows}: column 'author' is empty.")
                if not page_raw:
                    raise CommandError(f"Row {total_rows}: column 'page' is empty.")
                if not url:
                    raise CommandError(f"Row {total_rows}: column 'url' is empty.")

                try:
                    page = int(page_raw)
                except ValueError as exc:
                    raise CommandError(
                        f"Row {total_rows}: invalid page '{page_raw}'."
                    ) from exc

                page_counters[page] = page_counters.get(page, 0) + 1
                chunk_index = page_counters[page]

                translated_text = _translate_to_pt_br(
                    ollama_url=ollama_url,
                    model=translate_model,
                    text=raw_text,
                )
                rag_text = f"{translated_text} (Autor: {author})"
                embedding = _embed_text(
                    client=openai_client,
                    text=rag_text,
                )

                chunk_id = _build_chunk_id(
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                    text=rag_text,
                )
                conversations = [
                    {"role": "quote_original", "text": raw_text},
                    {"role": "quote_translated_pt_br", "text": translated_text},
                    {"role": "author", "text": author},
                    {"role": "source_url", "text": url},
                ]

                chunk, created = RagChunk.objects.update_or_create(
                    id=chunk_id,
                    defaults={
                        "source": source,
                        "page": page,
                        "chunk_index": chunk_index,
                        "raw_text": raw_text,
                        "conversations": conversations,
                        "text": rag_text,
                        "embedding": embedding,
                        "type": chunk_type,
                        "theme_id": THEME_RELACIONAMENTO,
                    },
                )
                if created:
                    total_created += 1
                else:
                    total_updated += 1

                self.stdout.write(
                    f"[{total_rows}] persisted chunk={chunk.id} page={page} index={chunk_index}"
                )

        self.stdout.write(
            self.style.SUCCESS(
                "Done. rows=%s created=%s updated=%s theme=%s"
                % (
                    total_rows,
                    total_created,
                    total_updated,
                    THEME_RELACIONAMENTO,
                )
            )
        )
