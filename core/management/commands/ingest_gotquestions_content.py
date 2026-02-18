import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from django.core.management.base import BaseCommand, CommandError
from openai import OpenAI

from core.models import RagChunk, ThemeV2
from services.crawlers.gotquestions_marriage_crawler import (
    GotQuestionsMarriageCrawler,
    QuestionAnswerPair,
)

LOGGER = logging.getLogger(__name__)

SOURCE_NAME = "gotquestions_content"
EMBEDDING_MODEL = "text-embedding-3-large"
SUMMARY_TRIGGER_CHARACTERS = 4000
SUMMARY_MAX_WORDS = 1500


@dataclass(frozen=True)
class ProcessedPair:
    article_title: str
    article_url: str
    question_original: str
    answer_original: str
    question_pt: str
    answer_pt: str
    summary_pt: Optional[str]
    final_answer_pt: str
    final_text_for_embedding: str
    verses: List[str]
    theme_id: str
    theme_confidence: float


class RagChunkProcessor:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def translate_and_summarize(
        self, pair: QuestionAnswerPair
    ) -> Tuple[str, str, Optional[str], str]:
        translation_prompt = (
            "Traduza para português brasileiro o par pergunta/resposta teológico abaixo.\\n"
            "Preserve fidelidade bíblica e doutrinária.\\n"
            'Retorne SOMENTE JSON neste formato: {"question_pt":"...","answer_pt":"..."}.\\n'
            "Sem texto fora do JSON.\\n\\n"
            f"Pergunta: {pair.question}\\n\\n"
            f"Resposta: {pair.answer}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": translation_prompt}],
            response_format={"type": "json_object"},
            reasoning_effort="low",
            max_completion_tokens=5000,
            timeout=60,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Empty translation response.")

        translated = json.loads(content)
        question_pt = str(translated.get("question_pt", "")).strip()
        answer_pt = str(translated.get("answer_pt", "")).strip()
        if not question_pt:
            raise RuntimeError("Translation JSON missing 'question_pt'.")
        if not answer_pt:
            raise RuntimeError("Translation JSON missing 'answer_pt'.")

        summary_pt = None
        final_answer_pt = answer_pt

        if len(pair.answer) > SUMMARY_TRIGGER_CHARACTERS:
            summary_prompt = (
                "Resuma o texto teológico abaixo em português brasileiro.\\n"
                f"Máximo de {SUMMARY_MAX_WORDS} palavras.\\n"
                "Mantenha fidelidade teológica e não altere posicionamento doutrinário.\\n"
                "Retorne apenas o resumo.\\n\\n"
                f"Texto: {answer_pt}"
            )
            summary_pt = self._chat_completion(summary_prompt, max_tokens=3500).strip()
            if not summary_pt:
                raise RuntimeError("Empty summary response.")
            final_answer_pt = summary_pt

        return question_pt, answer_pt, summary_pt, final_answer_pt

    def extract_theme(
        self,
        text_pt: str,
        allowed_themes: Dict[str, str],
    ) -> Tuple[str, float]:
        if not allowed_themes:
            raise RuntimeError("No ThemeV2 records found for theme extraction.")

        choices_text = "\\n".join(
            [
                f"- {theme_id}: {name}"
                for theme_id, name in sorted(allowed_themes.items())
            ]
        )

        prompt = (
            "Classifique o tema predominante do conteúdo abaixo usando APENAS um theme_id da lista permitida.\\n"
            "Retorne SOMENTE JSON válido no formato:\\n"
            '{"theme_id":"id_existente","confidence":0-1}\\n'
            "Sem texto adicional.\\n\\n"
            f"Temas permitidos:\\n{choices_text}\\n\\n"
            f"Conteúdo:\\n{text_pt}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            reasoning_effort="low",
            max_completion_tokens=300,
            timeout=60,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Theme extraction returned empty response.")

        parsed = json.loads(content)
        theme_id = str(parsed.get("theme_id", "")).strip()
        confidence = float(parsed.get("confidence"))

        if theme_id not in allowed_themes:
            raise RuntimeError(f"Invalid theme_id returned by LLM: '{theme_id}'.")
        if confidence < 0 or confidence > 1:
            raise RuntimeError("Theme confidence must be between 0 and 1.")

        return theme_id, confidence

    def generate_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        embedding = response.data[0].embedding
        if not isinstance(embedding, list):
            raise RuntimeError("Invalid embedding returned by OpenAI.")
        return embedding

    def _chat_completion(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
            max_completion_tokens=max_tokens,
            timeout=60,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("OpenAI returned invalid completion content type.")
        return content


class RagChunkService:
    def save_pair(
        self,
        pair: ProcessedPair,
        embedding: List[float],
        article_position: int,
        pair_index: int,
    ) -> str:
        article_digest = hashlib.sha1(
            pair.article_url.strip().encode("utf-8")
        ).hexdigest()[:16]
        chunk_id = self._build_chunk_id(SOURCE_NAME, article_digest, pair_index)

        conversations = [
            {"role": "source_url", "text": pair.article_url},
            {"role": "article_title", "text": pair.article_title},
            {"role": "question_original", "text": pair.question_original},
            {"role": "answer_original", "text": pair.answer_original},
            {"role": "question_pt", "text": pair.question_pt},
            {"role": "answer_pt", "text": pair.answer_pt},
            {"role": "summary_pt", "text": pair.summary_pt or ""},
            {
                "role": "verses_json",
                "text": json.dumps(pair.verses, ensure_ascii=False),
            },
            {"role": "theme_confidence", "text": f"{pair.theme_confidence:.4f}"},
        ]

        RagChunk.objects.update_or_create(
            id=chunk_id,
            defaults={
                "source": SOURCE_NAME,
                "page": article_position,
                "chunk_index": pair_index,
                "raw_text": f"Question: {pair.question_original}\\n\\nAnswer: {pair.answer_original}",
                "conversations": conversations,
                "text": pair.final_text_for_embedding,
                "embedding": embedding,
                "type": "content",
                "theme_id": pair.theme_id,
            },
        )
        return chunk_id

    def delete_stale_rows(
        self, article_url: str, persisted_chunk_ids: List[str]
    ) -> None:
        article_digest = hashlib.sha1(article_url.strip().encode("utf-8")).hexdigest()[
            :16
        ]
        (
            RagChunk.objects.filter(id__startswith=f"{SOURCE_NAME}:{article_digest}:q")
            .exclude(id__in=persisted_chunk_ids)
            .delete()
        )

    def _build_chunk_id(self, source: str, article_digest: str, pair_index: int) -> str:
        return f"{source}:{article_digest}:q{pair_index}"


class Command(BaseCommand):
    help = (
        "Realiza ingestão completa do índice content.html do GotQuestions "
        "em RagChunk por par pergunta/resposta."
    )

    def handle(self, *args, **options):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_model = os.environ.get("OPENAI_MODEL")
        if not openai_api_key:
            raise CommandError("OPENAI_API_KEY is required.")
        if not openai_model:
            raise CommandError("OPENAI_MODEL is required.")

        allowed_themes = dict(ThemeV2.objects.values_list("id", "name"))
        if not allowed_themes:
            raise CommandError(
                "ThemeV2 table is empty. Configure themes before ingestion."
            )

        crawler = GotQuestionsMarriageCrawler()
        processor = RagChunkProcessor(OpenAI(api_key=openai_api_key), openai_model)
        service = RagChunkService()

        articles_processed = 0
        pair_rows_persisted = 0
        failures = 0

        LOGGER.info("event=ingest_start source=%s", SOURCE_NAME)

        article_links = crawler.crawl_article_links()
        self.stdout.write(f"Artigos encontrados: {len(article_links)}")

        for article_position, article_url in enumerate(article_links, start=1):
            try:
                article_html = crawler.fetch_article(article_url)
                parsed_article = crawler.parse_article(article_url, article_html)
            except Exception as exc:
                failures += 1
                LOGGER.exception(
                    "event=article_failed stage=crawl index=%s total=%s url=%s error=%s",
                    article_position,
                    len(article_links),
                    article_url,
                    str(exc),
                )
                self.stderr.write(
                    f"[{article_position}/{len(article_links)}] FAIL CRAWL {article_url} | error={exc}"
                )
                continue

            persisted_ids_for_article: List[str] = []
            successful_pairs = 0

            for pair_index, pair in enumerate(parsed_article.pairs, start=1):
                try:
                    (
                        question_pt,
                        answer_pt,
                        summary_pt,
                        final_answer_pt,
                    ) = processor.translate_and_summarize(pair)
                    final_text = (
                        f"Pergunta: {question_pt}\\n\\n"
                        f"Resposta:\\n{final_answer_pt}"
                    )

                    theme_id, confidence = processor.extract_theme(
                        final_text, allowed_themes
                    )
                    embedding = processor.generate_embedding(final_text)

                    processed_pair = ProcessedPair(
                        article_title=parsed_article.title,
                        article_url=parsed_article.url,
                        question_original=pair.question,
                        answer_original=pair.answer,
                        question_pt=question_pt,
                        answer_pt=answer_pt,
                        summary_pt=summary_pt,
                        final_answer_pt=final_answer_pt,
                        final_text_for_embedding=final_text,
                        verses=pair.verses,
                        theme_id=theme_id,
                        theme_confidence=confidence,
                    )

                    chunk_id = service.save_pair(
                        pair=processed_pair,
                        embedding=embedding,
                        article_position=article_position,
                        pair_index=pair_index,
                    )
                    persisted_ids_for_article.append(chunk_id)
                    successful_pairs += 1
                    pair_rows_persisted += 1

                    LOGGER.info(
                        "event=pair_processed article_index=%s pair_index=%s url=%s theme_id=%s confidence=%.2f chunk_id=%s",
                        article_position,
                        pair_index,
                        article_url,
                        theme_id,
                        confidence,
                        chunk_id,
                    )
                except Exception as exc:
                    failures += 1
                    LOGGER.exception(
                        "event=pair_failed article_index=%s pair_index=%s url=%s error=%s",
                        article_position,
                        pair_index,
                        article_url,
                        str(exc),
                    )
                    self.stderr.write(
                        f"[{article_position}/{len(article_links)}] FAIL PAIR {pair_index} {article_url} | error={exc}"
                    )
                    continue

            service.delete_stale_rows(parsed_article.url, persisted_ids_for_article)

            if successful_pairs > 0:
                articles_processed += 1

            self.stdout.write(
                f"[{article_position}/{len(article_links)}] OK {article_url} | pairs={successful_pairs}/{len(parsed_article.pairs)}"
            )

        self.stdout.write(f"Artigos processados: {articles_processed}")
        self.stdout.write(f"Linhas persistidas (pares): {pair_rows_persisted}")
        self.stdout.write(f"Falhas: {failures}")
        self.stdout.write(self.style.SUCCESS("Ingestão finalizada."))
