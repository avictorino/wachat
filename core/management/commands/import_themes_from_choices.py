import os

import requests
from django.core.management.base import BaseCommand, CommandError

from core.models import ThemeV2
from core.themes import THEME_CHOICES

REQUEST_TIMEOUT_SECONDS = 120
THEME_IMPORT_MODEL = "llama3:8b"


def _generate_theme_prompt_partial(
    *,
    ollama_url: str,
    model: str,
    theme_id: str,
    theme_name: str,
) -> str:
    prompt = (
        "Você está gerando um BLOCO PARCIAL DE CONTROLE TEMÁTICO para runtime de um chatbot.\n"
        "Retorne apenas texto plano curto, sem markdown e sem explicações extras.\n"
        "Objetivo: orientar o comportamento da próxima resposta quando este tema estiver ativo.\n"
        "Inclua:\n"
        "- 1 linha de estado emocional do tema\n"
        "- 3 proibições objetivas\n"
        "- 3 exigências operacionais para a próxima resposta\n"
        "- 4 Resultado em Portugues brasileiro\n"
        "Seja direto e acionável.\n\n"
        f"Tema id: {theme_id}\n"
        f"Tema nome: {theme_name}\n"
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
    content = str(payload.get("response", "")).strip()
    if not content:
        raise RuntimeError(f"Ollama returned empty theme prompt for '{theme_id}'.")
    return content


class Command(BaseCommand):
    help = (
        "Importa ThemeV2 a partir de THEME_CHOICES e gera prompt parcial com Ollama "
        "para cada tema."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--ollama-url",
            required=False,
            help="Base URL do Ollama. Se omitido, usa OLLAMA_BASE_URL do ambiente.",
        )

    def handle(self, *args, **options):
        ollama_url = options.get("ollama_url") or os.environ.get("OLLAMA_BASE_URL")
        if not ollama_url:
            raise CommandError("Variável OLLAMA_BASE_URL é obrigatória.")

        created_count = 0
        updated_count = 0
        for theme_id, theme_name in THEME_CHOICES:
            prompt = _generate_theme_prompt_partial(
                ollama_url=ollama_url,
                model=THEME_IMPORT_MODEL,
                theme_id=theme_id,
                theme_name=theme_name,
            )
            _, created = ThemeV2.objects.update_or_create(
                id=theme_id,
                defaults={
                    "name": theme_name,
                    "prompt": prompt,
                },
            )
            if created:
                created_count += 1
            else:
                updated_count += 1

            self.stdout.write(f"theme={theme_id} persisted (created={created})")

        self.stdout.write(
            self.style.SUCCESS(
                f"Importação concluída. created={created_count} updated={updated_count}"
            )
        )
