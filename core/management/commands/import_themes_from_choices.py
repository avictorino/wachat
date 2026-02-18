from django.core.management.base import BaseCommand

from core.models import ThemeV2
from core.theme_prompt_generation import build_theme_prompt_partial
from core.themes import THEME_CHOICES


class Command(BaseCommand):
    help = (
        "Importa ThemeV2 a partir de THEME_CHOICES e gera prompt parcial com OpenAI "
        "para cada tema."
    )

    def handle(self, *args, **options):
        created_count = 0
        updated_count = 0
        for theme_id, theme_name in THEME_CHOICES:
            prompt = build_theme_prompt_partial(theme_name=theme_name)
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
